#!/usr/bin/env python
'''
FullScheduler_ortoolkit class for co-scheduling reservations
across multiple resources using time-slicing and an integer program.

Because time is discretized into time slices, this scheduler requires
information about how to generate the slices, so its signature has one
more argument than usual.

This implementation uses a SPARSE matrix representation and the ortoolkit solver
which can be configured to use Gurobi, GLPK, or CBC algorithms.

Author: Jason Eastman (jeastman@lcogt.net)
January 2014
'''

from datetime import timedelta,datetime

from kernel.slicedipscheduler_v2 import SlicedIPScheduler_v2
#from adaptive_scheduler.utils import timeit, metric_timer, SendMetricMixin
from kernel.reservation import OptimizationType

from ortools.sat.python import cp_model
from collections import defaultdict

import logging
import math
import os

logger = logging.getLogger(__name__)

ALGORITHMS = {
    'CBC': 'CBC_MIXED_INTEGER_PROGRAMMING',
    'GUROBI': 'GUROBI_MIXED_INTEGER_PROGRAMMING',
    'GLPK': 'GLPK_MIXED_INTEGER_PROGRAMMING',
    'SCIP': 'SCIP_MIXED_INTEGER_PROGRAMMING',
    'GLOP': 'GLOP_LINEAR_PROGRAMMING',
    'BOP': 'BOP_INTEGER_PROGRAMMING'
}


FALLBACK_ALGORITHM = ALGORITHMS[os.getenv('KERNEL_FALLBACK_ALGORITHM', 'SCIP')]

class Result(object):
    pass


class FullScheduler_ortoolkit(SlicedIPScheduler_v2):
    """ Performs scheduling using an algorithm from ORToolkit
    """
    #@metric_timer('kernel.init')
    def __init__(self, kernel, compound_reservation_list, 
            globally_possible_windows_dict, 
            contractual_obligation_list, 
            slice_size_seconds, mip_gap, warm_starts, kernel_params=''):
        super().__init__(compound_reservation_list, globally_possible_windows_dict, 
                         contractual_obligation_list, slice_size_seconds)
        self.schedulerIDstring = 'SlicedIPSchedulerSparse'
        self.kernel = kernel
        self.mip_gap = mip_gap
        self.warm_starts = warm_starts
        self.kernel_params = kernel_params

        self.algorithm = ALGORITHMS[kernel.upper()]

    # A stub to get the RA/dec by request ID
    # (REQUIRED FOR AIRMASS OPTIMIZATION)
    def get_target_coords_by_reqID(self, reqID):
        #print("FullScheduler_ortoolkit::get_target_coords_by_reqID")
        return 0.0, 0.0

    # A stub to get the lat/lon by resource
    # (REQUIRED FOR AIRMASS OPTIMIZATION)
    def get_earth_coords_by_resource(self, resource):
        #print("FullScheduler_ortoolkit::get_target_coords_by_resource")
        return 0.0, 0.0

    # A stub to translate winidx to UTC date/time
    # (REQUIRED FOR AIRMASS OPTIMIZATION)
    def get_utc_by_winidx(self, winidx):
        #print("FullScheduler_ortoolkit::get_utc_by_winidx")
        reqID = 0
        reservation = get_reservation_by_ID(reqID)
        start = reservation.possible_starts[winidx].internal_start
        quantum = reservation.possible_starts[winidx].all_slice_starts[-1] + \
                  self.time_slicing_dict[resource][1] - \
                  reservation.possible_starts[winidx].first_slice_start

        return self.possible_starts[winidx]

    # A stub to optimize requests by airmass
    def weight_by_airmass(self):
        #print("FullScheduler_ortoolkit::weight_by_airmass")
        return
        # for request in self.Yik:
        #     ra, dec = self.get_target_coords_by_reqID(request[0])
        #     lat, lon = self.get_earth_coords_by_resource(request[3])
        #     utc = self.get_utc_by_winidx(request[1])
        #
        #     local_hour_angle = calc_local_hour_angle(ra, lon, utc)
        #     alt = calculate_altitude(lat, dec, local_hour_angle)
        #     airmass = 1.0 / cos(pi / 2.0 - alt)
        #
        #     # map the airmass to a minimal weighting function
        #     maxairmass = 3
        #     minairmass = 1
        #     maxweight = 0.05
        #     minweight = -0.05
        #     slope = (maxweight - minweight) / (minairmass - maxairmass)
        #     intercept = maxweight - slope * minairmass
        #
        #     weight = airmass * slope + intercept
        #     request[2] = request[2] + weight

    #@timeit
    #@metric_timer('kernel.scheduling')

    def schedule_all(self, timelimit=0):
        # First, build the necessary data structures
        self.build_data_structures()

        model = cp_model.CpModel()
        
        # Create variables
        is_scheduled = {}
        start_times = {}
        for idx, yik in enumerate(self.Yik):
            resID, start_idx, priority, resource, _ = yik
            is_scheduled[idx] = model.NewBoolVar(f'is_scheduled_{resID}_{start_idx}')
            
            # Convert the upper bound to an integer
            upper_bound = math.ceil(self.globally_possible_windows_dict[resource].get_total_time().total_seconds())
            start_times[idx] = model.NewIntVar(0, upper_bound, f'start_time_{resID}_{start_idx}')

        # Constraint: One-of (eq 5)
        for i, oneof in enumerate(self.oneof_constraints):
            constraint_vars = []
            for r in oneof:
                constraint_vars.extend(is_scheduled[idx] for idx in r.Yik_entries)
                r.skip_constraint2 = True  # Set the flag as in the old implementation
            model.Add(sum(constraint_vars) <= 1).WithName(f'oneof_constraint_{i}')

        # Constraint: One-of (eq 5) "simplified" version:
        #for oneof in self.oneof_constraints:
        #    model.Add(sum(is_scheduled[idx] for r in oneof for idx in r.Yik_entries) <= 1)

        # Constraint: And (all or nothing) (eq 6)
        for andconstraint in self.and_constraints:
            and_var = model.NewBoolVar(f'and_var_{id(andconstraint)}')
            for r in andconstraint:
                for idx in r.Yik_entries:
                    model.Add(is_scheduled[idx] == and_var)

        # Constraint: No more than one request should be scheduled in each (timeslice, resource) (eq 3)
        for s in sorted(self.aikt.keys()):
            model.Add(sum(is_scheduled[idx] for idx in self.aikt[s]) <= 1).WithName(f'one_per_slice_constraint_{s}')

        # Constraint: No request should be scheduled more than once (eq 2)
        for r in self.reservation_list:
            if not hasattr(r, 'skip_constraint2'):
                model.Add(sum(is_scheduled[idx] for idx in r.Yik_entries) <= 1).WithName(f'one_per_reqid_constraint_{r.get_ID()}')

        # Add time window constraints
        #for idx, yik in enumerate(self.Yik):
        #    resID, start_idx, _, resource, _ = yik
        #    reservation = self.get_reservation_by_ID(resID)
        #    start = reservation.possible_starts[start_idx].internal_start
        #    end = start + timedelta(seconds=reservation.duration)
        #    
        #    model.Add(start_times[idx] == int(start.timestamp())).OnlyEnforceIf(is_scheduled[idx])
        #    model.Add(start_times[idx] + reservation.duration <= int(end.timestamp())).OnlyEnforceIf(is_scheduled[idx])

        # Objective: Maximize the merit functions of all scheduled requests (eq 1)
        objective = sum(self.Yik[idx][2] * is_scheduled[idx] for idx in range(len(self.Yik)))
#        objective = sum((self.Yik[idx][2] * is_scheduled[idx]) / self.get_reservation_by_ID(self.Yik[idx][0]).duration 
#                for idx in range(len(self.Yik)))
        model.Maximize(objective)

        # Create a solver and solve the model
        solver = cp_model.CpSolver()
        if timelimit > 0:
            solver.parameters.max_time_in_seconds = timelimit

        # Set other parameters if needed
        solver.parameters.log_search_progress = False
        #solver.parameters.log_search_progress = True

        status = solver.Solve(model)

        # Return the optimally-scheduled windows
        #r = Result()
        #r.xf = []
        #for request, winidx, priority, resource, isScheduled in requestLocations:
        #    r.xf.append(isScheduled.SolutionValue())
        #    if isScheduled.SolutionValue() > 0:
        #        print(request, winidx, priority, resource, isScheduled.SolutionValue())
        #logger.warn("Set SolutionValues of isScheduled")

        #return self.unpack_result(r)

        # Process the results
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            r = Result()
            r.xf = []
            for idx in range(len(self.Yik)):
                value = solver.BooleanValue(is_scheduled[idx])
                r.xf.append(value)
                if value:
                    resID, start_idx, _, resource, _ = self.Yik[idx]
                    reservation = self.get_reservation_by_ID(resID)
                    start = reservation.possible_starts[start_idx].internal_start
                    quantum = timedelta(seconds=reservation.duration)
                    reservation.schedule(start, quantum.total_seconds(), resource, self.schedulerIDstring)
                    self.commit_reservation_to_schedule(reservation)

            return self.schedule_dict
        else:
            print(f"Solver status: {solver.StatusName()}")
            return {}

    # Other methods remain the same...

    def ex4_schedule_all(self, timelimit=0):
        # First, build the necessary data structures
        self.build_data_structures()

        model = cp_model.CpModel()
        
        # Create variables
        is_scheduled = {}
        start_times = {}
        for idx, yik in enumerate(self.Yik):
            resID, start_idx, priority, resource, _ = yik
            is_scheduled[idx] = model.NewBoolVar(f'is_scheduled_{resID}_{start_idx}')
            start_times[idx] = model.NewIntVar(0, self.globally_possible_windows_dict[resource].get_total_time().total_seconds(), f'start_time_{resID}_{start_idx}')

        # Constraint: One-of (eq 5)
        for oneof in self.oneof_constraints:
            model.Add(sum(is_scheduled[idx] for r in oneof for idx in r.Yik_entries) <= 1)

        # Constraint: And (all or nothing) (eq 6)
        for andconstraint in self.and_constraints:
            and_var = model.NewBoolVar(f'and_var_{id(andconstraint)}')
            for r in andconstraint:
                for idx in r.Yik_entries:
                    model.Add(is_scheduled[idx] == and_var)

        # Constraint: No more than one request should be scheduled in each (timeslice, resource) (eq 3)
        for s in sorted(self.aikt.keys()):
            model.Add(sum(is_scheduled[idx] for idx in self.aikt[s]) <= 1)

        # Constraint: No request should be scheduled more than once (eq 2)
        for r in self.reservation_list:
            if not hasattr(r, 'skip_constraint2'):
                model.Add(sum(is_scheduled[idx] for idx in r.Yik_entries) <= 1)

        # Add time window constraints
        for idx, yik in enumerate(self.Yik):
            resID, start_idx, _, resource, _ = yik
            reservation = self.get_reservation_by_ID(resID)
            start = reservation.possible_starts[start_idx].internal_start
            end = start + timedelta(seconds=reservation.duration)
            
            model.Add(start_times[idx] == int(start.timestamp())).OnlyEnforceIf(is_scheduled[idx])
            model.Add(start_times[idx] + reservation.duration <= int(end.timestamp())).OnlyEnforceIf(is_scheduled[idx])

        # Objective: Maximize the merit functions of all scheduled requests (eq 1)
        objective = sum(self.Yik[idx][2] * is_scheduled[idx] for idx in range(len(self.Yik)))
        model.Maximize(objective)

        # Create a solver and solve the model
        solver = cp_model.CpSolver()
        if timelimit > 0:
            solver.parameters.max_time_in_seconds = timelimit

        # Set other parameters if needed
        solver.parameters.log_search_progress = True

        status = solver.Solve(model)

        # Process the results
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            r = Result()
            r.xf = []
            for idx in range(len(self.Yik)):
                value = solver.BooleanValue(is_scheduled[idx])
                r.xf.append(value)
                if value:
                    resID, start_idx, _, resource, _ = self.Yik[idx]
                    reservation = self.get_reservation_by_ID(resID)
                    start = reservation.possible_starts[start_idx].internal_start
                    quantum = timedelta(seconds=reservation.duration)
                    reservation.schedule(start, quantum.total_seconds(), resource, self.schedulerIDstring)
                    self.commit_reservation_to_schedule(reservation)

            return self.schedule_dict
        else:
            print(f"Solver status: {solver.StatusName()}")
            return {}

    # Other methods remain the same...

    def ex3_schedule_all(self, timelimit=0):
        model = cp_model.CpModel()
        
        # Create variables and build a mapping from resID to indices
        is_scheduled = {}
        start_times = {}
        res_id_to_indices = defaultdict(list)
        for idx, yik in enumerate(self.Yik):
            resID, start_idx, priority, resource = yik[:4]
            is_scheduled[idx] = model.NewBoolVar(f'is_scheduled_{resID}_{start_idx}')
            start_times[idx] = model.NewIntVar(0, self.globally_possible_windows_dict[resource].get_total_time().total_seconds(), f'start_time_{resID}_{start_idx}')
            res_id_to_indices[resID].append(idx)

        # Constraint: One-of (eq 5)
        for oneof in self.oneof_constraints:
            model.Add(sum(is_scheduled[idx] for r in oneof for idx in res_id_to_indices[r.get_ID()]) <= 1)

        # Constraint: And (all or nothing) (eq 6)
        for andconstraint in self.and_constraints:
            and_var = model.NewBoolVar(f'and_var_{id(andconstraint)}')
            for r in andconstraint:
                for idx in res_id_to_indices[r.get_ID()]:
                    model.Add(is_scheduled[idx] == and_var)

        # Constraint: No more than one request should be scheduled in each (timeslice, resource) (eq 3)
        for s in sorted(self.aikt.keys()):
            model.Add(sum(is_scheduled[idx] for idx in self.aikt[s]) <= 1)

        # Constraint: No request should be scheduled more than once (eq 2)
        for r in self.reservation_list:
            if not hasattr(r, 'skip_constraint2'):
                model.Add(sum(is_scheduled[idx] for idx in res_id_to_indices[r.get_ID()]) <= 1)

        # Add time window constraints
        for idx, yik in enumerate(self.Yik):
            resID, start_idx, _, resource = yik[:4]
            reservation = self.get_reservation_by_ID(resID)
            for window in reservation.free_windows_dict[resource].toTupleList():
                window_start = int(window[0].timestamp())
                window_end = int(window[1].timestamp())
                model.Add(start_times[idx] >= window_start).OnlyEnforceIf(is_scheduled[idx])
                model.Add(start_times[idx] + reservation.duration <= window_end).OnlyEnforceIf(is_scheduled[idx])

        # Objective: Maximize the merit functions of all scheduled requests (eq 1)
        objective = sum(self.Yik[idx][2] * is_scheduled[idx] for idx in range(len(self.Yik)))
        model.Maximize(objective)

        # Create a solver and solve the model
        solver = cp_model.CpSolver()
        if timelimit > 0:
            solver.parameters.max_time_in_seconds = timelimit

        # Set other parameters if needed
        solver.parameters.log_search_progress = True

        status = solver.Solve(model)

        # Process the results
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            r = Result()
            r.xf = []
            for idx in range(len(self.Yik)):
                value = solver.BooleanValue(is_scheduled[idx])
                r.xf.append(value)
                if value:
                    resID, start_idx, _, resource = self.Yik[idx][:4]
                    reservation = self.get_reservation_by_ID(resID)
                    start = reservation.possible_starts[start_idx].internal_start
                    quantum = timedelta(seconds=reservation.duration)
                    reservation.schedule(start, quantum.total_seconds(), resource, self.schedulerIDstring)
                    self.commit_reservation_to_schedule(reservation)

            return self.schedule_dict
        else:
            print(f"Solver status: {solver.StatusName()}")
            return {}

    # Other methods remain the same...
    def ex2_schedule_all(self, timelimit=0):
        model = cp_model.CpModel()
        
        # Create variables
        is_scheduled = {}
        for idx, yik in enumerate(self.Yik):
            resID, start_idx, priority, resource = yik[:4]
            is_scheduled[idx] = model.NewBoolVar(f'is_scheduled_{resID}_{start_idx}')

        # Constraint: One-of (eq 5)
        for oneof in self.oneof_constraints:
            model.Add(sum(is_scheduled[idx] for idx in oneof) <= 1)

        # Constraint: And (all or nothing) (eq 6)
        for andconstraint in self.and_constraints:
            and_var = model.NewBoolVar(f'and_var_{id(andconstraint)}')
            for idx in andconstraint:
                model.Add(is_scheduled[idx] == and_var)

        # Constraint: No more than one request should be scheduled in each (timeslice, resource) (eq 3)
        for s in sorted(self.aikt.keys()):
            model.Add(sum(is_scheduled[idx] for idx in self.aikt[s]) <= 1)

        # Constraint: No request should be scheduled more than once (eq 2)
        for r in self.reservation_list:
            if not hasattr(r, 'skip_constraint2'):
                model.Add(sum(is_scheduled[idx] for idx in r.Yik_entries) <= 1)

        # Add time window constraints
        for idx, yik in enumerate(self.Yik):
            resID, start_idx, _, resource = yik[:4]
            reservation = self.get_reservation_by_ID(resID)
            for window in reservation.free_windows_dict[resource].toTupleList():
                window_start = int(window[0].timestamp())
                window_end = int(window[1].timestamp())
                model.Add(start_times[idx] >= window_start).OnlyEnforceIf(is_scheduled[idx])
                model.Add(start_times[idx] + reservation.duration <= window_end).OnlyEnforceIf(is_scheduled[idx])

        # Set the objective to maximize the sum of priorities of scheduled reservations
        objective = sum(self.Yik[idx][2] * is_scheduled[idx] for idx in range(len(self.Yik)))
        model.Maximize(objective)

        # Create a solver and solve the model
        solver = cp_model.CpSolver()
        if timelimit > 0:
            solver.parameters.max_time_in_seconds = timelimit
        
        status = solver.Solve(model)

        # Process the results
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            r = Result()
            r.xf = []
            for idx in range(len(self.Yik)):
                value = solver.Value(is_scheduled[idx])
                r.xf.append(value)
                if value == 1:
                    resID, start_idx, _, resource = self.Yik[idx][:4]
                    reservation = self.get_reservation_by_ID(resID)
                    start = reservation.possible_starts[start_idx].internal_start
                    quantum = timedelta(seconds=reservation.duration)
                    reservation.schedule(start, quantum.total_seconds(), resource, self.schedulerIDstring)
                    self.commit_reservation_to_schedule(reservation)

            return self.schedule_dict
        else:
            print(f"Solver status: {solver.StatusName()}")
            return {}

    # Other methods remain the same...

    def ex_schedule_all(self, timelimit=0):
        #print("FullScheduler_ortoolkit::schedule_all")

        if not self.reservation_list:
            return self.schedule_dict

        # populate all the windows, requests, etc
        self.build_data_structures()

        # weight the priorities in each timeslice by airmass
        self.weight_by_airmass()

        # used to save a metric for when the primary algorithm fails to instantiate
        primary_algorithm_failed = 0

        # Instantiate the ORTools solver
        try:
            #solver = pywraplp.Solver_CreateSolver(self.algorithm)
            solver = pywraplp.Solver(self.algorithm, 0)
            if not solver:
                logger.warn(f"Failed to get a valid solver for {self.kernel}.")
                logger.warn(f"Defaulting to {FALLBACK_ALGORITHM} solver")
                primary_algorithm_failed = 1
                solver = pywraplp.Solver_CreateSolver(FALLBACK_ALGORITHM)
        except Exception as e:
            logger.warn(f"Failed to create a valid solver for {self.kernel}: {repr(e)}")
            logger.warn(f"Defaulting to {FALLBACK_ALGORITHM} solver")
            primary_algorithm_failed = 1
            solver = pywraplp.Solver_CreateSolver(FALLBACK_ALGORITHM)

        #self.send_metric('primary_algorithm_failed.occurence', primary_algorithm_failed)

        # Constraint: Decision variable (isScheduled) must be binary (eq 4)
        requestLocations = []
        vars_by_req_id = defaultdict(list)
        scheduled_vars = []
        solution_hints = []
        for r in self.Yik:
            # create a lut of req_id to decision vars for building the constraints later
            # [(reqID, window idx, priority, resource, isScheduled)]

            var = solver.Var(lb=0, ub=1, integer=True, name=f"bool_var_{r[0]}_{len(scheduled_vars)}")
            #var = solver.IntVar(0, 1, name=f"bool_var_{r[0]}_{len(scheduled_vars)}")
            #solver.Add(var == 0).OnlyEnforceIf(var.Not())
            #solver.Add(var == 1).OnlyEnforceIf(var)

            #var = solver.BoolVar(name=f"bool_var_{r[0]}_{len(scheduled_vars)}")
            scheduled_vars.append(var)
            vars_by_req_id[r[0]].append((r[0], r[1], r[2], r[3], var))
            requestLocations.append((r[0], r[1], r[2], r[3], var))
            solution_hints.append(r[4])

        # The warm-start hints (not supported in older ortools)
        if self.warm_starts:
            logger.info("Using warm start solution this run")
            solver.SetHint(variables=scheduled_vars, values=solution_hints)

        # Constraint: One-of (eq 5)
        i = 0
        for oneof in self.oneof_constraints:
            match = []
            for r in oneof:
                reqid = r.get_ID()
                match.extend(vars_by_req_id[reqid])
                r.skip_constraint2 = True  # does this do what I think it does?
            nscheduled_one = solver.Sum([isScheduled for reqid, winidx, priority, resource, isScheduled in match])
            solver.Add(nscheduled_one <= 1, 'oneof_constraint_' + str(i))
            i = i + 1

        # Constraint: And (all or nothing) (eq 6)
        i = 0
        for andconstraint in self.and_constraints:
            # add decision variable that must be equal to all "and"ed blocks
            andVar = solver.BoolVar(name=f"and_var_{str(i)}")
            j = 0
            for r in andconstraint:
                reqid = r.get_ID()
                match = vars_by_req_id[reqid]
                nscheduled_and = solver.Sum([isScheduled for reqid, winidx, priority, resource, isScheduled in match])
                solver.Add(andVar == nscheduled_and, 'and_constraint_' + str(i) + "_" + str(j))
                j = j + 1
            i = i + 1

        # Constraint: No more than one request should be scheduled in each (timeslice, resource) (eq 3)
        # self.aikt.keys() indexes the requests that occupy each (timeslice, resource)
        for s in sorted(self.aikt.keys()):
            match = []
            for timeslice in self.aikt[s]:
                match.append(requestLocations[timeslice])
            nscheduled1 = solver.Sum([isScheduled for reqid, winidx, priority, resource, isScheduled in match])
            solver.Add(nscheduled1 <= 1, 'one_per_slice_constraint_' + s)

        # Constraint: No request should be scheduled more than once (eq 2)
        # skip if One-of (redundant)
        for r in self.reservation_list:
            if not hasattr(r, 'skip_constraint2'):
                reqid = r.get_ID()
                match = vars_by_req_id[reqid]
                nscheduled2 = solver.Sum([isScheduled for reqid, winidx, priority, resource, isScheduled in match])
                solver.Add(nscheduled2 <= 1, 'one_per_reqid_constraint_' + str(reqid))

        # Objective: Maximize the merit functions of all scheduled requests (eq 1);
        objective = solver.Maximize(solver.Sum(
            [isScheduled * priority for _, _, priority, _, isScheduled in requestLocations])
        )

        # impose a time limit (ms) on the solve
        if timelimit > 0:
            solver.SetTimeLimit(int(timelimit * 1000))

        # Set kernel specific parameters if they are present
        if self.kernel_params:
            solver.SetSolverSpecificParametersAsString(self.kernel_params)

        params = pywraplp.MPSolverParameters()
        # Set the tolerance for the model solution to be within 1% of what it thinks is the best solution
        params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, self.mip_gap)

        print(f"Number of variables: {solver.NumVariables()}")
        print(f"Number of constraints: {solver.NumConstraints()}")

        # Solve the model
        solver.EnableOutput()
        # Set solver parameters
        solver.SetSolverSpecificParametersAsString("mip_tolerances_integrality=1e-9")
        solver.SetSolverSpecificParametersAsString("mip_feasibility_tolerance=1e-9")

        status = solver.Solve(params)
        logger.warn("Finished solving schedule")
        #print(f"Solver status: {solver.StatusName()}")
        print(f"Objective value: {solver.Objective().Value()}")
#        print(f"Number of scheduled reservations: {sum(var.solution_value() for var in isScheduled.values())}")

        # Check solver status
        print(f"Solver status: {status}")
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print("Warning: Solver did not find an optimal or feasible solution")



        # Return the optimally-scheduled windows
        r = Result()
        r.xf = []
        for request, winidx, priority, resource, isScheduled in requestLocations:
            r.xf.append(isScheduled.SolutionValue())
            if isScheduled.SolutionValue() > 0:
                print(request, winidx, priority, resource, isScheduled.SolutionValue())
        logger.warn("Set SolutionValues of isScheduled")

        return self.unpack_result(r)


def dump_matrix_sizes(f, A, Aeq, b, beq, n_res):
    # Don't write the header if the file already exists
    import os.path
    path_to_file = 'matrix_sizes.dat'
    write_header = True
    if os.path.isfile(path_to_file):
        write_header = False

    from datetime import datetime
    date_time_fmt = '%Y-%m-%d %H:%M:%S'
    now = datetime.utcnow()
    now_str = now.strftime(date_time_fmt)

    out_fh = open(path_to_file, 'a')
    fmt_str = "%-6s %-16s %-16s %-16s %-16s %-16s %-16s %-16s %-16s %-16s %-16s %-13s\n"

    out_hdr = fmt_str % ('N_res',
                         'F_shape', 'F_size',
                         'A_shape', 'A_size',
                         'b_shape', 'b_size',
                         'lb_shape', 'lb_size',
                         'ub_shape', 'ub_size',
                         'Ran_at')
    out_str = fmt_str % (n_res,
                         f.shape, m_size(f),
                         A.shape, sm_size(A),
                         b.shape, m_size(b),
                         lb.shape, m_size(lb),
                         ub.shape, m_size(ub),
                         now_str)
    if write_header:
        out_fh.write(out_hdr)
    out_fh.write(out_str)

    out_fh.close()


def m_size(m):
    return m.nbytes * m.dtype.itemsize


def sm_size(m):
    return m.getnnz() * m.dtype.itemsize


def print_matrix_size(matrix):
    print("Matrix shape: {}".format(matrix.shape))
    print("Matrix size (bytes): {}".format(matrix.nbytes * matrix.dtype.itemsize))
    print("Matrix type: {}".format(matrix.dtype))


def print_sparse_matrix_size(matrix):
    print("Matrix shape: {}".format(matrix.shape))
    print("Matrix size (bytes): {}".format(matrix.getnnz() * matrix.dtype.itemsize))
    print("Matrix type: {}".format(matrix.dtype))
