"""
kernel/solver.py - Simplified CP-SAT Solver
Merges the functionality of fullscheduler_cp_model.py and slicedipscheduler_v2.py
Removes all the dead code and experimental methods.
"""

import logging
import math
import numpy as np
from datetime import timedelta, datetime
from collections import defaultdict

from ortools.sat.python import cp_model

from kernel.scheduler_base import BaseScheduler
from kernel.reservation import OptimizationType

logger = logging.getLogger(__name__)


class PossibleStart:
    """Represents a possible start time for a reservation."""
    
    def __init__(self, resource, slice_starts, internal_start, airmass_coeff=1.0):
        self.resource = resource
        self.first_slice_start = slice_starts[0]
        self.all_slice_starts = slice_starts
        self.internal_start = internal_start
        self.airmass_coefficient = airmass_coeff
    
    def __lt__(self, other):
        return self.first_slice_start < other.first_slice_start


class CPScheduler(BaseScheduler):
    """
    Constraint Programming scheduler using OR-Tools CP-SAT solver.
    Handles time-sliced scheduling across multiple resources.
    """
    
    def __init__(self, compound_reservation_list, globally_possible_windows_dict,
                 slice_size_seconds, timelimit=180, mip_gap=0.05):
        """
        Initialize the CP scheduler.
        
        Args:
            compound_reservation_list: List of compound reservations to schedule
            globally_possible_windows_dict: Global visibility windows per resource
            slice_size_seconds: Time slice size in seconds
            timelimit: Maximum solver time in seconds
            mip_gap: Relative optimality gap
        """
        super().__init__(
            compound_reservation_list,
            globally_possible_windows_dict,
            []  # No contractual obligations
        )
        
        self.slice_size = slice_size_seconds
        self.timelimit = timelimit
        self.mip_gap = mip_gap
        
        # Data structures for the solver
        self.Yik = []  # Maps idx -> [resID, window_idx, priority, resource]
        self.aikt = {}  # Maps slice -> list of Yik indices
        
        # Initialize time slicing for each resource
        self.time_slicing = {
            resource: [0, slice_size_seconds]
            for resource in self.resource_list
        }
    
    def schedule_all(self):
        """Run the scheduling algorithm."""
        if not self.reservation_list:
            return self.schedule_dict
        
        # Build data structures
        self._build_data_structures()
        
        # Create and solve the CP model
        model = cp_model.CpModel()
        
        # Create decision variables
        is_scheduled = self._create_variables(model)
        
        # Add constraints
        self._add_constraints(model, is_scheduled)
        
        # Set objective
        self._set_objective(model, is_scheduled)
        
        # Solve
        solver = cp_model.CpSolver()
        if self.timelimit > 0:
            solver.parameters.max_time_in_seconds = self.timelimit
        solver.parameters.log_search_progress = False
        
        status = solver.Solve(model)
        
        # Process results
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self._process_solution(solver, is_scheduled)
            logger.info(f"Solver found solution with {len(self.schedule_dict)} scheduled")
        else:
            logger.warning(f"Solver status: {solver.StatusName()}")
        
        return self.schedule_dict
    
    def _build_data_structures(self):
        """Build the data structures needed for the solver."""
        for reservation in self.reservation_list:
            reservation.possible_starts = []
            reservation.Yik_entries = []
            
            # Calculate possible starts for all resources
            for resource in sorted(reservation.free_windows_dict.keys()):
                starts = self._get_possible_starts(
                    reservation.free_windows_dict[resource],
                    resource,
                    reservation.duration,
                    reservation.request
                )
                reservation.possible_starts.extend(starts)
            
            # Sort by time
            reservation.possible_starts.sort()
            
            # Create Yik entries with priorities
            for w_idx, ps in enumerate(reservation.possible_starts):
                Yik_idx = len(self.Yik)
                reservation.Yik_entries.append(Yik_idx)
                
                # Calculate priority
                priority = self._calculate_priority(reservation, ps, w_idx)
                
                # Add to Yik
                self.Yik.append([
                    reservation.resID,
                    w_idx,
                    priority,
                    ps.resource,
                    0  # warm start value (not used currently)
                ])
                
                # Build aikt mapping
                for slice_start in ps.all_slice_starts:
                    key = self._hash_slice(slice_start, ps.resource)
                    if key not in self.aikt:
                        self.aikt[key] = []
                    self.aikt[key].append(Yik_idx)
    
    def _get_possible_starts(self, intervals, resource, duration, request):
        """Calculate possible start times within given intervals."""
        possible_starts = []
        slice_length = self.time_slicing[resource][1]
        
        for start_time, end_time in intervals.toTupleList():
            # Align to slice boundaries
            if isinstance(start_time, datetime):
                aligned_start = self._align_to_slice(start_time, slice_length)
            else:
                aligned_start = start_time
            
            # Generate all possible starts
            current = aligned_start
            while (end_time - current).total_seconds() >= duration:
                num_slices = math.ceil(duration / slice_length)
                slice_starts = [
                    current + timedelta(seconds=i * slice_length)
                    for i in range(num_slices)
                ]
                
                # Calculate airmass coefficient if needed
                airmass_coeff = 1.0
                if request and request.optimization_type == OptimizationType.AIRMASS:
                    airmass_coeff = self._get_airmass_coefficient(
                        request, resource, current, duration
                    )
                
                possible_starts.append(
                    PossibleStart(resource, slice_starts, current, airmass_coeff)
                )
                
                current += timedelta(seconds=slice_length)
        
        return possible_starts
    
    def _calculate_priority(self, reservation, possible_start, window_idx):
        """Calculate priority for a reservation at a specific start time."""
        base_priority = reservation.priority
        
        if reservation.request:
            if reservation.request.optimization_type == OptimizationType.AIRMASS:
                # Apply airmass weighting
                base_priority /= possible_start.airmass_coefficient
            else:
                # Prefer earlier windows
                base_priority += 10 / (window_idx + 1.0)
            
            # Fair handling of long requests
            base_priority /= reservation.duration
            
            # Apply time-based priority
            base_priority *= reservation.request.calculate_time_based_priority(
                possible_start.internal_start
            )
        
        return base_priority
    
    def _get_airmass_coefficient(self, request, resource, start_time, duration):
        """Get airmass coefficient for optimization."""
        if not hasattr(request, 'airmass_data'):
            return 1.0
        
        if resource not in request.airmass_data:
            return 1.0
        
        data = request.airmass_data[resource]
        if not data['times']:
            return 1.0
        
        # Find the airmass at the midpoint of the observation
        mid_time = start_time + timedelta(seconds=duration/2)
        
        # Interpolate
        times_ts = [t.timestamp() for t in data['times']]
        mid_ts = mid_time.timestamp()
        
        airmass = np.interp(mid_ts, times_ts, data['airmasses'])
        return max(airmass, 1.0)  # Ensure it's at least 1
    
    def _create_variables(self, model):
        """Create CP-SAT variables."""
        is_scheduled = {}
        
        for idx in range(len(self.Yik)):
            resID, start_idx, _, _, _ = self.Yik[idx]
            is_scheduled[idx] = model.NewBoolVar(f'sched_{resID}_{start_idx}')
        
        return is_scheduled
    
    def _add_constraints(self, model, is_scheduled):
        """Add all constraints to the model."""
        
        # One-of constraints
        for i, oneof in enumerate(self.oneof_constraints):
            constraint_vars = []
            for r in oneof:
                constraint_vars.extend(is_scheduled[idx] for idx in r.Yik_entries)
                r.skip_constraint2 = True
            model.Add(sum(constraint_vars) <= 1).WithName(f'oneof_{i}')
        
        # And constraints (all or nothing)
        for i, andconstraint in enumerate(self.and_constraints):
            and_var = model.NewBoolVar(f'and_{i}')
            for r in andconstraint:
                for idx in r.Yik_entries:
                    model.Add(is_scheduled[idx] == and_var)
        
        # No more than one reservation per time slice
        for slice_key, indices in self.aikt.items():
            model.Add(sum(is_scheduled[idx] for idx in indices) <= 1).WithName(
                f'slice_{slice_key}'
            )
        
        # Each reservation scheduled at most once
        for r in self.reservation_list:
            if not hasattr(r, 'skip_constraint2'):
                model.Add(
                    sum(is_scheduled[idx] for idx in r.Yik_entries) <= 1
                ).WithName(f'once_{r.resID}')
    
    def _set_objective(self, model, is_scheduled):
        """Set the optimization objective."""
        objective = sum(
            self.Yik[idx][2] * is_scheduled[idx]
            for idx in range(len(self.Yik))
        )
        model.Maximize(objective)
    
    def _process_solution(self, solver, is_scheduled):
        """Process the solver solution and commit scheduled reservations."""
        for idx in range(len(self.Yik)):
            if solver.BooleanValue(is_scheduled[idx]):
                resID, start_idx, _, resource, _ = self.Yik[idx]
                
                reservation = self.get_reservation_by_ID(resID)
                start = reservation.possible_starts[start_idx].internal_start
                
                # Calculate actual quantum
                slices = reservation.possible_starts[start_idx].all_slice_starts
                quantum = (slices[-1] + timedelta(seconds=self.slice_size) - 
                          slices[0]).total_seconds()
                
                # Schedule and commit
                reservation.schedule(start, quantum, resource, 'CPScheduler')
                self.commit_reservation_to_schedule(reservation)
    
    def _align_to_slice(self, time, slice_size):
        """Align a datetime to slice boundaries."""
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (time - epoch).total_seconds()
        aligned_seconds = (seconds_since_epoch // slice_size) * slice_size
        return epoch + timedelta(seconds=aligned_seconds)
    
    def _hash_slice(self, start, resource):
        """Create a hash key for a time slice."""
        if isinstance(start, datetime):
            start_str = start.isoformat()
        else:
            start_str = str(start)
        return f"{resource}_{start_str}_{self.slice_size}"

