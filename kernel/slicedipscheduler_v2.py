#!/usr/bin/env python
'''
SlicedIPScheduler class for co-scheduling reservations
across multiple resources using time-slicing and an integer program.

Because time is discretized into time slices, this scheduler requires
information about how to generate the slices, so its signature has one
more argument than usual. 

Author: Sotiria Lampoudi (slampoud@gmail.com)
'''

import math
import numpy as np
from datetime import timedelta,datetime
from kernel.scheduler import Scheduler
from kernel.reservation import OptimizationType

class PossibleStart(object):
    def __init__(self, resource, slice_starts, internal_start, airmass_coefficient):
        self.resource = resource
        self.first_slice_start = slice_starts[0]
        self.all_slice_starts = slice_starts
        self.internal_start = internal_start
        self.airmass_coefficient = airmass_coefficient

    def __lt__(self, other):
        return self.first_slice_start < other.first_slice_start

    def __eq__(self, other):
        return self.first_slice_start == self.first_slice_start

    def __gt__(self, other):
        return self.first_slice_start > self.first_slice_start


class SlicedIPScheduler_v2(Scheduler):

    def __init__(self, compound_reservation_list,
                 globally_possible_windows_dict,
                 contractual_obligation_list,
                 slice_size_seconds):
        super().__init__(compound_reservation_list,
                           globally_possible_windows_dict,
                           contractual_obligation_list)
        # time_slicing_dict is a dictionary that maps: 
        # resource-> [slice_alignment, slice_length]
        #         self.resource_list = resource_list
        self.slice_size_seconds = slice_size_seconds
        self.time_slicing_dict = {}
        # these are the structures we need for the linear programming solver
        self.Yik = []  # maps idx -> [resID, window idx, priority, resource]
        self.aikt = {}  # maps slice -> Yik idxs
        self.schedulerIDstring = 'slicedIPscheduler'
        self.hashes = set()

        # Ensure unscheduled_reservation_list is properly initialized
        self.unscheduled_reservation_list = list(self.reservation_list)

        for r in self.resource_list:
            self.time_slicing_dict[r] = [0, self.slice_size_seconds]

    def hash_slice(self, start, resource, slice_length):
        string = "resource_" + resource + "_start_" + repr(start) + "_length_" + repr(slice_length)
        exists = string in self.hashes
        self.hashes.add(string)
        return string, exists

    def unhash_slice(self, mystr):
        #print("SlicedIPScheduler_v2::unhash_slice")
        l = mystr.split("_")
        return [l[1], int(l[3]), int(l[5])]

    def build_data_structures(self):
        # first we need to build up the list of discretized slices that each
        # reservation can begin in. These are represented as attributes
        # that get attached to the reservation object. 
        # The new attributes are:
        # slices_dict
        # internal_starts_dict
        # and the dicts are keyed by resource.
        # the description of slices and internal starts is in intervals.py
        for r in self.reservation_list:
            r.possible_starts = []
            r.Yik_entries = []
            
            # Calculate all possible starts for all resources
            for resource in sorted(r.free_windows_dict.keys()):
                r.possible_starts.extend(self.get_slices(
                    r.free_windows_dict[resource], 
                    resource, 
                    r.duration, 
                    r.request
                ))
            # reorder PossibleStarts
            r.possible_starts.sort()
            
            # Calculate priorities and build Yik entries
            # Use local indexing, will adjust global indices later
            w_idx = 0
            for ps in r.possible_starts:
                Yik_idx = len(self.Yik)
                r.Yik_entries.append(Yik_idx)
                # set the initial warm start solution
                scheduled = 0
                if r.previous_solution_reservation and r.previous_solution_reservation.scheduled_start == ps.internal_start and r.previous_solution_reservation.scheduled_resource == ps.resource:
                    scheduled = 1
                # now w_idx is the index into r.possible_starts, which have
                # been reordered by time.

                # Calculate priority
                if r.request and r.request.optimization_type == OptimizationType.AIRMASS: #***
                    # Apply the airmass coefficient into the priority
                    base_priority = r.priority / ps.airmass_coefficient
                else:
                    # Add the earlier window optimization priority factor to the effective priority
                    base_priority = r.priority + (10 / (w_idx + 1.0))

                base_priority /= r.duration # fair handling of long requests

                final_priority = base_priority * r.request.calculate_time_based_priority(ps.internal_start)

                self.Yik.append([r.resID, w_idx, final_priority, ps.resource, scheduled])
                w_idx += 1
                # build aikt
                # Collect slice information using local index
                for s in ps.all_slice_starts:
                    key, exists = self.hash_slice(s, ps.resource, self.time_slicing_dict[ps.resource][1])
                    #                        if key in self.aikt:
                    if exists:
                        self.aikt[key].append(Yik_idx)
                    else:
                        self.aikt[key] = [Yik_idx]
#            print(f"Reservation {r.resID}: {len(r.possible_starts)} possible starts")

    #    # Return the optimally-scheduled windows
    #    r = Result()
    #    r.xf = []
    #    for request, winidx, priority, resource, isScheduled in requestLocations:
    #        r.xf.append(isScheduled.SolutionValue())
    #        print(request, winidx, priority, resource, isScheduled)
    #    logger.warn("Set SolutionValues of isScheduled")

    def unpack_result(self, r):
        idx = 0
        for value in r.xf:
            if value == 1:
                resID = self.Yik[idx][0]
                start_idx = self.Yik[idx][1]
                resource = self.Yik[idx][3]
                reservation = self.get_reservation_by_ID(resID)
                start = reservation.possible_starts[start_idx].internal_start
                quantum = reservation.possible_starts[start_idx].all_slice_starts[-1] + \
                          timedelta(seconds=self.time_slicing_dict[resource][1]) - \
                          reservation.possible_starts[start_idx].all_slice_starts[0]
                reservation.schedule(start, quantum.total_seconds(), resource, self.schedulerIDstring)
                self.commit_reservation_to_schedule(reservation)
            idx += 1
        return self.schedule_dict

    def not_unpack_result(self, r):
        #print("SlicedIPScheduler_v2::unpack_result")
        idx = 0
        for value in r.xf:
            if value == 1:
                resID = self.Yik[idx][0]
                start_idx = self.Yik[idx][1]
                resource = self.Yik[idx][3]
                reservation = self.get_reservation_by_ID(resID)
                
                # Use the internal_start for the start
                start = reservation.possible_starts[start_idx].internal_start
                
                # Use the original duration of the reservation
                requested_duration = timedelta(seconds=reservation.duration)
                
                # Calculate the end of the last available slice
                last_slice_start = reservation.possible_starts[start_idx].all_slice_starts[-1]
                slice_length = timedelta(seconds=self.time_slicing_dict[resource][1])
                last_slice_end = last_slice_start + slice_length
                
                # Ensure the reservation doesn't exceed the last available slice
                end = min(start + requested_duration, last_slice_end)
                
                # Calculate the actual quantum (which might be shorter than requested if it doesn't fit)
                quantum = end - start
                
                if quantum < requested_duration:
                    print(f"Warning: Reservation {resID} scheduled for {quantum.total_seconds()}s instead of requested {requested_duration.total_seconds()}s")
                
                reservation.schedule(start, quantum.total_seconds(), resource, self.schedulerIDstring)
                self.commit_reservation_to_schedule(reservation)
            idx += 1
        return self.schedule_dict
   
    def get_slices(self, intervals, resource, duration, request):
        #print("SlicedIPScheduler_v2::get_slices")
        ps_list = []
        if resource in self.time_slicing_dict:
            slice_alignment = self.time_slicing_dict[resource][0]
            slice_length = self.time_slicing_dict[resource][1]
            slices = []
            internal_starts = []
            for t in intervals.toDictList():
                if t['type'] == 'start':
                    if isinstance(t['time'], datetime) and isinstance(slice_alignment, int):
                        slice_alignment = datetime.fromtimestamp(slice_alignment)
                    
                    if t['time'] <= slice_alignment:
                        start = slice_alignment
                        internal_start = slice_alignment
                    else:
                        time_diff = (t['time'] - slice_alignment).total_seconds()
                        start = slice_alignment + timedelta(seconds=int(time_diff / slice_length) * slice_length)
                        internal_start = t['time']
                elif t['type'] == 'end':
                    if isinstance(t['time'], datetime) and isinstance(slice_alignment, int):
                        slice_alignment = datetime.fromtimestamp(slice_alignment)
                    
                    if t['time'] < slice_alignment:
                        continue
                    while (t['time'] - start).total_seconds() >= duration:
                        num_slices = math.ceil(duration / slice_length)
#                        #tmp = [start + timedelta(seconds=i*slice_length) for i in range((internal_start - start).seconds // slice_length + 1)]
                        tmp = [start + timedelta(seconds=i*slice_length) for i in range(num_slices)]
                        slices.append(tmp)
                        internal_starts.append(internal_start)
                        start += timedelta(seconds=slice_length)
                        internal_start = start

            if internal_starts:  # Only proceed if we have valid start times
                if request and request.optimization_type == OptimizationType.AIRMASS: #****
                    airmasses_at_times = request.get_airmasses_within_kernel_windows(resource)
                    if airmasses_at_times['times']:  # Check if we have airmass data
                        mid_times = [t + timedelta(seconds=duration/2) for t in internal_starts]
                        interpolated_airmasses = np.interp([t.timestamp() for t in mid_times], 
                                     [t.timestamp() for t in airmasses_at_times['times']], 
                                     airmasses_at_times['airmasses'])
                    #    interpolated_airmasses = np.interp([i.timestamp() for i in internal_starts], 
                    #                                       [t.timestamp() for t in airmasses_at_times['times']], 
                    #                                       airmasses_at_times['airmasses'])
                    else:
                        interpolated_airmasses = np.ones(len(internal_starts))
                   #     print("warning: returning ones1")
                else:
                    interpolated_airmasses = np.ones(len(internal_starts))
                    #print("warning: returning ones2")
                
                for idx, w in enumerate(slices):
                    ps_list.append(PossibleStart(resource, w, internal_starts[idx], interpolated_airmasses[idx]))

        return ps_list

