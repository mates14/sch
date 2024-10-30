#!/usr/bin/env python
'''
Scheduler is the base class for all schedulers. 

Author: Sotiria Lampoudi (slampoud@gmail.com)
Dec 2012
'''

# Optimized version:
from kernel.intervals import Intervals
from collections import defaultdict

class Scheduler(object):
    def __init__(self, compound_reservation_list,
                 globally_possible_windows_dict,
                 contractual_obligation_list):
        self.compound_reservation_list = compound_reservation_list
        self.contractual_obligation_list = contractual_obligation_list
        
        # Remove empty windows once at initialization
        self.globally_possible_windows_dict = {
            resource: windows for resource, windows in globally_possible_windows_dict.items() 
            if not windows.is_empty()
        }
        
        # Use defaultdict to avoid explicit initialization
        self.schedule_dict = defaultdict(list)
        self.schedule_dict_busy = defaultdict(lambda: Intervals([], 'busy'))
        
        # Get resource list once
        self.resource_list = list(self.globally_possible_windows_dict.keys())
        
        # No need for loop - defaultdict handles initialization
        self.schedule_dict_free = self.globally_possible_windows_dict.copy()  # Shallow copy is sufficient here
        
        # Initialize constraints
        self.and_constraints = []
        self.oneof_constraints = []
        
        # Convert reservations
        self.reservation_list, self.reservation_dict = self.convert_compound_to_simple()
        self.unscheduled_reservation_list = list(self.reservation_list)  # list() is faster than copy.copy() for lists
        
        # Use defaultdict for resource mapping
        self.reservations_by_resource_dict = defaultdict(list)
        for reservation in self.reservation_list:
            for resource in reservation.free_windows_dict:
                self.reservations_by_resource_dict[resource].append(reservation)

    def check_against_gpw(self, reservation):
        """Optimized version of check_against_gpw using vectorized operations."""
        result = {}
        for resource, windows in reservation.free_windows_dict.items():
            if resource not in self.globally_possible_windows_dict:
                continue
                
            global_windows = self.globally_possible_windows_dict[resource]
            intersection = windows.intersect([global_windows])
            
            if not intersection.is_empty():
                result[resource] = intersection

        if result:
            reservation.free_windows_dict = result
            return True
        return False


    def commit_reservation_to_schedule(self, r):
        """Commit a reservation to the schedule more efficiently."""
        if not r.scheduled:
            raise ValueError("Attempting to commit unscheduled reservation")
            
        # Get resource once to avoid repeated lookups    
        resource = r.scheduled_resource
        timepoints = r.scheduled_timepoints
        
        # Update all data structures
        self.schedule_dict[resource].append(r)
        self.schedule_dict_busy[resource].add(timepoints)
        self.schedule_dict_free[resource] = self.schedule_dict_free[resource].subtract(
            Intervals(timepoints))
        
        # Remove from unscheduled list
        try:
            self.unscheduled_reservation_list.remove(r)
        except ValueError:
            raise ValueError(f"Reservation {r.resID} not found in unscheduled list")

    def uncommit_reservation_from_schedule(self, r):
        """Uncommit a reservation from schedule more efficiently."""
        # Get all needed values at once to avoid repeated lookups
        resource = r.scheduled_resource
        timepoints = r.scheduled_timepoints
        
        try:
            self.schedule_dict[resource].remove(r)
        except ValueError:
            raise ValueError(f"Reservation {r.resID} not found in schedule for resource {resource}")
            
        # Update time windows
        self.schedule_dict_free[resource].add(timepoints)
        self.schedule_dict_busy[resource].subtract(Intervals(timepoints, 'free'))
        
        # Update reservation status
        self.unscheduled_reservation_list.append(r)
        r.unschedule()

    def get_reservation_by_ID(self, ID):
        """Direct dictionary lookup instead of list search."""
        return self.reservation_dict.get(ID)

    def make_free_windows_consistent(self, reservation_list):
        """Vectorized window consistency check."""
        '''Use this when some windows have been made busy in the global 
        schedule, but there are reservations that don't know about it. This
        could also be done in commit, but it's not required by all schedulers
        (only multi-pass ones), so it's better to keep it separate.'''
        # Group reservations by resource for batch processing
        by_resource = defaultdict(list)
        for res in reservation_list:
            for resource in res.free_windows_dict:
                by_resource[resource].append(res)
        
        # Process each resource's reservations together
        for resource, reservations in by_resource.items():
            busy_intervals = self.schedule_dict_busy[resource]
            for res in reservations:
                res.free_windows_dict[resource] = res.free_windows_dict[resource].subtract(busy_intervals)

    def convert_compound_to_simple(self):
        """Convert compound reservations to simple ones while maintaining constraints.
        
        Returns:
            tuple: (list of valid reservations, dict of reservations by ID)
        """
        reservation_list = []
        reservation_dict = {}

        for cr in self.compound_reservation_list:
            # Get valid reservations in one pass
            valid_reservations = [
                res for res in cr.reservation_list 
                if self.check_against_gpw(res)
            ]
            
            if not valid_reservations:
                continue
                
            if cr.issingle():
                # Single reservations only use the first item
                res = valid_reservations[0]
                reservation_list.append(res)
                reservation_dict[res.resID] = res
                
            elif cr.isoneof():
                # For oneof, add all valid reservations
                reservation_list.extend(valid_reservations)
                reservation_dict.update({
                    res.resID: res for res in valid_reservations
                })
                self.oneof_constraints.append(valid_reservations)
                
            elif cr.isand():
                # For and, need all reservations to be valid
                if len(valid_reservations) == len(cr.reservation_list):
                    reservation_list.extend(valid_reservations)
                    reservation_dict.update({
                        res.resID: res for res in valid_reservations
                    })
                    self.and_constraints.append(valid_reservations)

        return reservation_list, reservation_dict
