"""
Updated scheduler_core.py to integrate sinfo system (minimal changes)
"""

import logging
from datetime import datetime, timedelta
from collections import defaultdict

from config_loader import Config
from database import load_all_requests, group_requests_by_target
from kernel.solver import CPScheduler
from kernel.intervals import Intervals
from kernel.reservation import Reservation, CompoundReservation
import astro_utils
from recorder import ScheduleRecorder

logger = logging.getLogger(__name__)

def run_scheduling_algorithm(config, available_telescopes, recorder):
    """
    Main scheduling algorithm with sinfo support.
    
    Args:
        config: Configuration object
        available_telescopes: Dict mapping telescope_name -> is_available
        recorder: ScheduleRecorder instance
    
    Returns:
        Dict mapping telescope_name -> list of scheduled observations
    """
    logger.info("Starting scheduling algorithm")
    
    # Get your existing slice_size
    slice_size = config.get_scheduler_param('slice_size', 300)
    
    # Load requests from all databases with sinfo support  
    all_requests = load_all_requests(config, available_telescopes)
    
    if not all_requests:
        logger.warning("No enabled requests found in any database")
        return {}
    
    # Prepare your existing scheduler inputs (use your existing prepare_scheduler_input logic)
    resources = {name: cfg for name, cfg in config.get_resources().items() 
                if available_telescopes.get(name, False)}
    
    # You'll need to adapt your existing prepare_scheduler_input function to get:
    # slice_centers, sun_positions, horizon_functions
    # For now, I'll assume you have a function that provides these:
    scheduler_inputs = prepare_scheduler_input(resources, config, slice_size, recorder)
    slice_centers = scheduler_inputs['slice_centers']
    sun_positions = scheduler_inputs['sun_positions'] 
    horizon_functions = scheduler_inputs['horizon_functions']
    
    # Create compound reservations using your existing logic + sinfo modifications
    compound_reservations = create_compound_reservations_with_sinfo(
        all_requests, resources, slice_centers, sun_positions, 
        horizon_functions, slice_size
    )
    
    if not compound_reservations:
        logger.warning("No valid compound reservations created")
        return {}
    
    # Create possible_windows_dict for the solver
    possible_windows_dict = {}
    for telescope in resources.keys():
        # This creates a union of all visibility windows for this telescope
        all_windows = []
        for cr in compound_reservations:
            for reservation in cr.reservation_list:
                if telescope in reservation.possible_windows_dict:
                    all_windows.append(reservation.possible_windows_dict[telescope])
        
        if all_windows:
            # Union all windows for this telescope
            combined = all_windows[0].union(all_windows[1:])
            possible_windows_dict[telescope] = combined
    
    # Run the CP scheduler (unchanged)
    scheduler = CPScheduler(
        compound_reservation_list=compound_reservations,
        globally_possible_windows_dict=possible_windows_dict,
        slice_size_seconds=slice_size,
        timelimit=config.get_scheduler_param('timelimit', 180),
        mip_gap=config.get_scheduler_param('mip_gap', 0.05)
    )
    
    schedule = scheduler.schedule_all()
    
    # Log scheduling statistics
    total_scheduled = sum(len(obs_list) for obs_list in schedule.values())
    total_requests = len(all_requests)
    
    logger.info(f"Scheduled {total_scheduled} observations from {total_requests} requests")
    
    for telescope, observations in schedule.items():
        if observations:
            total_time = sum(obs.scheduled_quantum for obs in observations)
            logger.info(f"{telescope}: {len(observations)} observations, "
                       f"{total_time}s total ({total_time/3600:.1f}h)")
    
    return schedule

def create_compound_reservations_with_sinfo(all_requests, resources, slice_centers, 
                                          sun_positions, horizon_functions, slice_size):
    """Create compound reservations with sinfo support (adapted from your original)."""
    compound_reservations = []
    
    # Group requests by target ID
    target_groups = group_requests_by_target(all_requests)
    
    for target_id, target_requests in target_groups.items():
        # Use first request for visibility calculation (they all have same coordinates)
        first_request = target_requests[0] 
        
        # Calculate visibility using your existing function
        visibility = astro_utils.calculate_visibility(
            first_request, resources, slice_centers, sun_positions, 
            horizon_functions, slice_size
        )
        
        # Check for AND type (dominant if any request has it)
        has_and_type = any(req.has_and_type() for req in target_requests)
        
        # Create reservations for each telescope that has this target
        target_reservations = []
        for request in target_requests:
            telescope = request.telescope_name
            
            if telescope not in visibility:
                continue
                
            # Create intervals and filter small ones
            intervals = Intervals(visibility[telescope])
            intervals.remove_intervals_smaller_than(request.duration)
            
            if intervals.is_empty():
                continue
            
            reservation = Reservation(
                priority=request.base_priority,
                duration=request.duration,
                possible_windows_dict={telescope: intervals},
                request=request
            )
            target_reservations.append(reservation)
        
        if not target_reservations:
            continue
        
        # Determine compound type based on sinfo
        if has_and_type:
            compound_type = 'and'
        elif len(target_reservations) > 1:
            compound_type = 'oneof'
        else:
            compound_type = 'single'
        
        compound_reservation = CompoundReservation(target_reservations, compound_type)
        compound_reservations.append(compound_reservation)
        
        logger.debug(f"Created {compound_type} compound reservation for target {target_id} "
                     f"with {len(target_reservations)} telescope options")
    
    logger.info(f"Created {len(compound_reservations)} compound reservations")
    return compound_reservations

# Integration notes:
# 1. You'll need to adapt your existing prepare_scheduler_input() function 
#    to work with the new multi-telescope request loading
# 2. The visibility calculation uses your existing astro_utils.calculate_visibility()
# 3. Compound reservation logic now respects sinfo parameters (type=and, etc.)
# 4. Each telescope gets its own request with telescope-specific duration
