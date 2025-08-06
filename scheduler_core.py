"""
Scheduler Core Module
Wraps the scheduling algorithm and preparation logic.
"""

import logging
from datetime import datetime, timedelta
from kernel.intervals import Intervals
from kernel.solver import CPScheduler
from kernel.reservation import Reservation, CompoundReservation
import astro_utils
import database

logger = logging.getLogger(__name__)


def run_scheduling_algorithm(config, available_telescopes, recorder):
    """Run the complete scheduling algorithm."""
    # Filter resources to available telescopes
    all_resources = config.get_resources()
    resources = {name: res for name, res in all_resources.items() 
                if available_telescopes.get(name, False)}
    
    if not resources:
        logger.warning("No available resources for scheduling")
        return {}
    
    # Get scheduling parameters
    slice_size = config.get_scheduler_param('slice_size', 300)
    
    # Prepare inputs
    logger.info("Preparing scheduler inputs")
    gpw_dict, reservations, start_time = _prepare_scheduler_input(
        resources, config, slice_size, recorder
    )
    
    if not reservations:
        logger.warning("No valid reservations to schedule")
        return {}
    
    # Run scheduler
    logger.info(f"Running scheduler with {len(reservations)} compound reservations")
    scheduler = CPScheduler(
        compound_reservation_list=reservations,
        globally_possible_windows_dict=gpw_dict,
        slice_size_seconds=slice_size,
        timelimit=config.get_scheduler_param('timelimit', 180),
        mip_gap=config.get_scheduler_param('mip_gap', 0.05)
    )
    
    schedule = scheduler.schedule_all()
    
    # Log results
    total_scheduled = sum(len(obs) for obs in schedule.values())
    logger.info(f"Scheduled {total_scheduled} observations across {len(schedule)} telescopes")
    
    return schedule


def _prepare_scheduler_input(resources, config, slice_size, recorder):
    """Prepare all inputs for the scheduler."""
    # Load horizons
    horizon_functions = {}
    for name, info in resources.items():
        horizon_file = config.get_scheduler_param('horizon_file', '/etc/rts2/horizon')
        min_alt = config.get_scheduler_param('min_altitude', 20.0)
        horizon_functions[name] = astro_utils.load_horizon(
            horizon_file, info['earth_location'], min_alt
        )
    
    # Determine time range
    start_time = recorder.get_next_available_time()
    end_time = astro_utils.find_next_sunrise(start_time, resources, slice_size)
    
    logger.info(f"Scheduling window: {start_time} to {end_time}")
    
    # Pre-calculate sun positions
    sun_positions, slice_centers = astro_utils.prepare_sun_positions(
        start_time, end_time, slice_size, resources
    )
    
    # Get global windows
    night_intervals = astro_utils.calculate_visibility(
        None, resources, slice_centers, sun_positions, horizon_functions
    )
    
    gpw_dict = {res: Intervals(windows) 
               for res, windows in night_intervals.items()}
    
    # Fetch requests from all telescopes
    all_requests = []
    requests_by_resource = {}
    
    for resource_name in resources:
        db_config = config.get_resource_db_config(resource_name)
        with database.get_connection(db_config) as conn:
            requests = database.fetch_requests(conn, slice_size, resource_name)
            all_requests.extend(requests)
            requests_by_resource[resource_name] = {r.id: r for r in requests}
    
    # Create compound reservations
    compound_reservations = _create_compound_reservations(
        all_requests, requests_by_resource, resources,
        slice_centers, sun_positions, horizon_functions
    )
    
    return gpw_dict, compound_reservations, start_time


def _create_compound_reservations(all_requests, requests_by_resource, resources,
                                 slice_centers, sun_positions, horizon_functions):
    """Create compound reservations for scheduling."""
    compound_reservations = []
    processed_ids = set()
    
    for request in all_requests:
        if request.id in processed_ids:
            continue
        
        # Calculate visibility
        visibility = astro_utils.calculate_visibility(
            request, resources, slice_centers, sun_positions, horizon_functions
        )
        
        # Create windows
        possible_windows = {}
        for res_name, windows in visibility.items():
            intervals = Intervals(windows)
            intervals.remove_intervals_smaller_than(request.duration)
            if not intervals.is_empty():
                possible_windows[res_name] = intervals
        
        if not possible_windows:
            continue
        
        # Check which telescopes have this target
        target_reservations = []
        for res_name, req_dict in requests_by_resource.items():
            if request.id in req_dict and res_name in possible_windows:
                reservation = Reservation(
                    priority=request.base_priority,
                    duration=request.duration,
                    possible_windows_dict={res_name: possible_windows[res_name]},
                    request=request
                )
                target_reservations.append(reservation)
        
        # Create compound reservation
        if len(target_reservations) > 1:
            compound_reservations.append(
                CompoundReservation(target_reservations, cr_type='oneof')
            )
        elif len(target_reservations) == 1:
            compound_reservations.append(
                CompoundReservation(target_reservations, cr_type='single')
            )
        
        processed_ids.add(request.id)
    
    logger.info(f"Created {len(compound_reservations)} compound reservations")
    return compound_reservations
