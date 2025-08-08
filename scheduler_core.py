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

    horizon_functions = _get_horizon_functions(resources, config)

    # Prepare inputs
    logger.info("Preparing scheduler inputs")
    gpw_dict, reservations, schedule_start_times_by_telescope = _prepare_scheduler_input(
        resources, config, slice_size, recorder, horizon_functions
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

    return {
        'schedule': schedule,
        'schedule_start_times': schedule_start_times_by_telescope,
        'horizon_functions': horizon_functions,
        'compound_reservations': reservations
    }


def _prepare_scheduler_input(resources, config, slice_size, recorder, horizon_functions):
    """Prepare all inputs for the scheduler."""
    # Ensure earth locations exist
    resources = astro_utils.ensure_earth_locations(resources)

    # Calculate last noon for opportunity target exclusion
    now = datetime.utcnow()
    last_noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now.hour < 12:
        last_noon -= timedelta(days=1)

    # MOVED UP: Fetch requests from all telescopes to determine start time
    all_requests = []
    requests_by_resource = {}
    schedule_start_times_by_resource = {}
    grb_detected = False

    for resource_name in resources:
        db_config = config.get_resource_db_config(resource_name)
        with database.get_connection(db_config) as conn:
            target_rows, schedule_start_time = database.fetch_requests(conn, resource_name, last_noon)

            schedule_start_times_by_resource[resource_name] = schedule_start_time
            # Convert rows to Request objects (existing conversion logic)
            requests = [convert_row_to_request(row, resource_name) for row in target_rows]

            # Check for GRBs
            if any(row['type_id'] == 'G' for row in target_rows):
                grb_detected = True
                logger.warning(f"*** GRB targets found on {resource_name}! ***")

            all_requests.extend(requests)
            requests_by_resource[resource_name] = {r.id: r for r in requests}
            schedule_start_times.append(schedule_start_time)

    # Determine actual start time based on database results
    if grb_detected:
        start_time = datetime.utcnow()  # Immediate scheduling for GRBs
        logger.warning("*** GRB MODE: Immediate scheduling ***")
    else:
        start_time = min(schedule_start_times_by_resource.values()) if schedule_start_times_by_resource else datetime.utcnow()
        logger.info(f"Peace mode: scheduling starts at {start_time}")

    # NOW continue with time-dependent calculations using the determined start_time
    end_time = astro_utils.find_next_sunrise(start_time, resources, slice_size)
    logger.info(f"Scheduling window: {start_time} to {end_time}")

    # Pre-calculate sun positions
    sun_positions, slice_centers = astro_utils.prepare_sun_positions(
        start_time, end_time, slice_size, resources
    )

    # Get global windows
    night_intervals = astro_utils.calculate_visibility(
        None, resources, slice_centers, sun_positions, horizon_functions, slice_size
    )
    gpw_dict = {res: Intervals(windows)
               for res, windows in night_intervals.items()}

    # Create compound reservations (now using already-fetched requests)
    compound_reservations = create_compound_reservations(
        all_requests, requests_by_resource, resources,
        slice_centers, sun_positions, horizon_functions, slice_size
    )

    logger.info("Preparing airmass data for optimization")
    prepare_airmass_data(compound_reservations, resources, slice_size)

    return gpw_dict, compound_reservations, schedule_start_times_by_resource


def _get_horizon_functions(resources, config):
    """Extract horizon functions creation logic - let load_horizon handle defaults."""
    resources = astro_utils.ensure_earth_locations(resources)
    horizon_functions = {}

    for name, info in resources.items():
        telescope_config = config.get_resources()[name]

        # Simple fallback: resource config -> global config -> None (let load_horizon handle it)
        horizon_file = telescope_config.get('horizon_file') or config.get_scheduler_param('default_horizon_file')
        min_altitude = telescope_config.get('min_altitude') or config.get_scheduler_param('default_min_altitude', 20.0)

        logger.info(f"Loading horizon for {name}: file={horizon_file}, min_alt={min_altitude}")

        # Let load_horizon handle None horizon_file gracefully
        horizon_functions[name] = astro_utils.load_horizon(
            horizon_file, info['earth_location'], min_altitude
        )

    return horizon_functions


def _create_compound_reservations(all_requests, requests_by_resource, resources,
                                 slice_centers, sun_positions, horizon_functions, slice_size):
    """Create compound reservations for scheduling."""
    compound_reservations = []
    processed_ids = set()

    for request in all_requests:
        if request.id in processed_ids:
            continue

        # Calculate visibility
        visibility = astro_utils.calculate_visibility(
            request, resources, slice_centers, sun_positions, horizon_functions, slice_size
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
        has_and_type = False

        for res_name, req_dict in requests_by_resource.items():
            if request.id in req_dict and res_name in possible_windows:
                telescope_request = req_dict[request.id]

                if telescope_request.has_and_type(): has_and_type = True

                reservation = Reservation(
                    priority=telescope_request.base_priority,
                    duration=telescope_request.duration,
                    possible_windows_dict={res_name: possible_windows[res_name]},
                    request=telescope_request
                )
                target_reservations.append(reservation)

        # Create compound reservation
        if has_and_type:  # Check for 'and' first (dominant)
            compound_reservations.append(
                CompoundReservation(target_reservations, cr_type='and')
            )
        elif len(target_reservations) > 1:
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

def _prepare_airmass_data(compound_reservations, resources, slice_size):
    """Calculate airmass data for reservations that need it (restored from original)."""
    from kernel.reservation import OptimizationType

    for compound_reservation in compound_reservations:
        for reservation in compound_reservation.reservation_list:
            if reservation.request and reservation.request.optimization_type == OptimizationType.AIRMASS:
                for resource_name, resource_info in resources.items():
                    if resource_name in reservation.possible_windows_dict:
                        windows = reservation.possible_windows_dict[resource_name]
                        window_tuples = windows.toTupleList()

                        all_times = []
                        for start, end in window_tuples:
                            # Calculate airmass every slice_size within each window
                            window_duration = (end - start).total_seconds()
                            num_points = max(2, int(window_duration / slice_size))
                            times = [start + timedelta(seconds=i*window_duration/(num_points-1))
                                   for i in range(num_points)]
                            all_times.extend(times)

                        if all_times:
                            reservation.request.cache_airmass(
                                resource_name, resource_info['location'], all_times
                            )
