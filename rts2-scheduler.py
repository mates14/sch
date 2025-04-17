#!/usr/bin/python3

import enum
import logging
from datetime import datetime, timedelta
import subprocess
import sys
import os
import argparse

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, HADec
import astropy.units as u

from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from scipy.interpolate import interp1d

import functools
import statistics
import time

# Project's own imports:
# 1. LCO legacy
from kernel.intervals import Intervals
from kernel.fullscheduler_cp_model import FullScheduler_ortoolkit
from kernel.reservation import Reservation, CompoundReservation, OptimizationType
# 2. Our own
from request import Request, fetch_requests
from sch_plot import visualize_schedule
from recorder import ScheduleRecorder
from config_loader import Config, ConfigurationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from contextlib import contextmanager

@contextmanager
def timing(name):
    start = time.time()
    yield
    end = time.time()
    print(f"timing {name}: {(end - start):.1f} ms")

def check_telescope_state():
    try:
        # Run the command and capture output
        result = subprocess.run(['/usr/local/bin/rts2-state', '-c'],
                              capture_output=True,
                              text=True,
                              check=True)

        # Convert output to integer
        state = int(result.stdout.strip())

        print(f"Current telescope state: {state}")

        # Check if state is valid
        if state not in [2, 3]:
            print(f"It is not the right time to run the scheduler: {state}. Needed 2 or 3.")
            sys.exit(0)

        return state

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get telescope state: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Failed to parse telescope state: {e}")
        sys.exit(1)


def upload_schedule_to_rts2(schedule, config):
    """
    Upload schedule to RTS2 for all telescopes

    Args:
        schedule: Dictionary mapping telescope names to observations
        config: Configuration object
    """
    # Connect to RTS2
    try:
        import rts2
        import rts2.target
        import rts2.rtsapi
        from operator import attrgetter
    except:
        logger.error("Cannot load RTS2 API")
        return

    # For each telescope in the schedule
    for telescope, observations in schedule.items():
        if not observations:  # Skip if no observations
            continue

        # Get RTS2 config for this telescope
        rts2_config = config.get_resource_rts2_config(telescope)
        if not rts2_config:
            logger.error(f"No RTS2 configuration found for telescope {telescope}")
            continue

        try:
            rts2.createProxy(
                url=rts2_config["url"],
                username=rts2_config["user"],
                password=rts2_config["password"]
            )

            sorted_observations = sorted(observations, key=attrgetter('scheduled_start'))

            queue = rts2.Queue(rts2.rtsapi.getProxy(), 'scheduler')

            # Clear the existing queue
            queue.clear()

            # For each scheduled observation
            for observation in sorted_observations:
                # Convert times to Unix timestamps (float)
                start_time = observation.scheduled_start.timestamp()
                end_time = (observation.scheduled_start +
                          timedelta(seconds=observation.scheduled_quantum)).timestamp()

                # Add the target to the queue
                queue.add_target(observation.request.id,
                                start=start_time,
                                end=end_time)
                logger.info(f"Added to queue: Target {observation.request.id}, "
                           f"Start: {observation.scheduled_start}, "
                           f"End: {observation.scheduled_start + timedelta(seconds=observation.scheduled_quantum)}, "
                           f"Name: {observation.request.name}")

            # Save the queue
            queue.load()
            logger.info(f"Schedule uploaded to RTS2 for telescope {telescope}")

        except Exception as e:
            logger.error(f"Error uploading schedule to RTS2 for telescope {telescope}: {e}")

    logger.info("Schedule upload process completed")


def connect_to_db(config) -> psycopg2.extensions.connection:
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**config)
        return conn
    except psycopg2.Error as e:
        print(f"Unable to connect to the database: {e}")
        raise


def hadec_to_altaz_vectorized(ha_array, dec_array, location):
    """Vectorized conversion from HA/Dec to Alt/Az."""
    time = Time('2024-10-21T12:20:00')

    # Create frames once
    hadec_frame = HADec(obstime=time, location=location)
    altaz_frame = AltAz(obstime=time, location=location)

    # Create coordinates for all points at once
    coords = SkyCoord(ha=ha_array*u.deg, dec=dec_array*u.deg, frame=hadec_frame)

    # Transform all points at once
    altaz = coords.transform_to(altaz_frame)

    return altaz.alt.degree, altaz.az.degree


def load_horizon(file_path: str, location: EarthLocation, min_altitude: float = 20.0) -> interp1d:
    """
    Load horizon data from a file and create an interpolation function.

    Args:
        file_path: Path to horizon file
        location: Observatory location
        min_altitude: Minimum allowed altitude in degrees (default: 20.0)

    Returns:
        Interpolation function that returns horizon altitude for given azimuth,
        ensuring returned values are never below min_altitude
    """
    az = []
    alt = []
    ha = []
    dec = []
    is_azalt = False

    # Read and classify points
    with timing("file reading"):
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('AZ-ALT'):
                    is_azalt = True
                    continue

                values = line.split()
                x = float(values[0])
                y = float(values[1])

                if is_azalt:
                    az.append(x)
                    # Apply minimum altitude constraint
                    alt.append(max(y, min_altitude))
                else:
                    ha.append(x * 15)  # Convert hours to degrees
                    dec.append(y)

    # Convert coordinates if needed
    if not is_azalt:
        with timing("hadec conversion batch"):
            ha_array = np.array(ha)
            dec_array = np.array(dec)
            alt_array, az_array = hadec_to_altaz_vectorized(ha_array, dec_array, location)
            az = list(az_array)
            # Apply minimum altitude constraint
            alt = list(np.maximum(alt_array, min_altitude))

    # Ensure circular condition and convert to arrays
    with timing("array conversion"):
        if az[0] != 360:
            az.append(360)
            alt.append(alt[0])
        az = np.array(az)
        alt = np.array(alt)

    with timing("interpolation setup"):
        result = interp1d(az, alt, kind='linear', fill_value='extrapolate')

    print(f"horizon: {len(az)} points, ", 'azalt' if is_azalt else 'hadec', " format")
    print(f"minimum altitude set to {min_altitude} degrees")
    return result


def prepare_sun_positions(start_time, end_time, slice_size, resources):
    """Pre-compute sun positions for all time slices."""
    time_diff = (end_time - start_time).total_seconds()
    n_slices = int(time_diff / slice_size)

    # Create time points at the center of each slice
    slice_centers = [
        start_time + timedelta(seconds=slice_size * i + slice_size/2)
        for i in range(n_slices)
    ]
    time_points = Time(slice_centers)

    # Calculate sun positions for each resource
    sun_positions = {}
    for resource_name, resource_info in resources.items():
        observatory = resource_info['earth_location']
        altaz_frame = AltAz(obstime=time_points, location=observatory)
        sun_altaz = get_sun(time_points).transform_to(altaz_frame)

        # Store altitude, mask, and frame
        sun_positions[resource_name] = {
            'altitudes': sun_altaz.alt.deg,
            'is_night': sun_altaz.alt < -15*u.deg,
            'frame': altaz_frame
        }

    return sun_positions, slice_centers


def calculate_visibility_with_precalc(target, resources, slice_centers, sun_positions, horizon_functions):
    """Calculate visibility using pre-calculated sun positions and telescope-specific horizons."""
    visibility_intervals = {}

    for resource_name, resource_info in resources.items():
        observatory = resource_info['earth_location']
        sun_low = sun_positions[resource_name]['is_night']
        horizon_func = horizon_functions[resource_name]

        if target is not None:
            # Calculate target position for the slice centers
            time_points = Time(slice_centers)
            altaz_frame = AltAz(obstime=time_points, location=observatory)
            target_coord = SkyCoord(ra=target.tar_ra*u.degree,
                                  dec=target.tar_dec*u.degree,
                                  frame='icrs')
            target_altaz = target_coord.transform_to(altaz_frame)

            # Check horizon limits using telescope-specific horizon
            horizon_alt = horizon_func(target_altaz.az.degree)
            target_high = target_altaz.alt.degree > horizon_alt

            is_visible = sun_low & target_high
        else:
            # For global windows, we only care about sun position
            is_visible = sun_low

        # Convert boolean array to time windows
        visible_intervals = []
        start_idx = None

        for i, visible in enumerate(is_visible):
            if visible and start_idx is None:
                start_idx = i
            elif not visible and start_idx is not None:
                start_time = slice_centers[start_idx]
                end_time = slice_centers[i]
                visible_intervals.append((start_time, end_time))
                start_idx = None

        # Handle case where visibility extends to the end
        if start_idx is not None:
            visible_intervals.append((
                slice_centers[start_idx],
                slice_centers[-1] + timedelta(seconds=300)  # Assuming slice_size=300
            ))

        visibility_intervals[resource_name] = visible_intervals

    return visibility_intervals


def load_horizon_for_resource(resource_name, resource_info, config):
    """Load horizon data for a specific resource."""
    # Get the horizon file path from resource config or use default
    horizon_file = None
    min_altitude = None

    # First check if resource has its own horizon configuration
    if 'horizon_file' in resource_info:
        horizon_file = resource_info['horizon_file']

    if 'min_altitude' in resource_info:
        min_altitude = resource_info['min_altitude']

    # Fall back to global config if necessary
    if horizon_file is None:
        horizon_file = config.get_scheduler_param('default_horizon_file')

    if min_altitude is None:
        min_altitude = config.get_scheduler_param('default_min_altitude', 20.0)

    # If still no horizon file, use a hardcoded default
    if horizon_file is None:
        horizon_file = "/etc/rts2/horizon"
        logger.warning(f"No horizon file specified for {resource_name}, using default: {horizon_file}")

    logger.info(f"Loading horizon for {resource_name} from {horizon_file} with min altitude {min_altitude}")

    # Load the horizon function
    return load_horizon(
        horizon_file,
        resource_info['earth_location'],
        min_altitude
    )

def prepare_resources(config: Config):
    """
    Prepare resources for telescopes from configuration.

    Args:
        config: Configuration object

    Returns:
        Dictionary of telescope resources with their properties
    """
    resources_config = config.get_resources()
    resources = {}

    for resource_id, resource_config in resources_config.items():
        resources[resource_id] = {
            'name': resource_config['name'],
            'location': resource_config['location']
        }

    # Convert to EarthLocation objects
    for resource in resources.values():
        loc = resource['location']
        resource['earth_location'] = EarthLocation(
            lat=loc['latitude']*u.deg,
            lon=loc['longitude']*u.deg,
            height=loc['elevation']*u.m
        )

    return resources


def fetch_requests_from_telescopes(resources, config, slice_size):
    """
    Fetch requests from all telescope databases.

    Args:
        resources: Dictionary of telescope resources
        config: Configuration object
        slice_size: Size of time slices in seconds

    Returns:
        Tuple of (all requests, dictionary of request dictionaries by resource)
    """
    connections = {}
    requests_by_resource = {}
    request_dicts_by_resource = {}
    all_requests = []

    try:
        # Connect to each telescope's database
        for resource_name, resource_info in resources.items():
            db_config = config.get_resource_db_config(resource_name)
            connections[resource_name] = connect_to_db(db_config)

            # Fetch requests for this telescope
            resource_requests = fetch_requests(
                connections[resource_name],
                slice_size,
                telescope_id=resource_name
            )

            # Store requests for this resource
            requests_by_resource[resource_name] = resource_requests

            # Create dictionary for fast lookup
            request_dicts_by_resource[resource_name] = {req.id: req for req in resource_requests}

            # Add to all requests
            all_requests.extend(resource_requests)

        return all_requests, request_dicts_by_resource

    finally:
        # Close all database connections
        for conn in connections.values():
            if conn:
                conn.close()


def create_compound_reservations(all_requests, request_dicts_by_resource,
                              resources, slice_centers, sun_positions, horizon_functions):
    """
    Create compound reservations for targets that appear in multiple telescopes.

    Args:
        all_requests: List of all requests from all telescopes
        request_dicts_by_resource: Dictionary mapping resource names to request dictionaries
        resources: Dictionary of telescope resources
        slice_centers: Pre-calculated slice centers
        sun_positions: Pre-calculated sun positions
        horizon_functions: Dictionary of horizon functions for each telescope

    Returns:
        List of compound reservations
    """
    compound_reservations = []
    processed_ids = set()

    for request in all_requests:
        # Skip if we've already processed this target
        if request.id in processed_ids:
            continue

        # Calculate visibility for this request
        visibility_windows = calculate_visibility_with_precalc(
            request, resources, slice_centers, sun_positions, horizon_functions
        )

        # Create and filter windows
        possible_windows_dict = {}
        for resource_name, windows in visibility_windows.items():
            intervals = Intervals(windows)
            intervals.remove_intervals_smaller_than(request.duration)
            if not intervals.is_empty():
                possible_windows_dict[resource_name] = intervals

        # Only proceed if we have valid windows
        if not possible_windows_dict:
            continue

        # Check which telescopes have this target
        target_requests = []
        for resource_name, request_dict in request_dicts_by_resource.items():
            if request.id in request_dict:
                resource_request = request_dict[request.id]

                # Assign visibility windows for this resource
                resource_request.possible_windows_dict = {}

                # Only add windows for this resource's telescope
                if resource_name in possible_windows_dict:
                    resource_request.possible_windows_dict[resource_name] = possible_windows_dict[resource_name]

                # Only add if it has valid windows
                if resource_request.possible_windows_dict:
                    # Create a reservation for this version of the target
                    reservation = Reservation(
                        priority=resource_request.base_priority,
                        duration=resource_request.duration,
                        possible_windows_dict=resource_request.possible_windows_dict,
                        request=resource_request
                    )
                    target_requests.append(reservation)

        # Create appropriate compound reservation based on how many telescopes have this target
        if len(target_requests) > 1:
            # Target exists in multiple telescopes - create a 'oneof' compound reservation
            compound_reservations.append(
                CompoundReservation(target_requests, cr_type='oneof')
            )
        elif len(target_requests) == 1:
            # Target exists in only one telescope - create a 'single' compound reservation
            compound_reservations.append(
                CompoundReservation(target_requests, cr_type='single')
            )

        # Mark as processed
        processed_ids.add(request.id)

    return compound_reservations


def prepare_airmass_data(compound_reservations, resources, slice_size):
    """Calculate airmass data for reservations that need it."""
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


def get_schedule_time_range(resources, slice_size, schedule_recorder):
    """Determine the scheduling time range."""
    # Get the proper start time based on current executing observation
    start_time = schedule_recorder.get_next_available_time()

    # Calculate sunrise times for all resources
    sunrise_times = []
    for resource_name, resource_info in resources.items():
        location = resource_info['earth_location']

        # Check sunrise for today and tomorrow
        for day_offset in [0, 1]:
            date = start_time.date() + timedelta(days=day_offset)
            midnight = datetime.combine(date, datetime.min.time())

            # Use slice_size for granularity
            slices_per_day = int(24 * 3600 / slice_size)
            times = Time(midnight) + np.linspace(0, 24, slices_per_day)*u.hour

            sun_altaz = get_sun(times).transform_to(AltAz(location=location, obstime=times))
            sun_up = sun_altaz.alt > 0*u.deg

            if np.any(sun_up):
                sunrise_index = np.where(sun_up)[0][0]
                sunrise = times[sunrise_index].datetime
                if sunrise > start_time:
                    sunrise_times.append(sunrise)
                    break

    # If no sunrises were found, use 24 hours from start_time as fallback
    if not sunrise_times:
        end_time = start_time + timedelta(hours=24)
    else:
        end_time = max(sunrise_times)

    # Align end_time to the nearest slice
    end_time = end_time.replace(microsecond=0)
    seconds_since_midnight = (end_time - end_time.replace(
        hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    aligned_seconds = (seconds_since_midnight // slice_size) * slice_size
    end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0) + \
               timedelta(seconds=aligned_seconds)

    return start_time, end_time


def prepare_scheduler_input(resources, config, slice_size, schedule_recorder):
    """
    Prepare input for the scheduler with support for multiple telescopes.

    Args:
        resources: Dictionary of telescope resources
        config: Configuration object
        slice_size: Size of time slices in seconds
        schedule_recorder: Schedule recorder instance

    Returns:
        Tuple of (global windows dict, compound reservations, start time)
    """
    with timing("load_horizons"):
        # Load horizon data for each telescope
        horizon_functions = {}
        for resource_name, resource_info in resources.items():
            horizon_functions[resource_name] = load_horizon_for_resource(
                resource_name, resource_info, config
            )

    with timing("get_schedule_time_range"):
        # Set up time range for scheduling
        start_time, end_time = get_schedule_time_range(resources, slice_size, schedule_recorder)
        logger.info(f"Schedule start time: {start_time}")
        logger.info(f"Schedule end time: {end_time}")

    with timing("current_schedule"):
        # Load current schedule
        current_schedule = schedule_recorder.load_current_schedule()

    with timing("excluded_targets"):
        # Get list of targets currently being observed or scheduled for later
        excluded_targets = set()
        if current_schedule is not None:
            current_time = datetime.utcnow()

            durations = [timedelta(seconds=float(d)) for d in current_schedule['duration'].value]

            current_mask = [(t <= current_time) and (t + d > current_time)
                           for t, d in zip(current_schedule['start_time'], durations)]

            if any(current_mask):
                # Add only the currently executing target to excluded set
                excluded_targets = set([current_schedule[current_mask]['target_id'].value[0]])
                logger.info(f"Currently executing target: {excluded_targets}")

    # Pre-compute sun positions and slice centers once
    sun_positions, slice_centers = prepare_sun_positions(
        start_time, end_time, slice_size, resources
    )

    # Get global visibility windows
    night_interval = calculate_visibility_with_precalc(
        None, resources, slice_centers, sun_positions, horizon_functions
    )
    globally_possible_windows_dict = {
        resource: Intervals(windows)
        for resource, windows in night_interval.items()
    }

    # Fetch and process targets from all telescopes
    with timing("fetch_requests"):
        all_requests, request_dicts_by_resource = fetch_requests_from_telescopes(
            resources, config, slice_size
        )

        # Filter out excluded targets
        all_requests = [req for req in all_requests if req.id not in excluded_targets]
        logger.info(f"Processing {len(all_requests)} requests after excluding currently executing targets")

    # Create compound reservations
    with timing("create_compound_reservations"):
        compound_reservations = create_compound_reservations(
            all_requests, request_dicts_by_resource,
            resources, slice_centers, sun_positions, horizon_functions
        )

    # Calculate airmass data for final set
    with timing("prepare_airmass_data"):
        prepare_airmass_data(compound_reservations, resources, slice_size)

    return globally_possible_windows_dict, compound_reservations, start_time


def run_scheduler(resources, config, slice_size=None):
    """Run the scheduler with the prepared inputs."""
    if slice_size is None:
        slice_size = config.get_scheduler_param('slice_size', 300)

    with timing("run_scheduler"):
        with timing("Schedule Recorder"):
            schedule_recorder = ScheduleRecorder(
                base_path=config.get_output_path('schedule_dir', './schedules')
            )

        # Prepare scheduler input
        with timing("Prepare Scheduler Input"):
            globally_possible_windows_dict, reservations, start_time = prepare_scheduler_input(
                resources, config, slice_size, schedule_recorder
            )

        # Initialize scheduler
        with timing("Initialize Scheduler"):
            scheduler = FullScheduler_ortoolkit(
                kernel='CBC',
                compound_reservation_list=reservations,
                globally_possible_windows_dict=globally_possible_windows_dict,
                contractual_obligation_list=[],
                slice_size_seconds=slice_size,
                mip_gap=config.get_scheduler_param('mip_gap', 0.05),
                warm_starts=False,
            )

        # Run scheduling
        with timing("ScheduleAll"):
            schedule = scheduler.schedule_all(
                timelimit=config.get_scheduler_param('timelimit', 180)
            )

        # Save the schedule
        with timing("Save Schedule"):
            schedule_recorder.save_schedule(schedule, calculated_at=datetime.utcnow())

        return schedule, schedule_recorder


def main():
    """Main function to run the scheduler."""
    parser = argparse.ArgumentParser(description='RTS2 dual telescope scheduler')
    parser.add_argument('--config', '-c', default='sch.cfg',
                        help='Path to configuration file')
    parser.add_argument('--skip-state-check', '-f', action='store_true',
                        help='Skip telescope state check (for testing)')
    args = parser.parse_args()

    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Check telescope state before proceeding
        if not args.skip_state_check:
            check_telescope_state()
        else:
            logger.warning("Skipping telescope state check")

        # Prepare resources for telescopes
        resources = prepare_resources(config)
        logger.info(f"Prepared resources for {len(resources)} telescopes")

        # Run the scheduler
        schedule, recorder = run_scheduler(resources, config)
        logger.info("Scheduling completed")

        # Upload to RTS2 if schedule is not empty
        with timing("Upload Schedule to RTS2"):
            if any(observations for observations in schedule.values()):
                upload_schedule_to_rts2(schedule, config)
            else:
                logger.warning("No observations scheduled, skipping RTS2 upload")

        # Visualize the schedule
        with timing("Visualize the Schedule"):
            date = datetime.utcnow()
            plot_file = os.path.join(
                config.get_output_path('plot_dir', './plots'),
                f"schedule-plot-{date.strftime('%Y%m%d-%H%M%S')}.png"
            )

            # Connect to main DB for visualization (we use the first resource's DB)
            first_resource_name = next(iter(resources.keys()))
            db_config = config.get_resource_db_config(first_resource_name)

            if db_config:
                try:
                    conn = connect_to_db(db_config)
                    visualize_schedule(schedule, resources, plot_file, conn)
                    conn.close()
                    logger.info(f"Schedule visualization saved to {plot_file}")
                except Exception as e:
                    logger.error(f"Error visualizing schedule: {e}")
            else:
                logger.error(f"No database configuration found for resource {first_resource_name}")

        logger.info("Scheduler execution completed successfully")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running scheduler: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
