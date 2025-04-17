#!/usr/bin/python3

import enum
import logging
from datetime import datetime, timedelta
import subprocess
import sys

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, HADec
import astropy.units as u

from typing import Dict, List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import DictCursor
from scipy.interpolate import interp1d

#from collections import defaultdict
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


logger = logging.getLogger(__name__)

from contextlib import contextmanager

@contextmanager
def timing(name):
    start = time.time()
    yield
    end = time.time()
    #print(f"timing {name}: {(end - start)*1000:.3f} ms")

# Configuration
DB_CONFIG = {
    "dbname": "stars",
    "user": "mates",
    "password": "pasewcic25",
    "host": "localhost"
}

JSON_CONFIG = {
    "url": "http://localhost:8889",
    "user": "scheduler",
    "password": "EdHijcabr2",
}

def get_rts2_state():
    json=rts2.createProxy(url=JSON_CONFIG['url'], username=JSON_CONFIG['user'], password=JSON_CONFIG['password'])
    return json.getState('centrald')

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


def upload_schedule_to_rts2(schedule, rts2_url, username, password):
    # Connect to RTS2
    try:
        import rts2
        import rts2.target
        import rts2.rtsapi
        from operator import attrgetter
    except:
        print("cannot load RTS2 API")
        return 

    rts2.createProxy(url=rts2_url, username=username, password=password)

    # For each telescope in the schedule
    for telescope, observations in schedule.items():
        # Create or get the queue for this telescope

        sorted_observations = sorted(observations, key=attrgetter('scheduled_start'))

        queue = rts2.Queue(rts2.rtsapi.getProxy(), 'scheduler')
        
        # Clear the existing queue
        queue.clear()

        # For each scheduled observation
        for observation in sorted_observations:
            # Add the target to the queue with no times
            # queue.add_target(observation.request.id)

            # Convert times to Unix timestamps (float)
            start_time = observation.scheduled_start.timestamp()
            end_time = (observation.scheduled_start + timedelta(seconds=observation.scheduled_quantum)).timestamp()

            # Add the target to the queue
            queue.add_target(observation.request.id,
                             start=start_time,
                             end=end_time)
            print(observation.request.id, observation.scheduled_start, observation.scheduled_start + timedelta(seconds=observation.scheduled_quantum), observation.request.name) 

        # Save the queue
        queue.load()

    print("Schedule uploaded to RTS2")


def connect_to_db() -> psycopg2.extensions.connection:
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Unable to connect to the database: {e}")
        raise

#from astropy.coordinates import EarthLocation, SkyCoord, AltAz, HADec
#from astropy.time import Time

def hadec_to_altaz(ha, dec, location):
    """
    Convert hour angle and declination to altitude and azimuth using Astropy.
    
    :param ha: Hour angle as an Astropy Quantity
    :param dec: Declination as an Astropy Quantity
    :param lat: Observer's latitude as an Astropy Quantity
    :param lon: Observer's longitude as an Astropy Quantity
    :param time: Time of observation as an Astropy Time object
    :return: SkyCoord object in AltAz frame
    """
    # Create EarthLocation object
    # location = EarthLocation(lat=lat, lon=lon)

    # it should not need time, lets use now
    time = Time('2024-10-21T12:20:00')

    # Create HADec coordinate
    coord = SkyCoord(ha=ha, dec=dec, frame=HADec(obstime=time, location=location))
    
    # Transform to AltAz frame
    altaz_frame = AltAz(obstime=time, location=location)
    altaz = coord.transform_to(altaz_frame)
    
    return altaz.alt.degree, altaz.az.degree

#print(f"Altitude: {altaz.alt:.2f}")
#print(f"Azimuth: {altaz.az:.2f}")

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


def calculate_visibility_with_precalc(target, resources, slice_centers, sun_positions, horizon_func):
    """Calculate visibility using pre-calculated sun positions."""
    visibility_intervals = {}
    
    for resource_name, resource_info in resources.items():
        observatory = resource_info['earth_location']
        sun_low = sun_positions[resource_name]['is_night']
        
        if target is not None:
            # Calculate target position for the slice centers
            time_points = Time(slice_centers)
            altaz_frame = AltAz(obstime=time_points, location=observatory)
            target_coord = SkyCoord(ra=target.tar_ra*u.degree, 
                                  dec=target.tar_dec*u.degree, 
                                  frame='icrs')
            target_altaz = target_coord.transform_to(altaz_frame)
            
            # Check horizon limits
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
                slice_centers[-1] + timedelta(seconds=slice_size)
            ))
        
        visibility_intervals[resource_name] = visible_intervals
    
    return visibility_intervals

def prepare_resources2():
    resources = {
        'telescope1': {
            'name': 'D50',
            'location': {
                'latitude': 49.9093889,
                'longitude': 14.7813631, 
                'elevation': 530
            }
        },
        'telescope2': {
            'name': 'SBT',
            'location': {
                'latitude': 49.9090806,
                'longitude': 14.7819092, 
                'elevation': 530
            }
        }
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

def prepare_resources():
    resources = {
        'telescope1': {
            'name': 'D50',
            'location': {
                'latitude': 49.9093889,
                'longitude': 14.7813631, 
                'elevation': 530
            }
        }
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

def get_targets(conn, slice_size, resources, start_time, end_time, horizon_func, 
                sun_positions=None, slice_centers=None):
    """
    Fetch targets and compute their visibility windows efficiently.
    
    Args:
        conn: Database connection
        slice_size: Size of time slices in seconds
        resources: Dictionary of telescope resources
        start_time, end_time: Schedule time range
        horizon_func: Function to calculate horizon limits
        sun_positions: Pre-computed sun positions (optional)
        slice_centers: Pre-computed slice centers (optional)
    """
    # First get all targets from database
    requests = fetch_requests(conn, slice_size)
    print(f"Fetched {len(requests)} requests from database")
    
    # Calculate sun positions and slice centers if not provided
    if sun_positions is None or slice_centers is None:
        time_diff = (end_time - start_time).total_seconds()
        n_slices = int(time_diff / slice_size)
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
            sun_positions[resource_name] = {
                'altitudes': sun_altaz.alt.deg,
                'is_night': sun_altaz.alt < -15*u.deg,
                'frame': altaz_frame
            }
    
    # Process all targets
    for request in requests:
        possible_windows_dict = {}
        
        for resource_name, resource_info in resources.items():
            observatory = resource_info['earth_location']
            sun_low = sun_positions[resource_name]['is_night']
            altaz_frame = sun_positions[resource_name]['frame']
            
            # Calculate target position for all slice centers at once
            target_coord = SkyCoord(
                ra=request.tar_ra*u.degree,
                dec=request.tar_dec*u.degree,
                frame='icrs'
            )
            target_altaz = target_coord.transform_to(altaz_frame)
            
            # Check horizon limits
            horizon_alt = horizon_func(target_altaz.az.degree)
            target_high = target_altaz.alt.degree > horizon_alt
            
            # Combine visibility conditions
            is_visible = sun_low & target_high
            
            # Convert boolean array to time windows
            visible_intervals = []
            start_idx = None
            
            for i, visible in enumerate(is_visible):
                if visible and start_idx is None:
                    start_idx = i
                elif not visible and start_idx is not None:
                    start_time = slice_centers[start_idx]
                    end_time = slice_centers[i]
                    if end_time - start_time >= timedelta(seconds=request.duration):
                        visible_intervals.append((start_time, end_time))
                    start_idx = None
            
            # Handle case where visibility extends to the end
            if start_idx is not None:
                end_time = slice_centers[-1] + timedelta(seconds=slice_size)
                if end_time - slice_centers[start_idx] >= timedelta(seconds=request.duration):
                    visible_intervals.append((slice_centers[start_idx], end_time))
            
            # Only create intervals if we found any valid windows
            if visible_intervals:
                possible_windows_dict[resource_name] = Intervals(visible_intervals)
        
        # Update the request with its visibility windows
        request.possible_windows_dict = possible_windows_dict
    
    return requests

def test_horizon_func(horizon_func):
    import matplotlib.pyplot as plt
    test_azimuths = np.linspace(0, 360, 360)
    test_altitudes = horizon_func(test_azimuths)
    plt.plot(test_azimuths, test_altitudes)
    plt.title("Horizon Function Test")
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Altitude (degrees)")
    plt.savefig("horizon_test.png")
    plt.close()

def get_schedule_time_range(resources, slice_size, schedule_recorder):
    # Get the proper start time based on current executing observation
    start_time = schedule_recorder.get_next_available_time()
    
    # Calculate sunrise times for all resources
    sunrise_times = []
    for resource_name, resource_info in resources.items():
        location = EarthLocation(
            lat=resource_info['location']['latitude']*u.deg,
            lon=resource_info['location']['longitude']*u.deg,
            height=resource_info['location']['elevation']*u.m
        )
        
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
    seconds_since_midnight = (end_time - end_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    aligned_seconds = (seconds_since_midnight // slice_size) * slice_size
    end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=aligned_seconds)

    return start_time, end_time

def get_schedule_time_range_simple(resources):
    now = datetime.utcnow()

    # Calculate the next noon (12:00 UT)
    if now.hour < 12:
        # If it's before noon, next noon is today
        next_noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    else:
        # If it's after noon, next noon is tomorrow
        tomorrow = now + timedelta(days=1)
        next_noon = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 12, 0, 0)

#        next_noon = (now + timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)

    # Use current time as start_time
    start_time = now
    end_time = next_noon

    return start_time, end_time

# Main function to prepare scheduler input
def prepare_scheduler_input(resources, slice_size, schedule_recorder):

    with timing("connect_to_db"):
        # Connect to the database
        conn = connect_to_db()
        
    with timing("load_horizon"):
        # this should be a part of resource info, but for now... 
        #horizon_func = load_horizon('horizon', EarthLocation(lon=14.7819092*u.deg, lat=49.9090806*u.deg, height=530*u.m) )
        horizon_func = load_horizon('/etc/rts2/horizon', EarthLocation(lon=14.7819092*u.deg, lat=49.9090806*u.deg, height=530*u.m) )
        #test_horizon_func(horizon_func)
       
    with timing("get_schedule_time_range"):
        # Set up time range for one night (example: tonight from sunset to sunrise)
        start_time, end_time = get_schedule_time_range(resources, slice_size, schedule_recorder)
#        print(f"Schedule start time: {start_time}")
#        print(f"Schedule end time: {end_time}") 
        
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
                print(f"Currently executing target: {excluded_targets}")
    
    # Pre-compute sun positions and slice centers once
    sun_positions, slice_centers = prepare_sun_positions(
        start_time, end_time, slice_size, resources
    )
    
    # Get global visibility windows
    night_interval = calculate_visibility_with_precalc(
        None, resources, slice_centers, sun_positions, horizon_func
    )
    globally_possible_windows_dict = {
        resource: Intervals(windows) 
        for resource, windows in night_interval.items()
    }
    
    # Process targets and create reservations in one pass
    with timing("process_targets_and_reservations"):
        requests = fetch_requests(conn, slice_size)
        compound_reservations = []
        
        for request in requests:
            if request.id in excluded_targets:
                continue
                
            # Calculate visibility once for this target
            visibility_windows = calculate_visibility_with_precalc(
                request, resources, slice_centers, sun_positions, horizon_func
            )
            
            # Create and filter windows
            possible_windows_dict = {}
            for resource_name, windows in visibility_windows.items():
                intervals = Intervals(windows)
                intervals.remove_intervals_smaller_than(request.duration)
                if not intervals.is_empty():
                    possible_windows_dict[resource_name] = intervals
            
            # Only create reservation if we found valid windows
            if possible_windows_dict:
                request.possible_windows_dict = possible_windows_dict
                reservation = Reservation(
                    priority=request.base_priority,
                    duration=request.duration,
                    possible_windows_dict=possible_windows_dict,
                    request=request
                )
                compound_reservations.append(
                    CompoundReservation([reservation], cr_type='single')
                )
    
    # Calculate airmass data for final set
    with timing("prepare_airmass_data"):
        prepare_airmass_data(compound_reservations, resources, slice_size)
        
    return globally_possible_windows_dict, compound_reservations, start_time

def prepare_scheduler_input_v2(resources, slice_size, schedule_recorder):
    with timing("connect_to_db"):
        # Connect to the database
        conn = connect_to_db()
        
    with timing("load_horizon"):
        # this should be a part of resource info, but for now... 
        #horizon_func = load_horizon('horizon', EarthLocation(lon=14.7819092*u.deg, lat=49.9090806*u.deg, height=530*u.m) )
        horizon_func = load_horizon('/etc/rts2/horizon', EarthLocation(lon=14.7819092*u.deg, lat=49.9090806*u.deg, height=530*u.m) )
        #test_horizon_func(horizon_func)
       
    with timing("get_schedule_time_range"):
        # Set up time range for one night (example: tonight from sunset to sunrise)
        start_time, end_time = get_schedule_time_range(resources, slice_size, schedule_recorder)
        print(f"Schedule start time: {start_time}")
        print(f"Schedule end time: {end_time}") 
        
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
                print(f"Currently executing target: {excluded_targets}")
     
    # Pre-compute sun positions once
    with timing("prepare_sun_positions"):
        sun_positions, slice_centers = prepare_sun_positions(
            start_time, end_time, slice_size, resources
        )
    
        # prepare targets
    with timing("get_targets"):
        requests = get_targets(conn, slice_size, resources, start_time, end_time, horizon_func, sun_positions, slice_centers)
#        requests = get_targets(conn, slice_size, resources, start_time, end_time, horizon_func)
        print(f"Fetched {len(requests)} requests from database")
        requests = [req for req in requests if req.id not in excluded_targets]
        print(f"Fetched {len(requests)} requests after excluding currently executing target")

    # Get global visibility windows
    with timing("globally_possible_windows_dict"):
        night_interval = calculate_visibility_with_precalc(
            None, resources, slice_centers, sun_positions, horizon_func
        )
        globally_possible_windows_dict = {
            resource: Intervals(windows) 
            for resource, windows in night_interval.items()
        }
    
    # Prepare reservations
    with timing("compound_reservations"):
        compound_reservations = []

        # Calculate visibility for all targets using same pre-computed data
        for request in requests:
            visibility_windows = calculate_visibility_with_precalc(
                request, resources, slice_centers, sun_positions, horizon_func
            )
            request.possible_windows_dict = {
                resource_name: Intervals(windows)
                for resource_name, windows in visibility_windows.items()
            }
    
            # Remove intervals smaller than the duration for each resource
            for resource_name, intervals in request.possible_windows_dict.items():
                intervals.remove_intervals_smaller_than(request.duration)

            # Create a Reservation object
            reservation = Reservation(
                priority=request.base_priority,
                duration=request.duration,
                possible_windows_dict=request.possible_windows_dict,
                request=request
            )

            # Wrap each Reservation in a CompoundReservation
            compound_reservations.append(CompoundReservation([reservation], cr_type='single'))

    with timing("prepare_airmass_data"):
        prepare_airmass_data(compound_reservations, resources, slice_size)

    return globally_possible_windows_dict, compound_reservations, start_time
       
def mp_prepare_airmass_data(compound_reservations, resources):
    all_reservations = [res for cr in compound_reservations for res in cr.reservation_list]
    
    def process_reservation_batch(reservations_batch):
        for reservation in reservations_batch:
            if hasattr(reservation, 'request') and reservation.request:
                if reservation.request.optimization_type == OptimizationType.AIRMASS:
                    for resource_name, resource_info in resources.items():
                        if resource_name in reservation.possible_windows_dict:
                            windows = reservation.possible_windows_dict[resource_name]
                            window_tuples = windows.toTupleList()
                            times = [start for start, end in window_tuples]
                            if times:
                                reservation.request.cache_airmass(resource_name, resource_info['location'], times)

    # Split reservations into batches
    batch_size = max(1, len(all_reservations) // (cpu_count() * 2))  # Adjust this value as needed
    reservation_batches = [all_reservations[i:i + batch_size] for i in range(0, len(all_reservations), batch_size)]

    # Process batches in parallel
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_reservation_batch, reservation_batches)

def prepare_airmass_data(compound_reservations, resources, slice_size):
    for compound_reservation in compound_reservations:
        for reservation in compound_reservation.reservation_list:
            _calculate_and_cache_airmass(reservation, resources, slice_size)

def _calculate_and_cache_airmass(reservation, resources, slice_size):
    if reservation.request.optimization_type == OptimizationType.AIRMASS:
        for resource_name, resource_info in resources.items():
            if resource_name in reservation.possible_windows_dict:
                windows = reservation.possible_windows_dict[resource_name]
                window_tuples = windows.toTupleList()
                
                all_times = []
                for start, end in window_tuples:
                    # Calculate airmass every slice_size (5 minutes) within each window
                    window_duration = (end - start).total_seconds()
                    num_points = max(2, int(window_duration / slice_size))  # At least 2 points, then one every 5 minutes
                    times = [start + timedelta(seconds=i*window_duration/(num_points-1)) for i in range(num_points)]
                    all_times.extend(times)
                
                if all_times:
                    reservation.request.cache_airmass(resource_name, resource_info['location'], all_times)

def print_debug_info(globally_possible_windows_dict, reservations):
    total_available_time = sum(windows.get_total_time().total_seconds() for windows in globally_possible_windows_dict.values())
    #total_available_time = sum(windows.get_total_time() for windows in globally_possible_windows_dict.values())
    total_requested_time = sum(r.duration for cr in reservations for r in cr.reservation_list)
    
    print(f"Total available time across all resources: {total_available_time} seconds")
    print(f"Total requested observation time: {total_requested_time} seconds")
    print(f"Utilization ratio: {total_requested_time / total_available_time:.2%}")
    
    for resource, windows in globally_possible_windows_dict.items():
        print(f"\nResource: {resource}")
        print(f"Available time: {windows.get_total_time().total_seconds()} seconds")
        print("Visibility windows:")
        for window in windows.toTupleList():
            print(f"  {window[0]} to {window[1]}")
    
    total = 0
    print("\nReservation durations:")
    for cr in reservations:
        for r in cr.reservation_list:
            print(f"Reservation {r.resID}: {r.duration} seconds, Priority: {r.priority}")
            total = total + r.duration
    return total

# Main scheduling flow
def run_scheduler(resources, slice_size=300):
    with timing("run_scheduler"):
        with timing("Schedule Recorder"):
            schedule_recorder = ScheduleRecorder()
            
            # Prepare scheduler input
        with timing("Prepare Scheduler Input"):
            globally_possible_windows_dict, reservations, start_time = prepare_scheduler_input(
                resources, slice_size, schedule_recorder
            )
            
            # Initialize scheduler
        with timing("Initialize Scheduler"):
            scheduler = FullScheduler_ortoolkit(
                kernel='CBC',
                compound_reservation_list=reservations,
                globally_possible_windows_dict=globally_possible_windows_dict,
                contractual_obligation_list=[],
                slice_size_seconds=slice_size,
                mip_gap=0.05,
                warm_starts=False,
            )
            
            # Run scheduling
        with timing("ScheduleAll"):
            schedule = scheduler.schedule_all(timelimit=180)
            
            # Save the schedule
        with timing("Save Schedule"):
            schedule_recorder.save_schedule(schedule, calculated_at=datetime.utcnow())
        
        return schedule, schedule_recorder

#check_telescope_state()

resources = prepare_resources()
schedule, recorder = run_scheduler(resources)

# Upload to RTS2 if schedule is not empty
with timing("Upload Schedule to RTS2"):
    if any(observations for observations in schedule.values()):
        upload_schedule_to_rts2(schedule, "http://localhost:8889", "scheduler", "EdHijcabr2")

# Visualize if needed
with timing("Visualize the Schedule"):
    date=datetime.utcnow()
    visualize_schedule(schedule, resources, f"schedule-plot-{date.strftime('%Y%m%d-%H%M%S')}.png", connect_to_db())


