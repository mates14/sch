"""
Astronomical Utilities Module
Handles visibility calculations, sun positions, horizons, and airmass.
"""

import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon, HADec
import astropy.units as u
from pathlib import Path
import logging

# for timing
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

@contextmanager
def timing(name):
    start = time.time()
    yield
    end = time.time()
    print(f"timing {name}: {(end - start):.1f}s")


def ensure_earth_locations(resources):
    """Ensure all resources have earth_location objects."""

    for name, info in resources.items():
        if 'earth_location' not in info:
            info['earth_location'] = EarthLocation(
                lat=info['location']['latitude']*u.deg,
                lon=info['location']['longitude']*u.deg,
                height=info['location']['elevation']*u.m
            )
    return resources


def prepare_sun_positions(start_time, end_time, slice_size, resources, night_horizon=-10):
    """Pre-compute sun positions for all time slices. (Legacy wrapper)"""
    celestial_data, slice_centers = prepare_celestial_positions(start_time, end_time, slice_size, resources, night_horizon)
    # Extract just sun data for backward compatibility
    sun_positions = {}
    for name in celestial_data:
        sun_positions[name] = {
            'altitudes': celestial_data[name]['sun']['altitudes'],
            'is_night': celestial_data[name]['sun']['is_night'],
            'frame': celestial_data[name]['frame']
        }
    return sun_positions, slice_centers

def prepare_celestial_positions(start_time, end_time, slice_size, resources, night_horizon=-10):
    """Pre-compute sun and moon positions for all time slices."""
    # Ensure earth locations exist
    resources = ensure_earth_locations(resources)

    n_slices = int((end_time - start_time).total_seconds() / slice_size)

    slice_centers = [
        start_time + timedelta(seconds=slice_size * (i + 0.5))
        for i in range(n_slices)
    ]
    time_points = Time(slice_centers)

    celestial_data = {}
    for name, info in resources.items():
        observatory = info['earth_location']
        altaz_frame = AltAz(obstime=time_points, location=observatory)

        # Calculate sun positions
        sun_altaz = get_sun(time_points).transform_to(altaz_frame)

        # Calculate moon positions
        moon_altaz = get_moon(time_points).transform_to(altaz_frame)

        celestial_data[name] = {
            'sun': {
                'altitudes': sun_altaz.alt.deg,
                'is_night': sun_altaz.alt < night_horizon*u.deg,
                'coordinates': sun_altaz
            },
            'moon': {
                'altitudes': moon_altaz.alt.deg,
                'azimuths': moon_altaz.az.deg,
                'coordinates': moon_altaz,
                'is_up': moon_altaz.alt > 0*u.deg
            },
            'frame': altaz_frame
        }

    return celestial_data, slice_centers


def calculate_moon_penalty(target_coord, celestial_data, resource_name, slice_idx,
                          min_distance=10.0, penalty_range=30.0):
    """
    Calculate moon distance penalty for target priority.

    Args:
        target_coord: Target SkyCoord
        celestial_data: Full celestial data dict
        resource_name: Telescope name
        slice_idx: Time slice index
        min_distance: Hard limit - no observations within this distance (degrees)
        penalty_range: Distance where penalties apply (degrees)

    Returns:
        penalty_factor: Multiplier for priority (0 = blocked, 1 = no penalty)
    """
    moon_data = celestial_data[resource_name]['moon']

    # Skip if moon is below horizon
    if not moon_data['is_up'][slice_idx]:
        return 1.0

    # Get moon coordinates for this time slice
    moon_altaz = moon_data['coordinates'][slice_idx]

    # Transform target to same frame
    altaz_frame = celestial_data[resource_name]['frame'][slice_idx]
    target_altaz = target_coord.transform_to(altaz_frame)

    # Calculate angular separation
    moon_distance = target_altaz.separation(moon_altaz).deg

    # Hard block within minimum distance
    if moon_distance < min_distance:
        return 0.0  # Completely blocked

    # Gradual penalty within penalty range
    if moon_distance < penalty_range:
        # Linear penalty from 0 to 1 as distance increases from min to penalty_range
        penalty_factor = (moon_distance - min_distance) / (penalty_range - min_distance)
        # Square it for stronger penalty closer to moon
        penalty_factor = penalty_factor ** 2
        return max(0.01, penalty_factor)  # Minimum 1% priority to avoid division issues

    # No penalty beyond penalty range
    return 1.0


def calculate_visibility(target, resources, slice_centers, sun_positions,
                        horizon_functions, slice_size):
    """Calculate target visibility using pre-calculated data.
    FIXED: Properly handles coordinate system conversion for horizon checks.
    """
    visibility_intervals = {}

    for resource_name, resource_info in resources.items():
        sun_low = sun_positions[resource_name]['is_night']
        horizon_func = horizon_functions[resource_name]

        if target is not None:
            # Calculate target position
            time_points = Time(slice_centers)
            observatory = resource_info['earth_location']
            altaz_frame = AltAz(obstime=time_points, location=observatory)

            target_coord = SkyCoord(
                ra=target.tar_ra*u.degree,
                dec=target.tar_dec*u.degree,
                frame='icrs'
            )
            target_altaz = target_coord.transform_to(altaz_frame)

            # FIXED: COORDINATE SYSTEM CONVERSION
            # Convert from astropy compass coordinates (0° = North)
            # to astronomical coordinates (0° = South) used by horizon file
            az_compass = target_altaz.az.degree
            az_astronomical = (az_compass + 180) % 360

            # Check horizon limits using converted coordinates
            horizon_alt = horizon_func(az_astronomical)
            target_high = target_altaz.alt.degree > horizon_alt

            is_visible = sun_low & target_high
        else:
            is_visible = sun_low

        # Convert to time windows
        visibility_intervals[resource_name] = _bool_array_to_intervals(
            is_visible, slice_centers, slice_size
        )

    return visibility_intervals


def _bool_array_to_intervals(visible_array, slice_centers, slice_size):
    """Convert boolean visibility array to time intervals."""
    intervals = []
    start_idx = None

    for i, visible in enumerate(visible_array):
        if visible and start_idx is None:
            start_idx = i
        elif not visible and start_idx is not None:
            intervals.append((slice_centers[start_idx], slice_centers[i]))
            start_idx = None

    if start_idx is not None:
        intervals.append((
            slice_centers[start_idx],
            slice_centers[-1] + timedelta(seconds=slice_size)
        ))

    return intervals

def create_default_horizon_from_altitude(min_altitude):
    """Create a simple horizon function from minimum altitude.

    Returns a horizon that's just min_altitude at all azimuths.
    """
    def horizon_function(az):
        """Simple horizon: constant altitude at all azimuths."""
        return min_altitude

    logger.info(f"Created default horizon with constant altitude {min_altitude}°")
    return horizon_function


def load_horizon(horizon_file: str, location: EarthLocation, min_altitude: float = 20.0) -> interp1d:
    """
    Load horizon data from a file and create an interpolation function.
    FIXED: Properly handles coordinate system transformations and boundary conditions.

    Args:
        horizon_file: Path to horizon file
        location: Observatory location (unused if file is already AZ-ALT)
        min_altitude: Minimum allowed altitude in degrees (default: 20.0)

    Returns:
        Interpolation function that returns horizon altitude for astronomical azimuth
        (0°=South), ensuring returned values are never below min_altitude
    """
    if horizon_file is None:
        logger.info(f"No horizon file, using constant min_altitude {min_altitude}°")
        return create_default_horizon_from_altitude(min_altitude)

    try:
        az = []
        alt = []
        ha = []
        dec = []
        is_azalt = False

        # Read and classify points
        with timing("file reading"):
            with open(horizon_file, 'r') as file:
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
                # Apply minimum altitude constraint and convert to astronomical coordinates
                alt = list(np.maximum(alt_array, min_altitude))
                # FIXED: Convert from astropy compass coordinates to astronomical coordinates
                az = [(a + 180) % 360 for a in az]

        # FIXED: Robust circular condition handling (restored from original)
        with timing("array conversion"):
            # Convert to numpy arrays for easier manipulation
            az = np.array(az)
            alt = np.array(alt)

            # Sort by azimuth
            sort_idx = np.argsort(az)
            az = az[sort_idx]
            alt = alt[sort_idx]

            # Check for 0° point
            has_zero = np.any(np.isclose(az, 0.0, atol=1e-6))

            # Check for 360° point
            has_360 = np.any(np.isclose(az, 360.0, atol=1e-6))

            # Handle missing endpoints
            if not has_zero:
                # Find altitude at 0° by interpolating from existing data
                # Wrap around: use points near 360° and near 0°
                if len(az) > 1:
                    # Simple approach: use the altitude from the last point (closest to 360°)
                    alt_at_zero = alt[-1]
                else:
                    alt_at_zero = min_altitude
                az = np.concatenate([[0.0], az])
                alt = np.concatenate([[alt_at_zero], alt])

            if not has_360:
                # Find altitude at 360° - should be same as at 0°
                zero_idx = np.where(np.isclose(az, 0.0, atol=1e-6))[0][0]
                alt_at_360 = alt[zero_idx]
                az = np.concatenate([az, [360.0]])
                alt = np.concatenate([alt, [alt_at_360]])

        with timing("interpolation setup"):
            result = interp1d(az, alt, kind='linear', fill_value='extrapolate')

        print(f"horizon: {len(az)} points, ", 'azalt' if is_azalt else 'hadec', " format")
        print(f"minimum altitude set to {min_altitude} degrees")
        print(f"azimuth range: {az.min():.1f}° to {az.max():.1f}°")
        return result

    except (FileNotFoundError, IOError) as e:
        logger.warning(f"Could not load horizon file {horizon_file}: {e}")
        logger.info(f"Falling back to constant min_altitude {min_altitude}°")
        return create_default_horizon_from_altitude(min_altitude)


def calculate_airmass(target, location, times):
    """Calculate airmass for target at given times and location.

    Args:
        target: Target object with tar_ra and tar_dec
        location: Either an EarthLocation object or a dict with lat/lon/elevation
        times: List of datetime objects
    """
    if target.tar_ra is None or target.tar_dec is None:
        return np.ones(len(times))

    target_coord = SkyCoord(ra=target.tar_ra*u.deg, dec=target.tar_dec*u.deg)

    # Handle both EarthLocation objects and location dicts
    if isinstance(location, EarthLocation):
        earth_loc = location
    else:
        earth_loc = EarthLocation(
            lat=location['latitude']*u.deg,
            lon=location['longitude']*u.deg,
            height=location['elevation']*u.m
        )

    astro_times = Time(times, format='datetime')
    altaz = target_coord.transform_to(AltAz(obstime=astro_times, location=earth_loc))

    return altaz.secz.value


def find_next_sunrise(start_time, resources, slice_size=300):
    """Find the next sunrise time across all resources."""
    sunrise_times = []

    for name, info in resources.items():
        location = info['earth_location']

        for day_offset in [0, 1]:
            date = start_time.date() + timedelta(days=day_offset)
            midnight = datetime.combine(date, datetime.min.time())

            times = Time(midnight) + np.linspace(0, 24, 288)*u.hour
            sun_altaz = get_sun(times).transform_to(
                AltAz(location=location, obstime=times)
            )
            sun_up = sun_altaz.alt > 0*u.deg

            if np.any(sun_up):
                sunrise_idx = np.where(sun_up)[0][0]
                sunrise = times[sunrise_idx].datetime
                # Strip timezone info to work in UTC-naive environment
                if sunrise.tzinfo is not None:
                    sunrise = sunrise.replace(tzinfo=None)
                if start_time.tzinfo is not None:
                    start_time = start_time.replace(tzinfo=None)
                if sunrise > start_time:
                    sunrise_times.append(sunrise)
                    break

    return max(sunrise_times) if sunrise_times else start_time + timedelta(hours=24)

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

