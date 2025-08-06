"""
Astronomical Utilities Module
Handles visibility calculations, sun positions, horizons, and airmass.
"""

import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, HADec
import astropy.units as u
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def prepare_sun_positions(start_time, end_time, slice_size, resources):
    """Pre-compute sun positions for all time slices."""
    n_slices = int((end_time - start_time).total_seconds() / slice_size)
    
    slice_centers = [
        start_time + timedelta(seconds=slice_size * (i + 0.5))
        for i in range(n_slices)
    ]
    time_points = Time(slice_centers)
    
    sun_positions = {}
    for name, info in resources.items():
        if 'earth_location' not in info:
            info['earth_location'] = EarthLocation(
                lat=info['location']['latitude']*u.deg,
                lon=info['location']['longitude']*u.deg,
                height=info['location']['elevation']*u.m
            )
        
        observatory = info['earth_location']
        altaz_frame = AltAz(obstime=time_points, location=observatory)
        sun_altaz = get_sun(time_points).transform_to(altaz_frame)
        
        sun_positions[name] = {
            'altitudes': sun_altaz.alt.deg,
            'is_night': sun_altaz.alt < -15*u.deg,
            'frame': altaz_frame
        }
    
    return sun_positions, slice_centers


def calculate_visibility(target, resources, slice_centers, sun_positions, 
                        horizon_functions):
    """Calculate target visibility using pre-calculated data."""
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
            
            # Check horizon
            horizon_alt = horizon_func(target_altaz.az.degree)
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


def load_horizon(file_path, location, min_altitude=20.0):
    """Load horizon data and create interpolation function."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"Horizon file {file_path} not found, using flat {min_altitude}°")
        return lambda az: np.full_like(az, min_altitude)
    
    az, alt = [], []
    is_azalt = False
    
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('AZ-ALT'):
                is_azalt = True
                continue
            
            values = line.split()
            if is_azalt:
                az.append(float(values[0]))
                alt.append(max(float(values[1]), min_altitude))
            else:
                # HA-Dec format (legacy)
                ha = float(values[0]) * 15  # hours to degrees
                dec = float(values[1])
                # Would need conversion here
    
    # Ensure circular
    if az and az[0] != 360:
        az.append(360)
        alt.append(alt[0])
    
    return interp1d(np.array(az), np.array(alt), kind='linear', 
                   fill_value='extrapolate')


def calculate_airmass(target, location, times):
    """Calculate airmass for target at given times and location."""
    if target.tar_ra is None or target.tar_dec is None:
        return np.ones(len(times))
    
    target_coord = SkyCoord(ra=target.tar_ra*u.deg, dec=target.tar_dec*u.deg)
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
                if sunrise > start_time:
                    sunrise_times.append(sunrise)
                    break
    
    return max(sunrise_times) if sunrise_times else start_time + timedelta(hours=24)

