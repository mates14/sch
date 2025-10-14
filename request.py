from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta, timezone
import logging
import re
import numpy as np
from math import ceil, cos, pi
from statistics import median
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
from psycopg2.extras import DictCursor

from kernel.reservation import OptimizationType
from simbad_magnitude_query import get_magnitude
from astro_utils import calculate_airmass

logger = logging.getLogger(__name__)

def parse_sinfo(sinfo_str):
    """Parse sinfo string into parameter dictionary."""
    if not sinfo_str:
        return {}

    params = {}
    for item in sinfo_str.split(','):
        if '=' in item:
            key, value = item.strip().split('=', 1)
            params[key.strip()] = value.strip()
    return params

class Request:
    def __init__(self, id, tar_ra, tar_dec, duration, state='PENDING', telescope_class='',
                 configuration_repeats=1, optimization_type=OptimizationType.AIRMASS,
                 scheduled_reservation=None, name=None, base_priority=100, comment=None,
                 bonus=None, bonus_time=None, next_observable=None, info=None,
                 interruptible=None, pm_ra=None, pm_dec=None, last_observation_time=None,
                 mag=None, sinfo="", telescope_name=None, type_id=None, grb_date=None):
        """
        Args:
            sinfo: sinfo string from this telescope's database
            telescope_name: Name of telescope this request is for
            type_id: Target type identifier (e.g., 'G' for GRB, 'O' for opportunity)
            grb_date: For GRB targets, the date/time of the GRB trigger
        """
        self.id = id
        self.tar_ra = tar_ra
        self.tar_dec = tar_dec
        self.mag = mag
        self.req_duration = duration
        self.state = state
        self.telescope_class = telescope_class
        self.configuration_repeats = configuration_repeats
        self.optimization_type = optimization_type
        self.scheduled_reservation = scheduled_reservation
        self.name = name
        self._base_priority = base_priority
        self.comment = comment
        self.bonus = bonus
        self.bonus_time = bonus_time
        self.next_observable = next_observable
        self.info = info
        self.interruptible = interruptible
        self.pm_ra = pm_ra
        self.pm_dec = pm_dec
        self.last_observation_time = last_observation_time
        self.airmass_data = {}
        self.moon_penalty_data = {}  # Cache moon penalties like airmass
        self.telescope_name = telescope_name
        self.type_id = type_id
        self.grb_date = grb_date

        # Parse sinfo for this telescope
        self.parsed_sinfo = parse_sinfo(sinfo)
        self.sinfo = sinfo

        self.log_details()

    def log_details(self):
        logger.debug(f"Created request: ID={self.id}, Telescope={self.telescope_name}, "
                     f"Base Priority={self._base_priority}, Duration={self.req_duration}s, "
                     f"RA={self.tar_ra:.5f}, Dec={self.tar_dec:.5f}, Name={self.name}, "
                     f"sinfo={self.sinfo}")

    @property
    def duration(self):
        return self.req_duration

    @property
    def base_priority(self):
        return self._base_priority

    def get_priority_timescale(self):
        """Get priority timescale from sinfo, default 1.0 day."""
        return float(self.parsed_sinfo.get('pscale', 1.0))

    def has_and_type(self):
        """Check if this request specifies type=and."""
        return self.parsed_sinfo.get('type') == 'and'

    def timescale_func(self, x):
        """
        Priority scaling function based on normalized time since last observation.

        Args:
            x: days_since_last_obs / timescale (normalized time)

        Returns:
            Priority multiplier (0.0 to 2.0)

        Behavior:
            - x < 1: Smooth rise from 0× to 2× priority (cosine curve)
            - 1 ≤ x < 2: Smooth decline from 2× to 1× priority
            - x ≥ 2: Constant 1× priority
        """
        if x < 1:
            # Rise from 0 to 2 over first timescale: -cos(x*π) + 1
            return -cos(x * pi) + 1
        elif x < 2:
            # Decline from 2 to 1 over second timescale: 1.5 - cos(x*π)/2
            return 1.5 - cos(x * pi) / 2
        else:
            # Constant priority after 2 timescales
            return 1.0

    def calculate_time_based_priority(self, start_time: datetime) -> float:
        """Calculate time-based priority factor using parameterized timescale."""
        if self.last_observation_time is None:
            return 2.0  # Double priority if never observed

        time_since_last_obs = start_time - self.last_observation_time
        days_since_last_obs = time_since_last_obs.total_seconds() / (24.0 * 3600)

        # Use parameterized timescale
        timescale = self.get_priority_timescale()

        # Adjust priority based on time since last observation
        priority_factor = self.timescale_func(days_since_last_obs / timescale)

        logger.debug(f"Request {self.id}: Base Priority={self._base_priority}, "
                     f"Days Since Last Obs={days_since_last_obs:.2f}, "
                     f"Timescale={timescale}, Priority Factor={priority_factor:.2f}")

        return priority_factor

    def cache_airmass(self, resource_name, resource_location, times):
        """Cache airmass data for later optimization."""
        airmass = calculate_airmass(self, resource_location, times)
        airmass = np.square(airmass / np.min(airmass))
        self.airmass_data[resource_name] = {'times': times, 'airmasses': airmass}

    def get_airmasses_within_kernel_windows(self, resource_name):
        """Get cached airmass data for the given resource."""
        return self.airmass_data.get(resource_name, {'times': [], 'airmasses': []})

    def cache_moon_penalties(self, resource_name, celestial_data, slice_centers, times,
                           min_distance=10.0, penalty_range=30.0):
        """Cache moon penalty data for later optimization."""
        from astro_utils import calculate_moon_penalty
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # Create target coordinate
        target_coord = SkyCoord(ra=self.tar_ra*u.deg, dec=self.tar_dec*u.deg)

        penalties = []
        for time in times:
            # Find corresponding slice index
            slice_idx = None
            for i, slice_center in enumerate(slice_centers):
                if abs((time - slice_center).total_seconds()) < 300:  # Within slice_size tolerance
                    slice_idx = i
                    break

            if slice_idx is not None:
                penalty = calculate_moon_penalty(
                    target_coord, celestial_data, resource_name, slice_idx,
                    min_distance, penalty_range
                )
                penalties.append(penalty)
            else:
                penalties.append(1.0)  # No penalty if time not found

        self.moon_penalty_data[resource_name] = {'times': times, 'penalties': penalties}

    def get_moon_penalties(self, resource_name):
        """Get cached moon penalty data for the given resource."""
        return self.moon_penalty_data.get(resource_name, {'times': [], 'penalties': []})

