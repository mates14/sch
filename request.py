from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta, timezone
import logging
import re
import numpy as np
from math import ceil
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
                 mag=None, sinfo="", telescope_name=None):
        """
        Args:
            sinfo: sinfo string from this telescope's database
            telescope_name: Name of telescope this request is for
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
        self.telescope_name = telescope_name
        
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
        """Get priority timescale from sinfo, default 30 days."""
        return float(self.parsed_sinfo.get('pscale', 30))
    
    def has_and_type(self):
        """Check if this request specifies type=and."""
        return self.parsed_sinfo.get('type') == 'and'

    def calculate_time_based_priority(self, start_time: datetime) -> float:
        """Calculate time-based priority factor using parameterized timescale."""
        if self.last_observation_time is None:
            return 2.0  # Double priority if never observed

        time_since_last_obs = start_time - self.last_observation_time
        days_since_last_obs = time_since_last_obs.total_seconds() / (24 * 3600)
        
        # Use parameterized timescale
        timescale = self.get_priority_timescale()

        # Adjust priority based on time since last observation
        if days_since_last_obs < 1: 
            priority_factor = 2*days_since_last_obs - 0.5
        elif days_since_last_obs < timescale: 
            priority_factor = 1 + ((days_since_last_obs-1) / timescale)
        else: 
            priority_factor = 2

        logger.debug(f"Request {self.id}: Base Priority={self._base_priority}, "
                     f"Days Since Last Obs={days_since_last_obs:.2f}, "
                     f"Timescale={timescale}, Priority Factor={priority_factor:.2f}")

        return priority_factor

    def cache_airmass(self, resource_name, resource_location, times):
        """Cache airmass data for later optimization."""
        airmass = self.calculate_airmass(resource_name, resource_location, times)
        airmass = np.square(airmass / np.min(airmass))
        self.airmass_data[resource_name] = {'times': times, 'airmasses': airmass}

    def get_airmasses_within_kernel_windows(self, resource_name):
        """Get cached airmass data for the given resource."""
        return self.airmass_data.get(resource_name, {'times': [], 'airmasses': []})