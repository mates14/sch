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

logger = logging.getLogger(__name__)

class Request:
    def __init__(self, id, tar_ra, tar_dec, duration, state='PENDING', telescope_class='',
                 configuration_repeats=1, optimization_type=OptimizationType.AIRMASS,
                 scheduled_reservation=None, name=None, base_priority=100, comment=None,
                 bonus=None, bonus_time=None, next_observable=None, info=None,
                 interruptible=None, pm_ra=None, pm_dec=None, last_observation_time=None, mag=None, sinfo=""):
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
        self.sinfo = sinfo  # scheduling info from the database

        self.log_details()

    def log_details(self):
        logger.debug(f"Created request: ID={self.id}, Base Priority={self._base_priority}, "
                     f"Duration={self.req_duration}s, RA={self.tar_ra:.5f}, "
                     f"Dec={self.tar_dec:.5f}, Name={self.name}, "
                     f"Last Observation={self.last_observation_time}")

    @property
    def duration(self):
        return self.req_duration

    @property
    def base_priority(self):
        return self._base_priority

    def calculate_time_based_priority(self, start_time: datetime) -> float:
        """
        Calculate the time-based priority factor for a given start time.
        """
        if self.last_observation_time is None:
            return 2.0  # Double priority if never observed

        time_since_last_obs = start_time - self.last_observation_time
        days_since_last_obs = time_since_last_obs.total_seconds() / (24 * 3600)

        # Adjust priority based on time since last observation
        if days_since_last_obs < 1: priority_factor = 2*days_since_last_obs-0.5 # (if < 0.5d -> negative priority!)
        elif days_since_last_obs < 31: priority_factor = 1 + ((days_since_last_obs-1) / 30)
        else: priority_factor = 2
#        priority_factor = 1 + ((days_since_last_obs-1) / 30)  # Increase by 100% after 30 days

        logger.debug(f"Request {self.id}: Base Priority={self._base_priority}, "
                     f"Days Since Last Obs={days_since_last_obs:.2f}, "
                     f"Time-based Priority Factor={priority_factor:.2f}")

#        return 1
        return priority_factor

    def calculate_airmass(self, resource_name, resource_location, times):
        """Calculate airmass for the target at given times and location."""
        if self.tar_ra is None or self.tar_dec is None:
            return np.ones(len(times))

        target = SkyCoord(ra=self.tar_ra*u.deg, dec=self.tar_dec*u.deg)
        location = EarthLocation(lat=resource_location['latitude']*u.deg,
                                 lon=resource_location['longitude']*u.deg,
                                 height=resource_location['elevation']*u.m)

        astro_times = Time(times, format='datetime')

        altaz = target.transform_to(AltAz(obstime=astro_times, location=location))

        airmass = altaz.secz
        return airmass.value

    def cache_airmass(self, resource_name, resource_location, times):
        """Cache airmass data for later optimization."""
        airmass = self.calculate_airmass(resource_name, resource_location, times)
        airmass = np.square(airmass / np.min(airmass)) # be strong in requesting a good airmass :)
        self.airmass_data[resource_name] = {'times': times, 'airmasses': airmass}

    def get_airmasses_within_kernel_windows(self, resource_name):
        """Get cached airmass data for the given resource."""
        return self.airmass_data.get(resource_name, {'times': [], 'airmasses': []})

