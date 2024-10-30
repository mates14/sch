from typing import Dict, List
from datetime import datetime, timedelta
import logging
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
                 interruptible=None, pm_ra=None, pm_dec=None, last_observation_time=None, mag=None):
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
        # This is a simple linear increase, you might want to use a more sophisticated formula
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
        airmass = self.calculate_airmass(resource_name, resource_location, times)
        airmass = np.square(airmass / np.min(airmass)) # be strong in requesting a good airmass :)
        self.airmass_data[resource_name] = {'times': times, 'airmasses': airmass}

    def get_airmasses_within_kernel_windows(self, resource_name):
        return self.airmass_data.get(resource_name, {'times': [], 'airmasses': []})

def calculate_median_durations(conn) -> Dict[int, float]:
    """
    Calculate median durations for each target based on historical observations.
    """
    durations = {}
    query = """
        SELECT tar_id, 
               EXTRACT(EPOCH FROM (obs_end - obs_start)) AS duration
        FROM observations
        WHERE obs_state = 0 AND obs_start IS NOT NULL AND obs_end IS NOT NULL
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur:
                tar_id, duration = row
                duration = float(duration)
                if tar_id not in durations:
                    durations[tar_id] = []
                durations[tar_id].append(duration)

        median_durations = {tar_id: median(dur_list) for tar_id, dur_list in durations.items()}
        logger.info(f"Calculated median durations for {len(median_durations)} targets")
        return median_durations

    except Exception as e:
        logger.error(f"Error calculating median durations: {e}")
        raise

def fetch_requests_unobserved(conn, slice_size: int) -> List[Request]:
    """
    Fetch observation requests from the database and create Request objects.
    Exclude targets that have been observed since the last noon.
    """
    requests = []
    median_durations = calculate_median_durations(conn)
    default_duration = 1800  # 30 minutes as default if no historical data

    # Calculate the timestamp for the last noon
    now = datetime.utcnow()
    last_noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now.hour < 12:
        last_noon -= timedelta(days=1)

    query = """
        SELECT t.tar_id, t.tar_ra, t.tar_dec, t.tar_priority, t.tar_name, 
               t.tar_comment, t.tar_bonus, t.tar_bonus_time, t.tar_next_observable, 
               t.tar_info, t.interruptible, t.tar_pm_ra, t.tar_pm_dec, t.tar_telescope_mode,
               (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state = 0) AS last_observation_time
        FROM targets t
        WHERE t.tar_enabled = TRUE AND t.tar_id BETWEEN 1000 AND 49999
        AND (
            (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state = 0) IS NULL
            OR (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state = 0) < %s
        )
    """
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (last_noon,))
            for row in cur:
                tar_id = row['tar_id']
                tar_ra = row['tar_ra']

                if tar_ra is None:
                    continue

                duration = median_durations.get(tar_id, default_duration)
                duration = int(ceil(duration / slice_size) * slice_size)
                    
                name = row['tar_name']
                mag = get_magnitude(name[3:] if name.startswith('V* ') else name),

                request = Request(
                    request_id=tar_id,
                    tar_ra=tar_ra,
                    tar_dec=row['tar_dec'],
                    mag = mag,
                    duration=duration,
                    state='PENDING',
                    telescope_class=row['tar_telescope_mode'] or '',
                    name=row['tar_name'],
                    base_priority=row['tar_priority'],
                    comment=row['tar_comment'],
                    bonus=row['tar_bonus'],
                    bonus_time=row['tar_bonus_time'],
                    next_observable=row['tar_next_observable'],
                    info=row['tar_info'],
                    interruptible=row['interruptible'],
                    pm_ra=row['tar_pm_ra'],
                    pm_dec=row['tar_pm_dec'],
                    last_observation_time=row['last_observation_time']
                )
                requests.append(request)
    except Exception as e:
        logger.error(f"Error fetching requests: {e}")
        raise
    logger.info(f"Fetched {len(requests)} requests")

    return requests


def fetch_requests(conn, slice_size: int) -> List[Request]:
    """
    Fetch observation requests from the database and create Request objects.
    """
    requests = []
    median_durations = calculate_median_durations(conn)
    default_duration = 1800  # 30 minutes as default if no historical data
    query = """
        SELECT t.tar_id, t.tar_ra, t.tar_dec, t.tar_priority, t.tar_name, 
               t.tar_comment, t.tar_bonus, t.tar_bonus_time, t.tar_next_observable, 
               t.tar_info, t.interruptible, t.tar_pm_ra, t.tar_pm_dec, t.tar_telescope_mode,
               (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id) AS last_observation_time
        FROM targets t
        WHERE 
        t.tar_enabled = TRUE
        AND t.tar_id BETWEEN 1000 AND 49999
    """
        #AND t.tar_priority > 0
        #AND t.tar_ra > 0
        #AND t.tar_dec > 0
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query)
            for row in cur:
                tar_id = row['tar_id']
                tar_ra = row['tar_ra']

                if tar_ra is None:
                    continue

                #duration = median_durations.get(tar_id, default_duration)
                #duration = int(ceil(duration / slice_size) * slice_size)

                name = row['tar_name']
                mag = get_magnitude(name[3:] if name.startswith('V* ') else name)

                try:
                    exptime = 60 * 10**(( mag - 15.0 )/1.25) # 30 sigma
                    reqtime = exptime * 1.1 # 30 sigma
                    print(name, mag, reqtime)
                    duration = ceil(reqtime/300)*300
                except (TypeError, ValueError): # get_magnitude returns some None, NoneType or so...
                    # duration = median_durations.get(tar_id, default_duration)
                    # duration = int(ceil(duration / slice_size) * slice_size)
                    duration = 900

                request = Request(
                    id=tar_id,
                    tar_ra=tar_ra,
                    tar_dec=row['tar_dec'],
                    mag = mag,
                    duration=duration,
                    state='PENDING',
                    telescope_class=row['tar_telescope_mode'] or '',
                    name=name,
                    base_priority=row['tar_priority'],
                    comment=row['tar_comment'],
                    bonus=row['tar_bonus'],
                    bonus_time=row['tar_bonus_time'],
                    next_observable=row['tar_next_observable'],
                    info=row['tar_info'],
                    interruptible=row['interruptible'],
                    pm_ra=row['tar_pm_ra'],
                    pm_dec=row['tar_pm_dec'],
                    last_observation_time=row['last_observation_time']
                )
                requests.append(request)
                try: magstr=f"{request.mag:4.1f}"
                except: magstr=" -- "
#                print(f"Request: id:{request.id} ra:{request.tar_ra:.3f} dec:{request.tar_dec:.3f} {magstr} {request.duration:.0f} {request.name}")
    except Exception as e:
        logger.error(f"Error fetching requests: {e}")
        raise
    logger.info(f"Fetched {len(requests)} requests")
    return requests
