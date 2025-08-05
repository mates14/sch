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

def calculate_median_durations(conn) -> Dict[int, float]:
    """
    Calculate median durations for each target based on historical observations.
    """
    durations = {}
    query = """
        SELECT tar_id,
               EXTRACT(EPOCH FROM (obs_end - obs_start)) AS duration
        FROM observations
        WHERE obs_state IS NOT NULL AND obs_start IS NOT NULL AND obs_end IS NOT NULL
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

def fetch_requests(conn, slice_size: int, telescope_id="None") -> List[Request]:
    """
    Fetch observation requests from the database and create Request objects.
    Includes scheduling information from the scheduling table when available.
    Now handles both normal targets (type_id='O') and GRB targets (type_id='G').

    Args:
        conn: Database connection
        slice_size: Size of time slices in seconds
        telescope_id: Optional identifier for which telescope the requests came from

    Returns:
        List of Request objects
    """
    requests = []
    median_durations = calculate_median_durations(conn)
    default_duration = 600  # 10 minutes as default if no historical data

    # Calculate exclusion time (since last noon)
    now = datetime.utcnow()
    last_noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now.hour < 12:
        last_noon -= timedelta(days=1)

    # Query for normal opportunity targets (type_id='O')
    # Exclude targets observed since last noon
    opportunity_query = """
        SELECT t.tar_id, t.tar_ra, t.tar_dec, t.tar_priority, t.tar_name,
               t.tar_comment, t.tar_bonus, t.tar_bonus_time, t.tar_next_observable,
               t.tar_info, t.interruptible, t.tar_pm_ra, t.tar_pm_dec, t.tar_telescope_mode,
               t.type_id,
               (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) AS last_observation_time,
               s.sinfo
        FROM targets t
        LEFT JOIN scheduling s ON t.tar_id = s.tar_id
        WHERE
        t.tar_enabled = TRUE
        AND t.tar_id BETWEEN 1000 AND 49999
        AND t.type_id = 'O'
        AND (
            (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) IS NULL
            OR (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) < %s
        )
        ORDER BY t.tar_priority DESC
    """

    # Query for GRB targets (type_id='G') with grb table data
    # GRBs are NOT subject to the "since last noon" exclusion
    grb_query = """
        SELECT t.tar_id, t.tar_ra, t.tar_dec, t.tar_priority, t.tar_name,
               t.tar_comment, t.tar_bonus, t.tar_bonus_time, t.tar_next_observable,
               t.tar_info, t.interruptible, t.tar_pm_ra, t.tar_pm_dec, t.tar_telescope_mode,
               t.type_id,
               (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) AS last_observation_time,
               (SELECT MIN(obs_start) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) AS first_observation_time,
               g.grb_date, g.grb_id,
               s.sinfo
        FROM targets t
        LEFT JOIN scheduling s ON t.tar_id = s.tar_id
        INNER JOIN grb g ON t.tar_id = g.tar_id
        WHERE
        t.tar_enabled = TRUE
        AND t.tar_id BETWEEN 1000 AND 49999
        AND t.type_id = 'G'
        ORDER BY t.tar_priority DESC
    """

    grb_count = 0
    opportunity_count = 0

    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Process opportunity targets first (with last_noon exclusion)
            cur.execute(opportunity_query, (last_noon,))
            for row in cur:
                request = _process_target_row(row, slice_size, median_durations, default_duration, False)
                if request:
                    requests.append(request)
                    opportunity_count += 1

            # Process GRB targets with time-based priority (no exclusion)
            cur.execute(grb_query)
            for row in cur:
                request = _process_grb_row(row, slice_size, median_durations, default_duration)
                if request:
                    requests.append(request)
                    grb_count += 1

    except Exception as e:
        logger.error(f"Error fetching requests: {e}")
        raise

    logger.info(f"Fetched {len(requests)} requests: {grb_count} active GRB targets, {opportunity_count} opportunity targets")
    logger.info(f"Excluded opportunity targets observed since: {last_noon}")

    if grb_count > 0:
        logger.warning(f"*** {grb_count} active GRB target(s) found with time-scaled high priority! ***")

    return requests


def _process_target_row(row, slice_size, median_durations, default_duration, is_grb=False):
    """Process a single target row (common logic for both O and G targets)"""
    tar_id = row['tar_id']
    tar_ra = row['tar_ra']
    type_id = row['type_id']

    if tar_ra is None:
        return None

    # Get sinfo from the scheduling table if available
    sinfo = row['sinfo']

    # Determine duration from sinfo if available
    duration = None
    mag = None
    if sinfo:
        # Parse sinfo for duration
        duration_match = re.search(r'duration=(\d+)', sinfo)
        if duration_match:
            duration = int(duration_match.group(1))

    name = row['tar_name']

    # If no duration from sinfo, calculate it the usual way
    if duration is None:
        if is_grb:
            # GRB targets: use a standard duration (30 minutes)
            duration = 1800  # 30 minutes for GRB follow-up
            mag = None
        else:
            # Normal targets: calculate from magnitude
            mag = get_magnitude(name[3:] if name.startswith('V* ') else name)

            # Properly handle masked magnitude values
            if mag is None or hasattr(mag, 'mask') and mag.mask:
                duration = 900  # Default duration for objects without magnitude
                mag = None  # Explicitly set to None to avoid masked array issues
            else:
                try:
                    exptime = 60 * 10**((mag - 15.0)/1.25)  # 30 sigma
                    reqtime = exptime * 1.1  # 30 sigma
                    duration = ceil(reqtime/300)*300
                except (TypeError, ValueError):
                    duration = 900

    base_priority = row['tar_priority']
    if base_priority is None:
        print("Caught priority None!")
        base_priority = 1

    # round up duration to slice_size
    duration = int(ceil(duration / slice_size) * slice_size)

    request = Request(
        id=tar_id,
        tar_ra=tar_ra,
        tar_dec=row['tar_dec'],
        mag=mag,
        duration=duration,
        state='PENDING',
        telescope_class=row['tar_telescope_mode'] or '',
        name=row['tar_name'],
        base_priority=base_priority,
        comment=row['tar_comment'],
        bonus=row['tar_bonus'],
        bonus_time=row['tar_bonus_time'],
        next_observable=row['tar_next_observable'],
        info=row['tar_info'],
        interruptible=row['interruptible'],
        pm_ra=row['tar_pm_ra'],
        pm_dec=row['tar_pm_dec'],
        sinfo=sinfo,  # Add the scheduling info to the request object
        last_observation_time=row['last_observation_time']
    )
    return request


def _process_grb_row(row, slice_size, median_durations, default_duration):
    """Process a GRB target row with time-based priority calculation"""
    # First process as normal target
    request = _process_target_row(row, slice_size, median_durations, default_duration, is_grb=True)
    if not request:
        return None

    # Now apply GRB-specific priority boost
    grb_date = row['grb_date']
    grb_id = row['grb_id']
    first_obs_time = row['first_observation_time']

    if grb_date is None:
        logger.warning(f"GRB target {request.id} has no grb_date, skipping")
        return None

    # Calculate time since GRB trigger
    now = datetime.utcnow()
    if grb_date.tzinfo is None:
        grb_date = grb_date.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    time_since_grb = now - grb_date
    hours_since_grb = time_since_grb.total_seconds() / 3600.0

    # Determine follow-up cutoff time
    if first_obs_time is None and hours_since_grb > 1.0:
        # GRB > 1h old and never observed: follow up for only 1 more hour
        cutoff_hours = hours_since_grb + 1.0
        follow_up_reason = "late discovery"
    else:
        # Standard case: follow up until 2h post trigger
        cutoff_hours = 2.0
        follow_up_reason = "standard"

    # Skip if past cutoff time
    if hours_since_grb > cutoff_hours:
        logger.info(f"GRB {grb_id} (target {request.id}) is {hours_since_grb:.2f}h old, "
                   f"past {follow_up_reason} cutoff of {cutoff_hours:.1f}h, skipping")
        return None

    # Calculate time-scaled priority: 100 / hours_since_trigger
    # Minimum 0.1 hours to avoid division issues for very fresh GRBs
    effective_hours = max(0.1, hours_since_grb)
    priority_multiplier = 100.0 / effective_hours
    request._base_priority = int(request._base_priority * priority_multiplier)

    logger.info(f"GRB {grb_id} (target {request.id}, {request.name}): "
               f"{hours_since_grb:.2f}h old, priority boost {priority_multiplier:.1f}x "
               f"-> final priority {request._base_priority} ({follow_up_reason} follow-up)")

    return request

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
               (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) AS last_observation_time
        FROM targets t
        WHERE t.tar_enabled = TRUE AND t.tar_id BETWEEN 1000 AND 49999
        AND (
            (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) IS NULL
            OR (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id AND obs_state IS NOT NULL) < %s
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
