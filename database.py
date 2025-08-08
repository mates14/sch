"""
Database Operations Module
Handles all database connections and queries.
"""

import psycopg2
from psycopg2.extras import DictCursor
from contextlib import contextmanager
from datetime import datetime, timedelta
from statistics import median
from math import ceil
import logging
import re

from typing import List, Dict
from request import Request

from simbad_magnitude_query import get_magnitude

logger = logging.getLogger(__name__)


@contextmanager
def get_connection(db_config):
    """Context manager for database connections."""
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def parse_sinfo(sinfo_str):
    """Parse sinfo string into parameter dictionary."""
    if not sinfo_str:
        return {}

    params = {}
    for item in sinfo_str.split(' '):
        if '=' in item:
            key, value = item.strip().split('=', 1)
            params[key.strip()] = value.strip()
    return params

def _calculate_median_durations(conn):
    """Calculate median observation durations by target."""
    durations = {}

    query = """
        SELECT tar_id, EXTRACT(EPOCH FROM (obs_end - obs_start)) AS duration
        FROM observations
        WHERE obs_state = 0 AND obs_start IS NOT NULL AND obs_end IS NOT NULL
    """

    with conn.cursor() as cur:
        cur.execute(query)
        for tar_id, duration in cur:
            if tar_id not in durations:
                durations[tar_id] = []
            durations[tar_id].append(float(duration))

    return {tid: median(durs) for tid, durs in durations.items()}


def _determine_duration(row, median_durations, slice_size):
    """Determine observation duration for a target."""
    # Check sinfo first
    if row['sinfo']:
        match = re.search(r'duration=(\d+)', row['sinfo'])
        if match:
            duration = int(match.group(1))
            return ceil(duration / slice_size) * slice_size

    # Use median if available
    if row['tar_id'] in median_durations:
        duration = median_durations[row['tar_id']]
        return ceil(duration / slice_size) * slice_size

    # Default
    return 900  # 15 minutes


def fetch_actual_observations(conn, start_time, end_time):
    """Fetch actual observations for comparison with schedule."""
    query = """
        SELECT o.tar_id,
               i.img_date + make_interval(secs := i.img_exposure/2.0) as mid_time,
               i.img_alt, i.img_az
        FROM images i
        JOIN observations o ON i.obs_id = o.obs_id
        WHERE i.img_date BETWEEN %s AND %s
        AND i.delete_flag = 'f'
        ORDER BY i.img_date
    """

    observations = {}

    with conn.cursor() as cur:
        cur.execute(query, (start_time, end_time))
        for row in cur:
            tar_id = row[0]
            if tar_id not in observations:
                observations[tar_id] = {
                    'times': [], 'altitudes': [], 'azimuths': []
                }
            observations[tar_id]['times'].append(row[1])
            observations[tar_id]['altitudes'].append(row[2])
            observations[tar_id]['azimuths'].append(row[3])

    return observations

def fetch_requests(conn, telescope_name, last_noon):
    """Fetch all requests excluding currently executing target"""

    # Get currently executing observation info
    current_obs = get_current_executing_observation(conn)
    current_executing_tar_id = current_obs['tar_id'] if current_obs else None
    schedule_start_time = current_obs['time_end'] if current_obs else datetime.utcnow()

    cursor = conn.cursor(cursor_factory=DictCursor)

    # Query for GRB targets (type_id='G') - these have priority
    grb_query = """
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
        AND t.type_id = 'G'
    """

    # Query for opportunity targets (type_id='O')
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
    """

    # Add exclusion of currently executing target to both queries
    grb_params = []
    opp_params = [last_noon]

    if current_executing_tar_id:
        grb_query += " AND t.tar_id != %s"
        opportunity_query += " AND t.tar_id != %s"
        grb_params.append(current_executing_tar_id)
        opp_params.append(current_executing_tar_id)

    grb_query += " ORDER BY t.tar_priority DESC"
    opportunity_query += " ORDER BY t.tar_priority DESC"

    # Execute queries
    cursor.execute(grb_query, grb_params)
    grb_targets = cursor.fetchall()

    cursor.execute(opportunity_query, opp_params)
    opp_targets = cursor.fetchall()

    return grb_targets + opp_targets, schedule_start_time

def get_current_executing_observation(conn):
    """Get the currently executing observation"""
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute("""
        SELECT tar_id, time_end
        FROM queue_targets
        WHERE queue_id IN (1, 2)
        AND time_start < NOW()
        ORDER BY time_start DESC
        LIMIT 1
    """)
    return cursor.fetchone()


def _process_target_row(row, slice_size, median_durations, default_duration, telescope_id, is_grb=False):
    """Process a single target row (common logic for both O and G targets)"""
    tar_id = row['tar_id']
    tar_ra = row['tar_ra']
    type_id = row['type_id']

    if tar_ra is None:
        return None

    # Get sinfo from the scheduling table if available
    sinfo = row.get('sinfo','') or ""
    parsed_sinfo = parse_sinfo(sinfo)

    # Determine duration from sinfo if available
    if 'duration' in parsed_sinfo:
            duration = int(parsed_sinfo['duration'])
            logger.debug(f"Using sinfo duration {duration}s for target {row['tar_id']}")
    else:
        if is_grb:
            # GRB targets: use a standard duration (30 minutes)
            duration = 3600  # 30 minutes for GRB follow-up
            mag = None
        else:
            # Normal targets: calculate from magnitude
            name = row.get('tar_name','')
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
#        mag=mag,
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
        sinfo=sinfo,
        telescope_name=telescope_id,
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


def get_current_executing_observation(conn):
    """Get the currently executing observation"""
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute("""
        SELECT tar_id, time_end
        FROM queue_targets
        WHERE queue_id IN (1, 2)
        AND time_start < NOW()
        ORDER BY time_start DESC
        LIMIT 1
    """)
    return cursor.fetchone()  # Returns {'tar_id': 1008, 'time_end': datetime(...)} or None


def get_schedule_start_info(conn):
    """Get when to start scheduling and what target to exclude"""
    current = get_current_executing_observation(conn)

    if current:
        return current['time_end'], current['tar_id']
    else:
        # Nothing currently executing
        return datetime.utcnow(), None


def has_grb_targets(conn):
    """Check if there are schedulable GRB targets"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM targets
        WHERE type_id = 'G' AND enabled = true
        -- Add any other GRB-ready conditions
    """)
    return cursor.fetchone()[0] > 0
