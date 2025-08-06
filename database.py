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


def fetch_requests(conn, slice_size, telescope_id=None):
    """Fetch observation requests from database."""
    from request import Request
    from kernel.reservation import OptimizationType
    
    requests = []
    median_durations = _calculate_median_durations(conn)
    
    query = """
        SELECT t.tar_id, t.tar_ra, t.tar_dec, t.tar_priority, t.tar_name,
               t.tar_comment, t.tar_bonus, t.tar_bonus_time, t.tar_next_observable,
               t.tar_info, t.interruptible, t.tar_pm_ra, t.tar_pm_dec, 
               t.tar_telescope_mode,
               (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id) AS last_obs,
               s.sinfo
        FROM targets t
        LEFT JOIN scheduling s ON t.tar_id = s.tar_id
        WHERE t.tar_enabled = TRUE AND t.tar_id BETWEEN 1000 AND 49999
    """
    
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query)
        for row in cur:
            if row['tar_ra'] is None:
                continue
            
            duration = _determine_duration(row, median_durations, slice_size)
            
            requests.append(Request(
                id=row['tar_id'],
                tar_ra=row['tar_ra'],
                tar_dec=row['tar_dec'],
                duration=duration,
                name=row['tar_name'],
                base_priority=row['tar_priority'] or 1,
                last_observation_time=row['last_obs'],
                sinfo=row['sinfo']
            ))
    
    logger.info(f"Fetched {len(requests)} requests")
    return requests


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

