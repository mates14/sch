"""
Telescope Operations Module
Handles telescope state checking and RTS2 communication.
"""

import logging
import http.client
import base64
import json
import psycopg2
from datetime import timedelta, timezone
from operator import attrgetter

logger = logging.getLogger(__name__)


def check_telescope_state(config):
    """Check state of all telescopes and return availability."""
    logger.info("Checking telescope states...")
    available_telescopes = {}
    
    for resource_name, resource_info in config.get_resources().items():
        available_telescopes[resource_name] = _check_single_telescope(
            resource_name, 
            config.get_resource_rts2_config(resource_name)
        )
    
    return available_telescopes


def _check_single_telescope(name, rts2_config):
    """Check a single telescope's state."""
    if not rts2_config:
        logger.warning(f"No RTS2 config for {name}")
        return False
    
    try:
        state_value = _get_centrald_state(rts2_config)
        if state_value is None:
            return False
        
        # Parse state
        on_state = (state_value & 0x30) >> 4
        day_state = state_value & 0x0f
        
        # Available if ON (0) and DUSK (2) or NIGHT (3)
        is_available = (on_state == 0 and day_state in [2, 3])
        
        logger.info(f"Telescope {name}: state=0x{state_value:x}, available={is_available}")
        return is_available
        
    except Exception as e:
        logger.error(f"Error checking {name}: {e}")
        return False


def _get_centrald_state(rts2_config):
    """Get centrald state from RTS2 JSON API."""
    url = rts2_config["url"].replace('http://', '')
    host_parts = url.split(':')
    host = host_parts[0]
    port = int(host_parts[1]) if len(host_parts) > 1 else 8889
    
    auth = base64.b64encode(
        f"{rts2_config['user']}:{rts2_config['password']}".encode()
    ).decode().strip()
    
    conn = http.client.HTTPConnection(host, port)
    conn.request("GET", "/api/get?d=centrald", None, 
                {"Authorization": f"Basic {auth}"})
    
    response = conn.getresponse()
    if response.status != 200:
        logger.error(f"HTTP error {response.status}")
        return None
    
    data = json.loads(response.read())
    return data.get('state')


def upload_schedule_to_rts2(schedule, config):
    """Upload schedule to RTS2 via direct database writes."""
    logger.info("Uploading schedule to RTS2")
    
    for telescope, observations in schedule.items():
        if not observations:
            continue
        
        _upload_telescope_schedule(telescope, observations, 
                                 config.get_resource_db_config(telescope))


def _upload_telescope_schedule(telescope, observations, db_config):
    """Upload schedule for a single telescope."""
    if not db_config:
        logger.error(f"No DB config for {telescope}")
        return
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Clear existing scheduler queue
        cursor.execute("DELETE FROM queues_targets WHERE queue_id = 2")
        
        # Get next qid
        cursor.execute("SELECT COALESCE(MAX(qid), 0) + 1 FROM queues_targets")
        next_qid = cursor.fetchone()[0]
        
        # Insert observations
        sorted_obs = sorted(observations, key=attrgetter('scheduled_start'))
        
        for i, obs in enumerate(sorted_obs):
            start = obs.scheduled_start
            end = start + timedelta(seconds=obs.scheduled_quantum)
            
            # Ensure timezone awareness
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            
            cursor.execute("""
                INSERT INTO queues_targets
                (qid, queue_id, tar_id, time_start, time_end, 
                 queue_order, repeat_n, repeat_separation)
                VALUES (%s, 2, %s, %s, %s, %s, -1, NULL)
            """, (next_qid + i, obs.request.id, start, end, i))
        
        conn.commit()
        logger.info(f"Uploaded {len(sorted_obs)} targets for {telescope}")
        
    except Exception as e:
        logger.error(f"Error uploading for {telescope}: {e}")
    finally:
        if conn:
            conn.close()

