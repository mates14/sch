"""
Telescope Operations Module
Handles telescope state checking and RTS2 communication.
"""

import logging
import base64
import json
import urllib.request
import urllib.error
import psycopg2
from datetime import datetime, timedelta, timezone
from operator import attrgetter

logger = logging.getLogger(__name__)

# Never let a stuck telescope/proxy hang the whole scheduling run.
CENTRALD_TIMEOUT = 10  # seconds

# centrald state bits, from base/kernel/include/status.h
SERVERD_STATUS_MASK = 0x00f  # day/night phase
SERVERD_DUSK = 0x002
SERVERD_NIGHT = 0x003
SERVERD_ONOFF_MASK = 0x030  # on / standby / soft off / hard off
SERVERD_ON = 0x000
BAD_WEATHER = 0x80000000


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
        on_state = state_value & SERVERD_ONOFF_MASK
        day_state = state_value & SERVERD_STATUS_MASK

        # Available if ON and DUSK or NIGHT
        is_available = (on_state == SERVERD_ON
                        and day_state in (SERVERD_DUSK, SERVERD_NIGHT))

        logger.info(f"Telescope {name}: state=0x{state_value:x}, "
                    f"available={is_available}"
                    f"{' (bad weather)' if state_value & BAD_WEATHER else ''}")
        return is_available

    except Exception as e:
        logger.error(f"Error checking {name}: {e}")
        return False


def _base_urls(rts2_config):
    """
    Ordered list of API base URLs to try for one telescope.

    'url' is the primary route, 'url_fallback' (a string or a list of strings)
    optional alternatives to the same telescope - e.g. going through the other
    site's reverse proxy when the direct one is down.
    """
    urls = [rts2_config["url"]]

    fallback = rts2_config.get("url_fallback")
    if isinstance(fallback, str):
        urls.append(fallback)
    elif fallback:
        urls.extend(fallback)

    return urls


def _get_centrald_state(rts2_config):
    """
    Get centrald state from the RTS2 JSON API.

    The configured URL is the base of the API, and is used as given: it can be
    plain http straight to an rts2-httpd ("http://localhost:8889") or https to
    a reverse-proxied one, including under a path prefix
    ("https://lascaux50.asu.cas.cz/images").  rts2-httpd now binds localhost
    only, so anything remote goes through the site's Apache proxy.
    """
    last_error = None

    for base_url in _base_urls(rts2_config):
        try:
            return _fetch_centrald_state(base_url, rts2_config)
        except Exception as e:
            last_error = e
            logger.warning(f"centrald query via {base_url} failed: {e}")

    if last_error is not None:
        raise last_error
    return None


def _fetch_centrald_state(base_url, rts2_config):
    """Fetch centrald state from one specific API base URL."""
    url = base_url.rstrip('/') + '/api/get?d=centrald'

    auth = base64.b64encode(
        f"{rts2_config['user']}:{rts2_config['password']}".encode()
    ).decode().strip()

    request = urllib.request.Request(url, headers={
        "Authorization": f"Basic {auth}",
        "Accept": "application/json",
    })

    with urllib.request.urlopen(request, timeout=CENTRALD_TIMEOUT) as response:
        data = json.load(response)

    return data.get('state')


def upload_schedule_to_rts2(schedule, config, schedule_start_times, commit_horizon=None):
    """Upload schedule to RTS2 via direct database writes."""
    logger.info("Uploading schedule to RTS2")

    for telescope, observations in schedule.items():
        if not observations:
            continue

        telescope_start_time = schedule_start_times.get(telescope, datetime.utcnow())

        _upload_telescope_schedule(telescope, observations,
                                 config.get_resource_db_config(telescope),
                                 telescope_start_time,
                                 commit_horizon or telescope_start_time)


def _upload_telescope_schedule(telescope, observations, db_config, schedule_start_time, commit_horizon):
    """Upload schedule for a single telescope."""
    if not db_config:
        logger.error(f"No DB config for {telescope}")
        return

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Keep what we are committed to (entries starting before the commit
        # horizon, see database.get_committed_observations), drop the rest
        cursor.execute("""
            DELETE FROM queues_targets
            WHERE queue_id = 2
            AND (time_end <= NOW() OR time_start >= %s)
        """, (commit_horizon,))

        # A forced or GRB plan starts at the horizon, before the committed
        # entries end: cut them short so they do not overlap it
        cursor.execute("""
            UPDATE queues_targets SET time_end = %s
            WHERE queue_id = 2
            AND time_start < %s AND time_end > %s
        """, (schedule_start_time, schedule_start_time, schedule_start_time))

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

