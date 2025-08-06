import logging
import psycopg2
from psycopg2.extras import DictCursor
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime

from request import Request, parse_sinfo
from kernel.reservation import Reservation, CompoundReservation
from simbad_magnitude_query import get_magnitude

logger = logging.getLogger(__name__)

@contextmanager
def get_connection(db_config):
    """Create database connection from config."""
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host']
    )
    try:
        yield conn
    finally:
        conn.close()

def calculate_duration_from_magnitude(magnitude, telescope_name=None):
    """Calculate exposure duration from magnitude using existing logic."""
    if magnitude is None:
        return 300  # Default 5 minutes
    
    # From simbad_magnitude_query.py logic
    exptime = 60 * 10**((magnitude - 15.0) / 1.25)  # 10 sigma
    reqtime = exptime * 14 / 6
    
    return int(reqtime)

def load_requests_from_telescope(telescope_name, telescope_config, available_telescopes):
    """
    Load requests from a single telescope database with sinfo support.
    
    Args:
        telescope_name: Name of telescope
        telescope_config: Telescope configuration dict
        available_telescopes: Dict of telescope availability
    
    Returns:
        List of Request objects for this telescope
    """
    if not available_telescopes.get(telescope_name, False):
        return []
    
    db_config = telescope_config['database']
    requests = []
    
    with get_connection(db_config) as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Query targets with optional sinfo
            query = """
                SELECT t.tar_id, t.tar_ra, t.tar_dec, t.tar_name, t.tar_priority,
                       t.tar_comment, t.tar_bonus, t.tar_bonus_time, t.tar_pm_ra, t.tar_pm_dec,
                       t.tar_mag, s.sinfo,
                       (SELECT MAX(obs_end) FROM observations WHERE tar_id = t.tar_id) as last_obs
                FROM targets t
                LEFT JOIN scheduling s ON t.tar_id = s.tar_id
                WHERE t.tar_enabled = 't'
            """
            cur.execute(query)
            
            for row in cur.fetchall():
                sinfo = row['sinfo'] or ""
                parsed_sinfo = parse_sinfo(sinfo)
                
                # Calculate duration: sinfo first, then magnitude fallback
                if 'duration' in parsed_sinfo:
                    duration = int(parsed_sinfo['duration'])
                    logger.debug(f"Using sinfo duration {duration}s for target {row['tar_id']}")
                else:
                    magnitude = row['tar_mag']
                    if magnitude is None and row['tar_name']:
                        # Try SIMBAD lookup as fallback
                        magnitude = get_magnitude(row['tar_name'])
                    
                    duration = calculate_duration_from_magnitude(magnitude, telescope_name)
                    logger.debug(f"Calculated duration {duration}s from magnitude for target {row['tar_id']}")
                
                request = Request(
                    id=row['tar_id'],
                    tar_ra=float(row['tar_ra']),
                    tar_dec=float(row['tar_dec']),
                    duration=duration,
                    name=row['tar_name'],
                    base_priority=int(row['tar_priority'] or 100),
                    comment=row['tar_comment'],
                    bonus=row['tar_bonus'],
                    bonus_time=row['tar_bonus_time'],
                    pm_ra=row['tar_pm_ra'],
                    pm_dec=row['tar_pm_dec'],
                    mag=row['tar_mag'],
                    last_observation_time=row['last_obs'],
                    sinfo=sinfo,
                    telescope_name=telescope_name
                )
                
                requests.append(request)
    
    logger.info(f"Loaded {len(requests)} requests from {telescope_name}")
    return requests

def load_all_requests(config, available_telescopes):
    """
    Load requests from all telescope databases.
    
    Returns:
        List of all Request objects from all telescopes
    """
    all_requests = []
    
    for telescope_name, telescope_config in config.get_resources().items():
        telescope_requests = load_requests_from_telescope(
            telescope_name, telescope_config, available_telescopes
        )
        all_requests.extend(telescope_requests)
    
    logger.info(f"Loaded {len(all_requests)} total requests from all telescopes")
    return all_requests

def group_requests_by_target(requests):
    """
    Group requests by target ID.
    
    Returns:
        Dict mapping target_id -> List of Request objects
    """
    target_groups = defaultdict(list)
    
    for request in requests:
        target_groups[request.id].append(request)
    
    return dict(target_groups)

def create_compound_reservations(requests, possible_windows_dict):
    """
    Create compound reservations from grouped requests.
    
    Args:
        requests: List of Request objects
        possible_windows_dict: Dict mapping telescope -> Intervals
    
    Returns:
        List of CompoundReservation objects
    """
    # Group requests by target
    target_groups = group_requests_by_target(requests)
    compound_reservations = []
    
    for target_id, target_requests in target_groups.items():
        # Check for AND type (dominant if any request has it)
        has_and_type = any(req.has_and_type() for req in target_requests)
        
        # Create reservations for each telescope
        reservations = []
        for request in target_requests:
            telescope = request.telescope_name
            
            if telescope not in possible_windows_dict:
                logger.warning(f"No visibility windows for {telescope}")
                continue
            
            telescope_windows = {telescope: possible_windows_dict[telescope]}
            
            reservation = Reservation(
                priority=request.base_priority,
                duration=request.duration,
                possible_windows_dict=telescope_windows,
                request=request
            )
            
            reservations.append(reservation)
        
        if not reservations:
            continue
        
        # Determine compound type
        if has_and_type:
            compound_type = 'and'
        elif len(reservations) > 1:
            compound_type = 'oneof'
        else:
            compound_type = 'single'
        
        compound_reservation = CompoundReservation(reservations, compound_type)
        compound_reservations.append(compound_reservation)
        
        logger.debug(f"Created {compound_type} compound reservation for target {target_id} "
                     f"with {len(reservations)} telescope options")
    
    logger.info(f"Created {len(compound_reservations)} compound reservations")
    return compound_reservations