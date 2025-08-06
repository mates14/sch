"""
Updated scheduler_core.py to integrate sinfo system (minimal changes)
"""

import logging
from datetime import datetime, timedelta
from collections import defaultdict

from config_loader import Config
from database_interface import load_all_requests, create_compound_reservations
from kernel.solver import CPScheduler
from kernel.intervals import Intervals
from visibility import calculate_visibility_windows
from recorder import ScheduleRecorder

logger = logging.getLogger(__name__)

def run_scheduling_algorithm(config, available_telescopes, recorder):
    """
    Main scheduling algorithm with sinfo support.
    
    Args:
        config: Configuration object
        available_telescopes: Dict mapping telescope_name -> is_available
        recorder: ScheduleRecorder instance
    
    Returns:
        Dict mapping telescope_name -> list of scheduled observations
    """
    logger.info("Starting scheduling algorithm")
    
    # Calculate visibility windows for all telescopes
    possible_windows_dict = {}
    for telescope_name, telescope_config in config.get_resources().items():
        if not available_telescopes.get(telescope_name, False):
            continue
        
        # Calculate visibility for this telescope
        windows = calculate_visibility_windows(
            telescope_config['location'],
            telescope_config['horizon_file'],
            config.get_scheduler_param('min_altitude', 20.0)
        )
        
        if not windows.is_empty():
            possible_windows_dict[telescope_name] = windows
    
    if not possible_windows_dict:
        logger.warning("No visibility windows available for any telescope")
        return {}
    
    # Load requests from all databases with sinfo support
    # This replaces the existing database loading logic
    requests = load_all_requests(config, available_telescopes)
    
    if not requests:
        logger.warning("No enabled requests found in any database")
        return {}
    
    # Create compound reservations (replaces existing logic)
    compound_reservations = create_compound_reservations(requests, possible_windows_dict)
    
    if not compound_reservations:
        logger.warning("No valid compound reservations created")
        return {}
    
    # Run the CP scheduler (unchanged)
    scheduler = CPScheduler(
        compound_reservation_list=compound_reservations,
        globally_possible_windows_dict=possible_windows_dict,
        slice_size_seconds=config.get_scheduler_param('slice_size', 300),
        timelimit=config.get_scheduler_param('timelimit', 180),
        mip_gap=config.get_scheduler_param('mip_gap', 0.05)
    )
    
    schedule = scheduler.schedule_all()
    
    # Log scheduling statistics
    total_scheduled = sum(len(obs_list) for obs_list in schedule.values())
    total_requests = len(requests)
    
    logger.info(f"Scheduled {total_scheduled} observations from {total_requests} requests")
    
    for telescope, observations in schedule.items():
        if observations:
            total_time = sum(obs.scheduled_quantum for obs in observations)
            logger.info(f"{telescope}: {len(observations)} observations, "
                       f"{total_time}s total ({total_time/3600:.1f}h)")
    
    return schedule

# Example sinfo usage patterns:
SINFO_EXAMPLES = """
Examples of sinfo in database:

Basic duration override:
INSERT INTO scheduling VALUES (1108, 'duration=600');

Weekly monitoring cadence:
INSERT INTO scheduling VALUES (2045, 'duration=180,pscale=7');

Force simultaneous observation:
INSERT INTO scheduling VALUES (3021, 'duration=300,type=and');

Multiple parameters:
INSERT INTO scheduling VALUES (4012, 'duration=450,pscale=14,type=and');
"""