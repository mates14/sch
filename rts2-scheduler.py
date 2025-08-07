#!/usr/bin/python3
"""
RTS2 Scheduler - Main Orchestrator
Simplified main script that coordinates the scheduling process.
"""

import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path

from config_loader import Config, ConfigurationError
from telescope_ops import check_telescope_state, upload_schedule_to_rts2
from scheduler_core import run_scheduling_algorithm
from database import get_connection
from visualization import create_schedule_plot, create_enhanced_plots
from recorder import ScheduleRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main scheduler orchestrator."""
    parser = argparse.ArgumentParser(description='RTS2 telescope scheduler')
    parser.add_argument('--config', '-c', default='rts2-scheduler.cfg',
                       help='Path to configuration file')
    parser.add_argument('--skip-state-check', '-f', action='store_true',
                       help='Skip telescope state check (for testing)')
    parser.add_argument('--no-upload', action='store_true',
                       help='Skip uploading schedule to RTS2')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip creating visualization plots')
    args = parser.parse_args()

    try:
        # Load configuration
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Initialize schedule recorder
        schedule_dir = Path(config.get_output_path('schedule_dir', './schedules'))
        recorder = ScheduleRecorder(base_path=schedule_dir)

        # Check telescope availability
        if not args.skip_state_check:
            available_telescopes = check_telescope_state(config)
            if not any(available_telescopes.values()):
                logger.warning("No telescopes available for scheduling")
                return 0
        else:
            logger.warning("Skipping telescope state check")
            available_telescopes = {name: True for name in config.get_resources()}

        # Run the scheduling algorithm
        result = run_scheduling_algorithm(config, available_telescopes, recorder)
        if not result or not result.get('schedule') or not any(result['schedule'].values()):
            logger.warning("No observations scheduled")
            return 0

        schedule = result['schedule']
        horizon_functions = result['horizon_functions']
        compound_reservations = result['compound_reservations']

        # Save the schedule
        recorder.save_schedule(schedule, calculated_at=datetime.utcnow())
        logger.info("Schedule saved")

        # Upload to RTS2
        if not args.no_upload:
            upload_schedule_to_rts2(schedule, config)
            logger.info("Schedule uploaded to RTS2")

        # Create visualization
        if not args.no_plot:
            plot_dir = Path(config.get_output_path('plot_dir', './plots'))
            plot_dir.mkdir(exist_ok=True)

            plot_file = plot_dir / f"schedule-{datetime.utcnow():%Y%m%d-%H%M%S}.png"

            # Get first available telescope's DB for visualization
            first_telescope = next(iter(available_telescopes))
            db_config = config.get_resource_db_config(first_telescope)

            with get_connection(db_config) as conn:
                create_schedule_plot(schedule, config.get_resources(),
                                   plot_file, conn)

                create_enhanced_plots(schedule, config.get_resources(), horizon_functions,
                                compound_reservations, plot_dir, conn)
                logger.info(f"Visualization saved to {plot_file}")

        logger.info("Scheduler completed successfully")
        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

