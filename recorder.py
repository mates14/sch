from astropy.table import Table, Column, vstack
from datetime import datetime, timedelta
from pathlib import Path
from os import system

class ScheduleRecorder:
    def __init__(self, base_path="./schedules"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def get_schedule_filename(self, date=None):
        if date is None:
            date = datetime.utcnow()
        return self.base_path / f"schedule-{date.strftime('%Y%m%d')}.ecsv"

    def get_specific_schedule_filename(self, filename):
        """Get a specific schedule file path"""
        if isinstance(filename, Path):
            return filename
        return self.base_path / filename

    def _convert_times_to_string(self, table):
        """Convert datetime columns to ISO format strings"""
        if 'start_time' in table.colnames:
            table['start_time'] = [t.isoformat() for t in table['start_time']]
        if 'calculated_at' in table.colnames:
            table['calculated_at'] = [t.isoformat() for t in table['calculated_at']]
        return table

    def _convert_strings_to_datetime(self, table):
        """Convert ISO format strings back to datetime objects"""
        if 'start_time' in table.colnames:
            table['start_time'] = [datetime.fromisoformat(t) for t in table['start_time']]
        if 'calculated_at' in table.colnames:
            table['calculated_at'] = [datetime.fromisoformat(t) for t in table['calculated_at']]
        return table

    def save_schedule(self, schedule, calculated_at):
        rows = []
        for telescope, observations in schedule.items():
            for obs in observations:
                rows.append({
                    'telescope': telescope,
                    'target_id': obs.request.id,
                    'start_time': obs.scheduled_start,
                    'duration': obs.scheduled_quantum,
                    'ra': obs.request.tar_ra,
                    'dec': obs.request.tar_dec,
                    'calculated_at': calculated_at,
                    'priority': obs.request.base_priority,
                    'status': 'PLANNED'  # PLANNED, EXECUTING, COMPLETED, FAILED
                })

        if not rows:
            return

        # Create table
        t = Table(rows=rows)
        t.meta['CALCULATED'] = calculated_at.isoformat()

        # Save to file
        filename = self.get_schedule_filename()
        if filename.exists():
            # Load existing and merge
            existing = self.load_current_schedule()
            if existing is not None:
                # Keep past observations and merge with new future ones
                current_time = datetime.utcnow()
                past_mask = existing['start_time'] < current_time
                future_mask = ~past_mask

                merged = vstack([
                    existing[past_mask],
                    t
                ])
        else:
            merged = t

        # Convert datetime objects to strings before saving
        merged_str = self._convert_times_to_string(merged.copy())
        merged_str.write(filename, format='ascii.ecsv', overwrite=True)

        # make a backup for debugging purposes
        date = datetime.utcnow()
        sfn = self.base_path / f"schedule-{date.strftime('%Y%m%d-%H%M%S')}.ecsv"
        system(f"cp {filename} {sfn}")

    def load_current_schedule(self):
        """Load the current schedule file"""
        filename = self.get_schedule_filename()
        return self.load_schedule(filename)

    def load_schedule(self, filename):
        """Load a specific schedule file

        Args:
            filename: Either a string filename or Path object pointing to the schedule file

        Returns:
            Table: The loaded schedule table, or None if the file doesn't exist
        """
        filepath = self.get_specific_schedule_filename(filename)
        if not filepath.exists():
            return None

        # Load and convert string times back to datetime objects
        table = Table.read(filepath, format='ascii.ecsv')
        return self._convert_strings_to_datetime(table)

    def get_next_available_time(self):
        """Get the end time of the currently executing observation"""
        schedule = self.load_current_schedule()
        if schedule is None:
            return datetime.utcnow()

        current_time = datetime.utcnow()

        # Convert duration to numeric values
        durations = [timedelta(seconds=d.item()) for d in schedule['duration'].value.astype(int)]

        mask = ((schedule['start_time'].value <= current_time) &
            [(t + d > current_time) for t, d in zip(schedule['start_time'].value, durations)])

        if not any(mask):
            return current_time

        current_obs = schedule[mask][0]
        return current_obs['start_time'] + timedelta(seconds=float(current_obs['duration']))
