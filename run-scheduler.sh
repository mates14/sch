#!/bin/sh
# Run the scheduler as cron used to: from the directory above this one
# (plots/ and schedules/ are relative paths), all output appended to
# rts2-scheduler.log there. rts2-sensor-scheduler runs this; extra
# arguments go to the scheduler (--force, --no-upload, ...).
#
# The lock serialises runs: while both cron and the daemon start the
# scheduler, the second waits for the first instead of planning alongside.
# A daemon started by root with --run-as keeps root's HOME, and python
# then misses the user's site-packages (~/.local, where ortools is).
HOME=$(getent passwd "$(id -u)" | cut -d: -f6)
export HOME

SCH=$(dirname "$(readlink -f "$0")")
cd "$SCH/.." || exit 1
exec flock -w 600 /tmp/rts2-scheduler.lock \
	"$SCH/rts2-scheduler.py" -c "$SCH/rts2-scheduler.cfg" "$@" >> rts2-scheduler.log 2>&1
