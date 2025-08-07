#!/usr/bin/python3

from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import DictCursor
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from recorder import ScheduleRecorder

def add_time_delta_array(datetimes, seconds):
    """Helper function to add time deltas to arrays of datetimes"""
    return [t + timedelta(seconds=float(s)) for t, s in zip(datetimes, seconds)]

def fetch_actual_observations(conn, start_time, end_time):
    """
    Fetch actual observation data from RTS2 database.
    Returns a dict mapping target_id to list of observation points.
    """
    query = """
        SELECT
            o.tar_id,
            i.img_date + make_interval(secs := i.img_exposure/2.0) as mid_time,
            i.img_alt,
            i.img_az
        FROM images i
        JOIN observations o ON i.obs_id = o.obs_id
        WHERE i.img_date BETWEEN %s AND %s
        ORDER BY i.img_date
    """

    actual_observations = {}

    try:
        with conn.cursor() as cur:
            cur.execute(query, (start_time, end_time))
            for row in cur:
                tar_id = row[0]
                if tar_id not in actual_observations:
                    actual_observations[tar_id] = {
                        'times': [],
                        'altitudes': [],
                        'azimuths': []
                    }
                actual_observations[tar_id]['times'].append(row[1])
                actual_observations[tar_id]['altitudes'].append(row[2])
                actual_observations[tar_id]['azimuths'].append(row[3])

    except Exception as e:
        logger.error(f"Error fetching actual observations: {e}")
        raise

    return actual_observations

def calculate_elevation(ra, dec, location, time):
    target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    alt_az = target.transform_to(AltAz(obstime=time, location=location))
    return alt_az.alt.deg

def plot_schedule(schedule, targets, resources, output_file='schedule_plot.png', conn=None):
    fig, axs = plt.subplots(len(resources), 1, figsize=(15, 5*len(resources)), sharex=True)
    if len(resources) == 1:
        axs = [axs]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(targets)))
    color_map = {target.id: color for target, color in zip(targets, colors)}

    # Get time range for fetching actual observations
    all_times = []
    for observations in schedule.values():
        for obs in observations:
            all_times.extend([
                obs.scheduled_start,
                obs.scheduled_start + timedelta(seconds=obs.scheduled_quantum)
            ])
    start_time = min(all_times)
    end_time = max(all_times)

    # Fetch actual observations if database connection provided
    actual_observations = None
    if conn is not None:
        actual_observations = fetch_actual_observations(conn, start_time, end_time)

    for i, (telescope, observations) in enumerate(schedule.items()):
        ax = axs[i]
        ax.set_title(f'Schedule for {telescope}')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_ylim(0, 90)

        location = EarthLocation(
            lat=resources[telescope]['location']['latitude']*u.deg,
            lon=resources[telescope]['location']['longitude']*u.deg,
            height=resources[telescope]['location']['elevation']*u.m
        )

        # Plot planned observations
        for observation in observations:
            target = observation.request
            start_time = observation.scheduled_start
            duration = observation.scheduled_quantum
            end_time = start_time + timedelta(seconds=duration)

            times = [start_time + timedelta(minutes=m) for m in range(0, int(duration/60) + 1)]
            elevations = [calculate_elevation(target.tar_ra, target.tar_dec, location, Time(t))
                         for t in times]

            line = ax.plot(times, elevations, color=color_map[target.id],
                          linewidth=2, linestyle='-', label=f'Planned {target.id}')

            # Add target ID label
            mid_index = len(times) // 2
            mid_time = times[mid_index]
            mid_elevation = elevations[mid_index]
            ax.annotate(f'{target.id}',
                        (mid_time, mid_elevation),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2',
                                fc='white',
                                ec='none',
                                alpha=0.7))

            # Plot actual observations if available
            if actual_observations and target.id in actual_observations:
                act_obs = actual_observations[target.id]
                ax.plot(act_obs['times'], act_obs['altitudes'],
                       color=color_map[target.id],
                       linestyle=':',
                       marker='.',
                       markersize=4,
                       label=f'Actual {target.id}')

    plt.xlabel('Time')
    plt.gcf().autofmt_xdate()

    # Add legend
    if actual_observations:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_schedule_plot(schedule, resources, output_file='schedule_plot.png', conn=None):
    """
    Visualize schedule with both planned and actual observations.

    Args:
        schedule: Dictionary mapping telescope names to lists of observations
        resources: Dictionary of telescope resources and their properties
        output_file: Path to save the plot
        conn: Database connection for fetching actual observations
    """
    # Get target list directly from the schedule
    target_list = []
    for observations in schedule.values():
        for obs in observations:
            if obs.request not in target_list:  # Avoid duplicates
                target_list.append(obs.request)

    plot_schedule(schedule, target_list, resources, output_file, conn)
    print(f"Schedule plot saved to {output_file}")

def plot_schedule_polar_altaz(schedule, resources, horizon_functions, compound_reservations=None,
                              output_file='schedule_polar.png', conn=None):
    """
    Create a polar alt-az plot showing the schedule.

    Args:
        schedule: Dictionary mapping telescope names to observations
        resources: Dictionary of telescope resources
        horizon_functions: Dictionary of horizon interpolation functions
        compound_reservations: List of compound reservations with visibility windows (optional)
        output_file: Output file path
        conn: Database connection for actual observations (optional)
    """
    fig, axes = plt.subplots(1, len(resources), figsize=(6*len(resources), 6),
                            subplot_kw=dict(projection='polar'))
    if len(resources) == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, 20))  # More colors for targets

    for ax_idx, (telescope, observations) in enumerate(schedule.items()):
        if not observations:
            continue

        ax = axes[ax_idx]
        resource_info = resources[telescope]
        horizon_func = horizon_functions[telescope]

        # Set up the polar plot
        ax.set_title(f'{telescope}', pad=10, fontsize=10)
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)  # Clockwise (astronomical convention)
        ax.set_ylim(0, 90)  # 0 = zenith, 90 = horizon
        ax.set_ylabel('Zenith Distance', labelpad=15, fontsize=8)

        # Create zenith distance labels (inverted altitude)
        ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=7)
        ax.tick_params(axis='x', labelsize=7)

        # Plot horizon
        az_horizon = np.linspace(0, 360, 360)
        alt_horizon = [horizon_func(az) for az in az_horizon]
        zenith_dist_horizon = [90 - alt for alt in alt_horizon]

        # Convert to radians for polar plot
        az_horizon_rad = np.radians(az_horizon)
        ax.plot(az_horizon_rad, zenith_dist_horizon, 'k-', linewidth=1.5, alpha=0.8)
        ax.fill_between(az_horizon_rad, zenith_dist_horizon, 90,
                       color='lightgray', alpha=0.3)

        # Get Earth location
        location = resource_info['earth_location']

        # Get time range for the plot - back to original 30 minutes around observations
        all_times = []
        for obs in observations:
            all_times.extend([
                obs.scheduled_start,
                obs.scheduled_start + timedelta(seconds=obs.scheduled_quantum)
            ])

        if not all_times:
            continue

        plot_start = min(all_times) - timedelta(minutes=30)
        plot_end = max(all_times) + timedelta(minutes=30)

        # Plot target availability and scheduled observations
        color_idx = 0
        for obs in observations:
            target = obs.request
            color = colors[color_idx % len(colors)]
            color_idx += 1

            # Option 1: Use pre-calculated visibility windows if available
            if compound_reservations is not None:
                # Find the corresponding reservation
                target_reservation = None
                for cr in compound_reservations:
                    for res in cr.reservation_list:
                        if res.request and res.request.id == target.id:
                            target_reservation = res
                            break
                    if target_reservation:
                        break

                if target_reservation and telescope in target_reservation.possible_windows_dict:
                    # Use scheduler's pre-calculated availability windows
                    windows = target_reservation.possible_windows_dict[telescope]
                    window_tuples = windows.toTupleList()

                    # Plot availability windows (thin lines)
                    for window_start, window_end in window_tuples:
                        # Create time points within this window
                        window_duration = (window_end - window_start).total_seconds()
                        num_points = max(10, int(window_duration / 300))  # Point every 5 minutes
                        window_times = [window_start + timedelta(seconds=i*window_duration/(num_points-1))
                                      for i in range(num_points)]

                        # Calculate positions for this window
                        astro_times_window = Time(window_times)
                        altaz_frame_window = AltAz(obstime=astro_times_window, location=location)
                        target_coord = SkyCoord(ra=target.tar_ra*u.degree,
                                              dec=target.tar_dec*u.degree,
                                              frame='icrs')
                        target_altaz_window = target_coord.transform_to(altaz_frame_window)

                        # Convert to astronomical azimuth
                        az_astro_window = (target_altaz_window.az.degree + 180) % 360
                        alt_window = target_altaz_window.alt.degree
                        zenith_dist_window = 90 - alt_window
                        az_window_rad = np.radians(az_astro_window)

                        # Plot availability window (thin line)
                        ax.plot(az_window_rad, zenith_dist_window,
                               color=color, alpha=0.4, linewidth=0.8)

                    # Plot scheduled observation (thick line)
                    obs_start = obs.scheduled_start
                    obs_end = obs.scheduled_start + timedelta(seconds=obs.scheduled_quantum)

                    # Create time points for scheduled period
                    obs_duration = obs.scheduled_quantum
                    num_obs_points = max(5, int(obs_duration / 300))
                    obs_times = [obs_start + timedelta(seconds=i*obs_duration/(num_obs_points-1))
                               for i in range(num_obs_points)]

                    astro_times_obs = Time(obs_times)
                    altaz_frame_obs = AltAz(obstime=astro_times_obs, location=location)
                    target_altaz_obs = target_coord.transform_to(altaz_frame_obs)

                    az_astro_obs = (target_altaz_obs.az.degree + 180) % 360
                    alt_obs = target_altaz_obs.alt.degree
                    zenith_dist_obs = 90 - alt_obs
                    az_obs_rad = np.radians(az_astro_obs)

                    # Plot scheduled observation (thick line)
                    ax.plot(az_obs_rad, zenith_dist_obs, color=color,
                           linewidth=3, alpha=0.9)

                    # Add target ID at midpoint (no frame)
                    if len(az_obs_rad) > 0:
                        mid_idx = len(az_obs_rad) // 2
                        ax.annotate(f'{target.id}',
                                  (az_obs_rad[mid_idx], zenith_dist_obs[mid_idx]),
                                  xytext=(3, 3), textcoords='offset points',
                                  fontsize=6, ha='left', va='bottom',
                                  color=color, weight='bold')

                    continue  # Skip fallback method below

            # Option 2: Fallback - calculate visibility if pre-calculated data not available
            # Calculate target track over the plot period
            time_step = timedelta(minutes=5)
            plot_times = []
            current_time = plot_start
            while current_time <= plot_end:
                plot_times.append(current_time)
                current_time += time_step

            astro_times = Time(plot_times)
            altaz_frame = AltAz(obstime=astro_times, location=location)
            target_coord = SkyCoord(ra=target.tar_ra*u.degree,
                                  dec=target.tar_dec*u.degree,
                                  frame='icrs')
            target_altaz = target_coord.transform_to(altaz_frame)

            # Convert to astronomical azimuth (0° = South)
            az_astro = (target_altaz.az.degree + 180) % 360
            alt_target = target_altaz.alt.degree
            zenith_dist = 90 - alt_target

            # Show availability: entire period where target is above horizon
            above_horizon = alt_target > horizon_func(az_astro)

            if np.any(above_horizon):
                az_available = az_astro[above_horizon]
                zenith_dist_available = zenith_dist[above_horizon]
                times_available = np.array(plot_times)[above_horizon]

                # Convert azimuth to radians
                az_available_rad = np.radians(az_available)

                # Plot entire availability period (thin line)
                ax.plot(az_available_rad, zenith_dist_available,
                       color=color, alpha=0.4, linewidth=0.8)

                # Calculate and plot scheduled observation period (thick line)
                obs_start = obs.scheduled_start
                obs_end = obs.scheduled_start + timedelta(seconds=obs.scheduled_quantum)
                obs_mask = (times_available >= obs_start) & (times_available <= obs_end)

                if np.any(obs_mask):
                    az_obs = az_available_rad[obs_mask]
                    zenith_obs = zenith_dist_available[obs_mask]

                    # Plot scheduled observation (thick line)
                    ax.plot(az_obs, zenith_obs, color=color,
                           linewidth=3, alpha=0.9)

                    # Add target ID at midpoint of observation (no frame)
                    if len(az_obs) > 0:
                        mid_idx = len(az_obs) // 2
                        ax.annotate(f'{target.id}',
                                  (az_obs[mid_idx], zenith_obs[mid_idx]),
                                  xytext=(3, 3), textcoords='offset points',
                                  fontsize=6, ha='left', va='bottom',
                                  color=color, weight='bold')

        # Fetch and plot actual observations if database connection provided
        if conn is not None:
            try:
                actual_obs = fetch_actual_observations(conn, plot_start, plot_end)

                for target_id, obs_data in actual_obs.items():
                    if obs_data['times']:
                        # Convert observed alt/az to zenith distance
                        az_obs_astro = np.array(obs_data['azimuths'])  # Assuming already in astronomical coords
                        alt_obs = np.array(obs_data['altitudes'])
                        zenith_dist_obs = 90 - alt_obs

                        az_obs_rad = np.radians(az_obs_astro)

                        # Plot actual observations as smaller scatter points
                        ax.scatter(az_obs_rad, zenith_dist_obs,
                                 c='red', marker='x', s=15, alpha=0.8)

            except Exception as e:
                print(f"Could not plot actual observations: {e}")

        # Add grid
        ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_schedule_with_polar(schedule, resources, horizon_functions, compound_reservations=None,
                                  output_dir='./plots', conn=None):
    """
    Create both time-based and polar alt-az visualizations of the schedule.

    Args:
        schedule: Dictionary mapping telescope names to observations
        resources: Dictionary of telescope resources
        horizon_functions: Dictionary of horizon functions for each telescope
        compound_reservations: List of compound reservations with visibility windows (optional)
        output_dir: Directory to save plots
        conn: Database connection for actual observations
    """
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')

    # Create time-based plot (existing functionality)
    #time_plot_file = os.path.join(output_dir, f"schedule-time-{timestamp}.png")
    #visualize_schedule(schedule, resources, time_plot_file, conn)

    # Create polar alt-az plot (new functionality)
    polar_plot_file = os.path.join(output_dir, f"schedule-polar-{timestamp}.png")
    plot_schedule_polar_altaz(schedule, resources, horizon_functions, compound_reservations, polar_plot_file, conn)

#    print(f"Time-based plot saved to {time_plot_file}")
    print(f"Polar alt-az plot saved to {polar_plot_file}")


# Simple function to call from rts2-scheduler.py
def create_enhanced_plots(schedule, resources, horizon_functions, compound_reservations, plot_dir, conn=None):
    """
    Create enhanced schedule visualizations.
    Simple wrapper for easy integration.
    """
    visualize_schedule_with_polar(schedule, resources, horizon_functions, compound_reservations, plot_dir, conn)

def plot_schedule_vs_actual(date, resources, conn, recorder, schedule_file=None, output_file='schedule_comparison.png'):
    """
    Plot schedule comparison using ScheduleRecorder to load either current or specific schedule.
    Points are aligned to 5-minute intervals and centered within these intervals.
    """
    # Load schedule using recorder
    if schedule_file is not None:
        schedule = recorder.load_schedule(schedule_file)
    else:
        schedule = recorder.load_current_schedule()

    if schedule is None:
        print(f"No schedule found for {'specified file' if schedule_file else 'date ' + str(date)}")
        return

    # Get time range from schedule
    start_time = min(schedule['start_time'])
    end_times = add_time_delta_array(schedule['start_time'], schedule['duration'])
    end_time = max(end_times)

    # Fetch actual observations
    actual_obs = fetch_actual_observations(conn, start_time, end_time)

    # Create plot
    fig, axs = plt.subplots(len(resources), 1, figsize=(15, 5*len(resources)), sharex=True)
    if len(resources) == 1:
        axs = [axs]

    # Create color map for targets
    unique_targets = np.unique(schedule['target_id'])
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_targets)))
    color_map = {tid: color for tid, color in zip(unique_targets, colors)}

    def round_to_5min_interval(dt):
        """Round datetime to nearest 5-minute interval"""
        minutes = dt.minute
        rounded_minutes = (minutes // 5) * 5
        return dt.replace(minute=rounded_minutes, second=0, microsecond=0)

    def get_interval_center(dt):
        """Get the center point of the 5-minute interval"""
        interval_start = round_to_5min_interval(dt)
        return interval_start + timedelta(minutes=2.5)

    for i, (telescope, resource_info) in enumerate(resources.items()):
        ax = axs[i]
        ax.set_title(f'{telescope}', pad=10, fontsize=10)
        ax.set_ylabel('Elevation (degrees)', fontsize=9)
        ax.set_ylim(0, 90)

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=8)

        location = EarthLocation(
            lat=resource_info['location']['latitude']*u.deg,
            lon=resource_info['location']['longitude']*u.deg,
            height=resource_info['location']['elevation']*u.m
        )

        # Plot planned observations
        telescope_schedule = schedule[schedule['telescope'] == telescope]
        for row in telescope_schedule:
            times = [row['start_time'] + timedelta(minutes=m)
                    for m in range(0, int(row['duration']/60) + 1)]
            elevations = [calculate_elevation(row['ra'], row['dec'], location, Time(t))
                         for t in times]

            ax.fill_between(times, elevations, alpha=0.2, color=color_map[row['target_id']])
            ax.plot(times, elevations, color=color_map[row['target_id']],
                   linestyle='-', linewidth=1, alpha=0.8)

            # Add small target label
            mid_idx = len(times) // 2
            ax.annotate(f'{row["target_id"]}',
                       (times[mid_idx], elevations[mid_idx]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', linewidth=0.5))

        # Plot actual observations with aligned 5-minute bins
        for target_id, obs in actual_obs.items():
            if target_id in color_map:
                # Create bins aligned to 5-minute intervals
                bins = {}
                for t, alt in zip(obs['times'], obs['altitudes']):
                    bin_start = round_to_5min_interval(t)
                    if bin_start not in bins:
                        bins[bin_start] = []
                    bins[bin_start].append(alt)

                # Calculate bin centers and statistics
                bin_centers = [get_interval_center(start) for start in bins.keys()]
                bin_means = [np.mean(alts) for alts in bins.values()]
                bin_stds = [np.std(alts) if len(alts) > 1 else 0 for alts in bins.values()]

                # Plot actual observations as diamonds at bin centers
                ax.scatter(bin_centers, bin_means,
                          color=color_map[target_id],
                          marker='D',
                          s=20,
                          alpha=0.8)

                # Add error bars
                ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                           color=color_map[target_id],
                           fmt='none',
                           alpha=0.3,
                           linewidth=0.5)

    # Format x-axis
    plt.xlabel('Time (UTC)', fontsize=9)
    date_formatter = plt.matplotlib.dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.gcf().autofmt_xdate()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    # Database connection
    DB_CONFIG = {
        "dbname": "stars",
        "user": "mates",
        "password": "pasewcic25",
        "host": "localhost"
    }

    conn = psycopg2.connect(**DB_CONFIG)

    # Resources definition
    resources = {
        'telescope1': {
            'name': 'D50',
            'location': {
                'latitude': 49.9093889,
                'longitude': 14.7813631,
                'elevation': 530
            }
        }
    }

    # Initialize recorder
    recorder = ScheduleRecorder()

    # Plot today's schedule
    today = datetime.utcnow().date()
    plot_schedule_vs_actual(today, resources, conn, recorder, schedule_file="/home/mates/schedules/schedule-20241028.ecsv")
    print("Comparison plot saved as schedule_comparison.png")

if __name__ == "__main__":
    main()
