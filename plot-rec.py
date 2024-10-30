#!/usr/bin/python3

from astropy.table import Table
import matplotlib.pyplot as plt
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
    """Fetch actual observation data from RTS2 database"""
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
    
    return actual_observations

def calculate_elevation(ra, dec, location, time):
    target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    alt_az = target.transform_to(AltAz(obstime=time, location=location))
    return alt_az.alt.deg

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

def ex_plot_schedule_vs_actual(date, resources, conn, recorder, schedule_file=None, output_file='schedule_comparison.png'):
    """
    Plot schedule comparison using ScheduleRecorder to load either current or specific schedule.
    
    Args:
        date: Date to plot schedule for
        resources: Dictionary of telescope resources
        conn: Database connection
        recorder: ScheduleRecorder instance
        schedule_file: Optional specific schedule file to load (default: None, uses current)
        output_file: Output file path
    """
    # Load schedule using recorder
    if schedule_file is not None:
        schedule = recorder.load_schedule(schedule_file)
    else:
        schedule = recorder.load_current_schedule()
        
    if schedule is None:
        print(f"No schedule found for {'specified file' if schedule_file else 'date ' + str(date)}")
        return
    
    # Rest of the function remains the same...
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
    
    # Create color map for targets with distinct colors
    unique_targets = np.unique(schedule['target_id'])
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_targets)))
    color_map = {tid: color for tid, color in zip(unique_targets, colors)}
    
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
            
            # Plot planned observation as semi-transparent filled area
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
        
        # Plot actual observations
        for target_id, obs in actual_obs.items():
            if target_id in color_map:
                start_minutes = (min(obs['times']) - start_time).total_seconds() / 60
                all_minutes = [(t - start_time).total_seconds() / 60 for t in obs['times']]
                
                # Bin the observations (every 5 minutes)
                bin_size = 5
                bins = {}
                for minute, alt in zip(all_minutes, obs['altitudes']):
                    bin_idx = int(minute / bin_size)
                    if bin_idx not in bins:
                        bins[bin_idx] = []
                    bins[bin_idx].append(alt)
                
                binned_times = [start_time + timedelta(minutes=idx * bin_size) for idx in bins.keys()]
                binned_alts = [np.mean(alts) for alts in bins.values()]
                
                # Plot actual observations as diamonds
                ax.scatter(binned_times, binned_alts,
                          color=color_map[target_id],
                          marker='D',
                          s=20,
                          alpha=0.8)
                
                # Add error bars
                yerr = [np.std(alts) if len(alts) > 1 else 0 for alts in bins.values()]
                ax.errorbar(binned_times, binned_alts, yerr=yerr,
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
