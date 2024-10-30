import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from datetime import datetime, timedelta
import numpy as np

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
        AND i.delete_flag = 'f'
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

def visualize_schedule(schedule, resources, output_file='schedule_plot.png', conn=None):
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
