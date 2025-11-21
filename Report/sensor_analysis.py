import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("MILESTONE 2: SENSOR PROCESSING ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: UNDERSTANDING SENSOR DATA ERRORS
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: UNDERSTANDING SENSOR DATA ERRORS")
print("=" * 80)

# Load acceleration data
accel_data = pd.read_csv('/mnt/project/ACCELERATION.csv')
print(f"\nLoaded {len(accel_data)} acceleration data points")

# Extract data
timestamp = accel_data['timestamp'].values
acceleration = accel_data['acceleration'].values
noisy_acceleration = accel_data['noisyacceleration'].values

# Calculate velocities using trapezoidal integration
dt = timestamp[1] - timestamp[0]
velocity_clean = np.cumsum(acceleration) * dt
velocity_noisy = np.cumsum(noisy_acceleration) * dt

# Calculate distances using trapezoidal integration
distance_clean = np.cumsum(velocity_clean) * dt
distance_noisy = np.cumsum(velocity_noisy) * dt

# Report final distances
print(f"\n--- Results ---")
print(f"Final distance (clean acceleration): {distance_clean[-1]:.4f} meters")
print(f"Final distance (noisy acceleration): {distance_noisy[-1]:.4f} meters")
print(f"Difference: {abs(distance_clean[-1] - distance_noisy[-1]):.4f} meters")
print(f"Percent error: {abs(distance_clean[-1] - distance_noisy[-1]) / distance_clean[-1] * 100:.2f}%")

# Create plots for Part 1
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Acceleration comparison
axes[0].plot(timestamp, acceleration, 'b-', label='Clean Acceleration', linewidth=2)
axes[0].plot(timestamp, noisy_acceleration, 'r-', label='Noisy Acceleration', alpha=0.7)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Acceleration (m/s²)')
axes[0].set_title('Acceleration vs Noisy Acceleration')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Velocity comparison
axes[1].plot(timestamp, velocity_clean, 'b-', label='Velocity (Clean)', linewidth=2)
axes[1].plot(timestamp, velocity_noisy, 'r-', label='Velocity (Noisy)', alpha=0.7)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_title('Velocity Calculated from Clean vs Noisy Acceleration')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Distance comparison
axes[2].plot(timestamp, distance_clean, 'b-', label='Distance (Clean)', linewidth=2)
axes[2].plot(timestamp, distance_noisy, 'r-', label='Distance (Noisy)', alpha=0.7)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Distance (m)')
axes[2].set_title('Distance Traveled from Clean vs Noisy Acceleration')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/part1_sensor_errors.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved Part 1 plots to part1_sensor_errors.png")
plt.close()

# ============================================================================
# PART 2: STEP DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: STEP DETECTION")
print("=" * 80)

# Load walking data
walking_data = pd.read_csv('/mnt/project/WALKING.csv')
print(f"\nLoaded {len(walking_data)} walking data points")

# Convert timestamp to seconds (from nanoseconds)
walking_data['time_s'] = (walking_data['timestamp'] - walking_data['timestamp'].iloc[0]) / 1e9

# Calculate magnitude of acceleration
walking_data['accel_mag'] = np.sqrt(
    walking_data['accel_x']**2 + 
    walking_data['accel_y']**2 + 
    walking_data['accel_z']**2
)

print(f"Time range: {walking_data['time_s'].min():.2f}s to {walking_data['time_s'].max():.2f}s")

# Smoothing function using moving average
def smooth_signal(signal, window_size=10):
    """Apply moving average smoothing to signal"""
    return uniform_filter1d(signal, size=window_size, mode='nearest')

# Apply smoothing to acceleration magnitude
window_size = 15  # Optimized for step detection
walking_data['accel_mag_smooth'] = smooth_signal(walking_data['accel_mag'], window_size)

# Plot raw vs smoothed data
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(walking_data['time_s'], walking_data['accel_mag'], 
        'gray', alpha=0.4, label='Raw Acceleration Magnitude', linewidth=0.8)
ax.plot(walking_data['time_s'], walking_data['accel_mag_smooth'], 
        'b-', label=f'Smoothed (window={window_size})', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration Magnitude (m/s²)')
ax.set_title('Walking Data: Raw vs Smoothed Acceleration Magnitude')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/part2_data_preparation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved Part 2 data preparation plot")
plt.close()

# Step Detection Algorithm
def detect_steps(accel_mag, time_array, threshold_factor=0.15, min_step_time=0.3):
    """
    Detect steps using peak detection on acceleration magnitude.
    
    Parameters:
    - accel_mag: smoothed acceleration magnitude array
    - time_array: corresponding time array
    - threshold_factor: factor above mean to consider a peak (default 0.15)
    - min_step_time: minimum time between steps in seconds (default 0.3s)
    
    Returns:
    - step_indices: indices of detected steps
    - step_times: times of detected steps
    """
    # Calculate threshold: mean + factor * (max - mean)
    mean_accel = np.mean(accel_mag)
    max_accel = np.max(accel_mag)
    threshold = mean_accel + threshold_factor * (max_accel - mean_accel)
    
    # Find peaks above threshold
    step_indices = []
    step_times = []
    last_step_idx = -1
    min_samples = int(min_step_time / np.mean(np.diff(time_array)))
    
    for i in range(1, len(accel_mag) - 1):
        # Check if current point is a local maximum
        if (accel_mag[i] > accel_mag[i-1] and 
            accel_mag[i] > accel_mag[i+1] and 
            accel_mag[i] > threshold):
            
            # Check minimum time between steps
            if last_step_idx == -1 or (i - last_step_idx) >= min_samples:
                step_indices.append(i)
                step_times.append(time_array[i])
                last_step_idx = i
    
    return np.array(step_indices), np.array(step_times)

# Detect steps
step_indices, step_times = detect_steps(
    walking_data['accel_mag_smooth'].values,
    walking_data['time_s'].values,
    threshold_factor=0.15,
    min_step_time=0.3
)

num_steps = len(step_indices)
print(f"\n--- Step Detection Results ---")
print(f"Number of steps detected: {num_steps}")
print(f"Average time between steps: {np.mean(np.diff(step_times)):.3f}s")
print(f"Step frequency: {num_steps / walking_data['time_s'].max():.2f} steps/second")

# Plot step detection results
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(walking_data['time_s'], walking_data['accel_mag_smooth'], 
        'b-', label='Smoothed Acceleration', linewidth=1.5)
ax.plot(step_times, walking_data['accel_mag_smooth'].iloc[step_indices], 
        'ro', markersize=8, label=f'Detected Steps (n={num_steps})')
mean_accel = np.mean(walking_data['accel_mag_smooth'])
max_accel = np.max(walking_data['accel_mag_smooth'])
threshold = mean_accel + 0.15 * (max_accel - mean_accel)
ax.axhline(y=threshold, color='g', linestyle='--', 
           label=f'Detection Threshold', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration Magnitude (m/s²)')
ax.set_title(f'Step Detection Results: {num_steps} Steps Detected')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/part2_step_detection.png', dpi=300, bbox_inches='tight')
print("✓ Saved Part 2 step detection plot")
plt.close()

# ============================================================================
# PART 3: DIRECTION DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: DIRECTION DETECTION")
print("=" * 80)

# Load turning data (handle trailing commas)
turning_data = pd.read_csv('/home/claude/TURNING_clean.csv')
# Fill NaN values with 0 for magnetometer data
turning_data = turning_data.fillna(0)
print(f"\nLoaded {len(turning_data)} turning data points")

# Convert timestamp to seconds
turning_data['time_s'] = (turning_data['timestamp'] - turning_data['timestamp'].iloc[0]) / 1e9

# Use gyroscope Z-axis for rotation detection (yaw)
turning_data['gyro_z_smooth'] = smooth_signal(turning_data['gyro_z'], window_size=20)

print(f"Time range: {turning_data['time_s'].min():.2f}s to {turning_data['time_s'].max():.2f}s")

# Plot raw vs smoothed gyroscope data
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(turning_data['time_s'], turning_data['gyro_z'], 
        'gray', alpha=0.4, label='Raw Gyro Z', linewidth=0.8)
ax.plot(turning_data['time_s'], turning_data['gyro_z_smooth'], 
        'r-', label='Smoothed Gyro Z (window=20)', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angular Velocity (rad/s)')
ax.set_title('Turning Data: Raw vs Smoothed Gyroscope Z-axis')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/part3_data_preparation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved Part 3 data preparation plot")
plt.close()

# Direction Detection Algorithm
def detect_turns(gyro_z, time_array, threshold=0.3, min_turn_time=0.5):
    """
    Detect turns using gyroscope z-axis data.
    
    Parameters:
    - gyro_z: smoothed gyroscope z-axis data (angular velocity in rad/s)
    - time_array: corresponding time array
    - threshold: minimum angular velocity to consider a turn (rad/s)
    - min_turn_time: minimum duration for a turn (seconds)
    
    Returns:
    - turn_segments: list of (start_idx, end_idx, angle, direction) tuples
    """
    dt = np.mean(np.diff(time_array))
    min_samples = int(min_turn_time / dt)
    
    turn_segments = []
    in_turn = False
    turn_start = 0
    
    for i in range(len(gyro_z)):
        if not in_turn and abs(gyro_z[i]) > threshold:
            # Start of a turn
            in_turn = True
            turn_start = i
        elif in_turn and abs(gyro_z[i]) <= threshold:
            # End of a turn
            if i - turn_start >= min_samples:
                # Calculate total angle turned (integrate angular velocity)
                turn_gyro = gyro_z[turn_start:i]
                angle = np.sum(turn_gyro) * dt * (180 / np.pi)  # Convert to degrees
                direction = "Right" if angle < 0 else "Left"
                turn_segments.append((turn_start, i, angle, direction))
            in_turn = False
    
    # Handle case where turn continues to end of data
    if in_turn and len(gyro_z) - turn_start >= min_samples:
        turn_gyro = gyro_z[turn_start:]
        angle = np.sum(turn_gyro) * dt * (180 / np.pi)
        direction = "Right" if angle < 0 else "Left"
        turn_segments.append((turn_start, len(gyro_z), angle, direction))
    
    return turn_segments

# Detect turns
turns = detect_turns(
    turning_data['gyro_z_smooth'].values,
    turning_data['time_s'].values,
    threshold=0.3,
    min_turn_time=0.5
)

print(f"\n--- Direction Detection Results ---")
print(f"Number of turns detected: {len(turns)}")
for i, (start_idx, end_idx, angle, direction) in enumerate(turns):
    start_time = turning_data['time_s'].iloc[start_idx]
    end_time = turning_data['time_s'].iloc[end_idx]
    duration = end_time - start_time
    print(f"Turn {i+1}: {direction:5s} | Angle: {angle:7.2f}° | "
          f"Duration: {duration:.2f}s | Time: {start_time:.2f}s - {end_time:.2f}s")

# Calculate total rotation
total_angle = sum([angle for _, _, angle, _ in turns])
print(f"\nTotal rotation: {total_angle:.2f}°")

# Plot turn detection results
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(turning_data['time_s'], turning_data['gyro_z_smooth'], 
        'b-', label='Smoothed Gyro Z', linewidth=1.5)
ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Threshold (+)')
ax.axhline(y=-0.3, color='g', linestyle='--', alpha=0.5, label='Threshold (-)')

# Highlight turn regions
colors = {'Left': 'orange', 'Right': 'purple'}
for i, (start_idx, end_idx, angle, direction) in enumerate(turns):
    time_seg = turning_data['time_s'].iloc[start_idx:end_idx]
    gyro_seg = turning_data['gyro_z_smooth'].iloc[start_idx:end_idx]
    label = f'{direction} Turn' if i == 0 or direction != turns[i-1][3] else None
    ax.fill_between(time_seg, gyro_seg, alpha=0.3, color=colors[direction], label=label)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Angular Velocity (rad/s)')
ax.set_title(f'Turn Detection Results: {len(turns)} Turns Detected')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/part3_turn_detection.png', dpi=300, bbox_inches='tight')
print("✓ Saved Part 3 turn detection plot")
plt.close()

# ============================================================================
# PART 4: TRAJECTORY PLOTTING
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: TRAJECTORY PLOTTING")
print("=" * 80)

# Load combined walking and turning data
trajectory_data = pd.read_csv('/home/claude/WALKING_AND_TURNING_clean.csv')
print(f"\nLoaded {len(trajectory_data)} trajectory data points")

# Convert timestamp to seconds
trajectory_data['time_s'] = (trajectory_data['timestamp'] - trajectory_data['timestamp'].iloc[0]) / 1e9

# Calculate acceleration magnitude
trajectory_data['accel_mag'] = np.sqrt(
    trajectory_data['accel_x']**2 + 
    trajectory_data['accel_y']**2 + 
    trajectory_data['accel_z']**2
)

# Smooth the data
trajectory_data['accel_mag_smooth'] = smooth_signal(trajectory_data['accel_mag'], window_size=15)
trajectory_data['gyro_z_smooth'] = smooth_signal(trajectory_data['gyro_z'], window_size=20)

# Detect steps in trajectory
traj_step_indices, traj_step_times = detect_steps(
    trajectory_data['accel_mag_smooth'].values,
    trajectory_data['time_s'].values,
    threshold_factor=0.15,
    min_step_time=0.3
)

print(f"\nDetected {len(traj_step_indices)} steps in trajectory data")

# Detect turns in trajectory
traj_turns = detect_turns(
    trajectory_data['gyro_z_smooth'].values,
    trajectory_data['time_s'].values,
    threshold=0.3,
    min_turn_time=0.5
)

print(f"Detected {len(traj_turns)} turns in trajectory data")

# Build trajectory
# Assume each step is approximately 0.7 meters (average human step length)
step_length = 0.7  # meters

# Initialize position and heading
x, y = 0, 0
heading = 90  # Start facing North (90 degrees in standard orientation)
positions = [(x, y)]
headings = [heading]

# Track cumulative angle change
cumulative_angle = 0

# Process each time step
for i in range(len(traj_step_times)):
    step_time = traj_step_times[i]
    
    # Find all turns that occurred before this step
    if i == 0:
        prev_time = 0
    else:
        prev_time = traj_step_times[i-1]
    
    # Calculate rotation from gyroscope between previous step and current step
    time_mask = (trajectory_data['time_s'] >= prev_time) & (trajectory_data['time_s'] <= step_time)
    gyro_segment = trajectory_data.loc[time_mask, 'gyro_z_smooth'].values
    
    if len(gyro_segment) > 0:
        dt = np.mean(np.diff(trajectory_data.loc[time_mask, 'time_s'].values))
        if not np.isnan(dt):
            angle_change = np.sum(gyro_segment) * dt * (180 / np.pi)
            cumulative_angle += angle_change
            heading += angle_change
    
    # Move forward one step in the current heading direction
    x += step_length * np.cos(np.radians(heading))
    y += step_length * np.sin(np.radians(heading))
    
    positions.append((x, y))
    headings.append(heading)

# Extract x and y coordinates
x_coords = [pos[0] for pos in positions]
y_coords = [pos[1] for pos in positions]

print(f"\n--- Trajectory Results ---")
print(f"Total steps taken: {len(traj_step_times)}")
print(f"Total distance traveled: {len(traj_step_times) * step_length:.2f} meters")
print(f"Total rotation: {cumulative_angle:.2f}°")
print(f"Final position: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f}) meters")
print(f"Final heading: {headings[-1]:.2f}° from North")

# Plot trajectory
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the path
ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6, label='Path')
ax.plot(x_coords, y_coords, 'bo', markersize=4, alpha=0.4)

# Mark start and end points
ax.plot(x_coords[0], y_coords[0], 'go', markersize=15, label='Start', zorder=5)
ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=15, label='End', zorder=5)

# Add arrows to show direction every few steps
arrow_interval = max(1, len(x_coords) // 15)
for i in range(0, len(x_coords)-1, arrow_interval):
    dx = x_coords[i+1] - x_coords[i]
    dy = y_coords[i+1] - y_coords[i]
    if dx != 0 or dy != 0:
        ax.arrow(x_coords[i], y_coords[i], dx, dy, 
                head_width=0.3, head_length=0.2, fc='blue', ec='blue', 
                alpha=0.5, zorder=3)

# Annotate key turning points
for i, (start_idx, end_idx, angle, direction) in enumerate(traj_turns):
    turn_time = trajectory_data['time_s'].iloc[start_idx]
    # Find closest step to this turn
    step_idx = np.argmin(np.abs(traj_step_times - turn_time))
    if step_idx < len(x_coords):
        ax.annotate(f'{direction}\n{abs(angle):.0f}°', 
                   xy=(x_coords[step_idx], y_coords[step_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('X Position (meters)', fontsize=12)
ax.set_ylabel('Y Position (meters)', fontsize=12)
ax.set_title('Walking Trajectory\n(Starting facing North)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/part4_trajectory.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved Part 4 trajectory plot")
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. part1_sensor_errors.png - Acceleration, velocity, and distance comparisons")
print("  2. part2_data_preparation.png - Raw vs smoothed walking data")
print("  3. part2_step_detection.png - Step detection results")
print("  4. part3_data_preparation.png - Raw vs smoothed turning data")
print("  5. part3_turn_detection.png - Turn detection results")
print("  6. part4_trajectory.png - Complete walking trajectory")
print("\n" + "=" * 80)
