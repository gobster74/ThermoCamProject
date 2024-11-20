import numpy as np
import matplotlib.pyplot as plt

# File paths for the frame buffer and timestamps
frame_file_path = r'frame_data\frame_buffer_PI 640i_1731578605.npy'
timestamp_file_path = r'frame_data\frame_buffer_PI 640i_1731578605_times.npy'

frame_buffer = np.load(frame_file_path)
timestamps = np.load(timestamp_file_path)

print(f"Frame buffer shape: {frame_buffer.shape}")
print(f"Timestamps shape: {timestamps.shape}")

selected_frame = frame_buffer[9]  

mean_temp = np.mean(selected_frame)
max_temp = np.max(selected_frame)
min_temp = np.min(selected_frame)
std_dev_temp = np.std(selected_frame)

#define an ROI 
roi = selected_frame[100:200, 150:250] 
mean_roi = np.mean(roi)
max_roi = np.max(roi)
min_roi = np.min(roi)
std_dev_roi = np.std(roi)

# results
print(f"Mean Temperature (whole image): {mean_temp:.2f}")
print(f"Max Temperature (whole image): {max_temp:.2f}")
print(f"Min Temperature (whole image): {min_temp:.2f}")
print(f"Standard Deviation (whole image): {std_dev_temp:.2f}")

print(f"\nMetrics for ROI (100:200, 150:250):")
print(f"Mean Temperature (ROI): {mean_roi:.2f}")
print(f"Max Temperature (ROI): {max_roi:.2f}")
print(f"Min Temperature (ROI): {min_roi:.2f}")
print(f"Standard Deviation (ROI): {std_dev_roi:.2f}")

plt.imshow(selected_frame, cmap='hot')
plt.title(f'Thermal Image at Timestamp: {timestamps[0]}')
plt.colorbar()
plt.show()

plt.imshow(roi, cmap='hot')
plt.title('Region of Interest (ROI)')
plt.colorbar()
plt.show()