import pyOptris as optris
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# USB connection initialization
try:
    optris.usb_init('6060300.xml')
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)

# Retrieve image sizes
w, h = optris.get_thermal_image_size()
pw, ph = optris.get_palette_image_size()
print('{} x {}'.format(w, h))

def convert_to_temperature_image(temperature_array):
    # Normalize the temperature values to the range [0, 255] for display
    normalized_temperature = ((temperature_array - np.min(temperature_array)) / 
                              (np.max(temperature_array) - np.min(temperature_array)) * 255).astype(np.uint8)

    # Ensure that the normalized values are within the range [0, 255]
    normalized_temperature = np.clip(normalized_temperature, 0, 255)

    # Create a colormap (you can customize this based on your preference)
    colormap = cv2.applyColorMap(normalized_temperature, cv2.COLORMAP_JET)

    return colormap

frames_captured = 5000
frame_buffer = np.empty((frames_captured, h, w), dtype=np.uint8)  # Adjust size as necessary

start_time = time.time()

times = []
frame_counter = 0
counter = 2000

try:
    for count in range(frames_captured):
        # Capture thermal image
        frame = optris.get_thermal_image(w, h)
        
        # Store frame in buffer
        frame_buffer[count] = frame

        if (count + 1) % counter == 0:
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            if len(times) > 1:
                mean_time_per_frame = (times[-1] - times[-2]) / counter
                mean_freq_per_frame = counter / (times[-1] - times[-2])
                print(f"Mean Time per frame is {mean_time_per_frame:.3f} s")
                print(f"Mean Freq per frame is {mean_freq_per_frame:.2f} Hz")

        frame_counter += 1

except Exception as e:
    print(f"Error during capture: {e}")

finally:
    end_time = time.time()
    recording_time = end_time - start_time
    print(f"Recording Time is {recording_time:.2f} s")
    print(f"Mean Time per frame is {recording_time / frames_captured:.3f} s")
    print(f"Mean Freq per frame is {frames_captured / recording_time:.2f} Hz")
    
    # Terminate connection
    optris.terminate()

    # Optionally save the buffer or visualize some frames
    # Example of saving a few frames
    for i in range(min(10, frames_captured)):
        plt.imshow(convert_to_temperature_image(frame_buffer[i]), cmap='jet')
        plt.title(f'Frame {i}')
        plt.colorbar()
        plt.show()



