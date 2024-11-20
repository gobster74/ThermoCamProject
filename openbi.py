import numpy as np

frame_buffer_file = 'frame_buffer_c_ulong(2)_1731548661.bin'
times_file = 'frame_buffer_c_ulong(2)_1731548661_times.bin'

try:
    with open(frame_buffer_file, 'rb') as f:
        frame_buffer_640i = np.load(f)

    with open(times_file, 'rb') as f:
        times_computer_640i = np.load(f)

    first_frame = frame_buffer_640i[0]
    first_timestamp = times_computer_640i[0]

    print(f"First frame: {first_frame.shape}")
    print(f"First timestamp: {first_timestamp}")

except Exception as e:
    print(f"An error occurred {e}")