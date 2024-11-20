import numpy as np
import cv2
import pyOptris as optris
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk
from threading import Lock
import os

# Global variables
recording = False
frame_buffer_1m = []
frame_buffer_640i = []
times_computer_1m = []
times_computer_640i = []
running = True
frame_mode = 'reduced'  # for initialization
recording_lock = Lock()
log_dir = "logs"
frame_data_dir = "frame_data"
running_event = threading.Event()


# Camera frame configuration, information is already inside of the xml file
camera_frames = {
    'PI 1M': {
        'full': '17092037f.xml',
        'reduced': '17092037.xml'
    },
    'PI 640i': {
        'full': '6060300f.xml',
        'reduced': '6060300.xml'
    }
}

def initialize_cameras():
    print("Initializing cameras...")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Load XML file 
    xml_files = [
        camera_frames['PI 1M'][frame_mode],
        camera_frames['PI 640i'][frame_mode]
    ]

    # PI 1M
    err,ID1 = optris.multi_usb_init(xml_files[0],None, os.path.join(log_dir, f'log_1m_{int(time.time())}.log'))
    if err != 0:
        print(f"Failed to initialize PI 1M: {err}")
        return False, None, None, None, None
    print(f"PI 1M ID: {ID1} Serial: {optris.get_multi_get_serial(ID1)}")

    # PI 640i
    err,ID2 = optris.multi_usb_init(xml_files[1],None, os.path.join(log_dir, f'log_640i_{int(time.time())}.log'))
    if err != 0:
        print(f"Failed to initialize PI 640i: {err}")
        return False, None, None, None, None
    print(f"PI 640i ID: {ID2} Serial: {optris.get_multi_get_serial(ID2)}")

    print("Cameras initialized successfully.")

    #dimesions for the GUI (test) 
    width_1m, height_1m = 764, 480
    width_640i, height_640i = 640, 480
 
    return True,ID1,ID2, width_1m + width_640i, max(height_1m, height_640i)

def reinitialize_cameras():
    global ID1, ID2, total_width, total_height

    close_camera()

    success, ID1, ID2, total_width, total_height = initialize_cameras()
    if success:
        print(f"Cameras switched to {frame_mode} frame mode")
        start_cameras(ID1, ID2)
    else: print(f"Processes failed")

def close_camera():
    try:
        optris.terminate()
        print("Cameras terminated successfully")
    except Exception as e:
        print(f"Failed to terminate cameras: {e}")

def toggle_recording(ID):
    global recording
    with recording_lock:
        recording = not recording
    if recording:
        print(f"Recording started on {ID}")
    else:
        stop_recording(ID)

def stop_recording(ID):
    global frame_buffer_1m, frame_buffer_640i, times_computer_1m, times_computer_640i
    with recording_lock:
        if ID == ID1 and frame_buffer_1m:
            save_recording(ID, frame_buffer_1m, times_computer_1m)
            frame_buffer_1m=[]
            times_computer_1m = []
        elif ID == ID2 and frame_buffer_640i:
            save_recording(ID, frame_buffer_640i, times_computer_640i)
            frame_buffer_640i = []
            times_computer_640i = []
        else:
            print(f"No data to save for {ID}")

if not os.path.exists(frame_data_dir):
    os.makedirs(frame_data_dir)

def save_recording(camera_name, frame_buffer, timestamps):       
    timestamp = int(time.time())
    filename_prefix = f'{frame_data_dir}/frame_buffer_{camera_name}_{timestamp}'
    with open(f'{filename_prefix}.bin', 'wb') as f:
        np.save(f, np.array(frame_buffer))
    with open(f'{filename_prefix}_times.bin', 'wb') as f:
        np.save(f, np.array(timestamps))
    print(f'Recording stopped and files saved for {camera_name}: {filename_prefix}')

def switch_frame():
    global frame_mode
    frame_mode = 'reduced' if frame_mode == 'full' else 'full'
    print(f"Switch requested to {frame_mode} frame mode, reinitializing cameras")
    reinitialize_cameras()

def process_pi_1m_fps(ID):
    w, h = 764, 480  
    frame_count = 0
    start_time = time.time()

    try:
        while frame_count < 100:  
            
            thermal_image = optris.get_multi_thermal_image(ID, w, h)[0]
            frame_count += 1

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"PI 1M FPS: {fps:.2f}")
    except Exception as e:
        print(f"Error capturing frame from PI 1M: {e}")

def process_pi_640i_fps(ID):
    w, h = 640, 480
    frame_count = 0
    start_time = time.time()

    target_fps = 125
    frame_time = 1 / target_fps 

    try:
        while frame_count < 100:
            thermal_image = optris.get_multi_thermal_image(ID, w, h)[0]
            frame_count += 1
            elapsed_time = time.time() - start_time
            time_to_next_frame = frame_count * frame_time - elapsed_time
            if time_to_next_frame > 0:
                time.sleep(time_to_next_frame)
                
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"PI 640i FPS: {fps:.2f}")

    except Exception as e:
        print(f"Error capturing frame from PI 640i: {e}")


def start_cameras(ID1,ID2):
    threading.Thread(target=process_pi_1m_fps, daemon=True,args=(ID1,)).start()
    threading.Thread(target=process_pi_640i_fps, daemon=True,args=(ID2,)).start()

def create_gui(total_width, total_height):
    global label_img_1m, label_img_640i

    window = tk.Tk()
    window.title("Thermal Camera Control")
    window.geometry(f"{total_width}x{total_height + 100}")  

    # Frame for camera display
    frame_display = tk.Frame(window)
    frame_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Label to display camera output for PI 1M
    label_img_1m = tk.Label(frame_display, width = total_width // 2, height = total_height)
    label_img_1m.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Label to display camera output for PI 640i
    label_img_640i = tk.Label(frame_display, width = total_width // 2, height = total_height)
    label_img_640i.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Controls at the bottom
    frame_controls = tk.Frame(window)
    frame_controls.pack(side=tk.BOTTOM, fill=tk.X)

    record_button_1m = tk.Button(frame_controls, text="Start/Stop Recording (PI 1M)",
                                  command=lambda: toggle_recording(ID1))
    record_button_1m.pack(side=tk.LEFT, padx=5, pady=5)

    record_button_640i = tk.Button(frame_controls, text="Start/Stop Recording (PI 640i)",
                                    command=lambda: toggle_recording(ID2))
    record_button_640i.pack(side=tk.LEFT, padx=5, pady=5)

    switch_frame_button = tk.Button(frame_controls, text="Switch Full/Reduced Frame", command=switch_frame)
    switch_frame_button.pack(side=tk.LEFT, padx=5, pady=5)

    quit_button = tk.Button(frame_controls, text="Quit", command=lambda: on_closing(window))
    quit_button.pack(side=tk.LEFT, padx=5, pady=5)

    window.protocol("WM_DELETE_WINDOW", lambda: on_closing(window))
    window.mainloop()

def on_closing(window):
    global running_event
    running_event.clear()
    close_camera()
    window.quit()
    window.destroy()

if __name__ == "__main__":
    # Initialize cameras
    success, ID1, ID2, _, _ = initialize_cameras()
    if not success:
        print("Cameras failed to initialize...")
    else:
        # Test FPS for PI 1M
        process_pi_1m_fps(ID1)
        process_pi_640i_fps(ID1)
        close_camera()
