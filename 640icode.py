import numpy as np
import cv2
import pyOptris as optris
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk
from threading import Lock
import os
from tkinter import filedialog

# Global variables
recording = False
frame_buffer_1m = []
frame_buffer_640i = []
times_computer_1m = []
times_computer_640i = []
running = True
frame_mode = 'full'
recording_lock = Lock()
log_dir = "logs"
frame_data_dir = "frame_data"
running_event = threading.Event()
roi_start_1m = None
roi_end_1m = None
selected_roi_1m = None
roi_start_640i = None
roi_end_640i = None
selected_roi_640i = None

# Video writers
video_writer_1m = None
video_writer_640i = None

# Camera frame configuration
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

# Initialize cameras
def initialize_cameras():
    print("Initializing cameras...")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    xml_files = [
        camera_frames['PI 1M'][frame_mode],
        camera_frames['PI 640i'][frame_mode]
    ]

    # PI 1M
    err, ID1 = optris.multi_usb_init(xml_files[0], None, os.path.join(log_dir, f'log_1m_{int(time.time())}.log'))
    if err != 0:
        print(f"Failed to initialize PI 1M: {err}")
        return err, None, None
    print(f"PI 1M ID: {ID1} Serial: {optris.get_multi_get_serial(ID1)}")

    # PI 640i
    err, ID2 = optris.multi_usb_init(xml_files[1], None, os.path.join(log_dir, f'log_640i_{int(time.time())}.log'))
    if err != 0:
        print(f"Failed to initialize PI 640i: {err}")
        return err, None, None
    print(f"PI 640i ID: {ID2} Serial: {optris.get_multi_get_serial(ID2)}")

    print("Cameras initialized successfully.")
    return 0, ID1, ID2

# Mouse event handlers for selecting ROI
def on_mouse_down_1m(event):
    global roi_start_1m, roi_end_1m
    roi_start_1m = (event.x, event.y)
    roi_end_1m = None

def on_mouse_drag_1m(event):
    global roi_start_1m, roi_end_1m
    roi_end_1m = (event.x, event.y)

def on_mouse_up_1m(event):
    global roi_start_1m, roi_end_1m, selected_roi_1m
    roi_end_1m = (event.x, event.y)
    if roi_start_1m and roi_end_1m:
        x1, y1 = roi_start_1m
        x2, y2 = roi_end_1m
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        selected_roi_1m = (x, y, w, h)
        print(f"Selected ROI for PI 1M: {selected_roi_1m}")

def on_mouse_down_640i(event):
    global roi_start_640i, roi_end_640i
    roi_start_640i = (event.x, event.y)
    roi_end_640i = None

def on_mouse_drag_640i(event):
    global roi_start_640i, roi_end_640i
    roi_end_640i = (event.x, event.y)

def on_mouse_up_640i(event):
    global roi_start_640i, roi_end_640i, selected_roi_640i
    roi_end_640i = (event.x, event.y)
    if roi_start_640i and roi_end_640i:
        x1, y1 = roi_start_640i
        x2, y2 = roi_end_640i
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        selected_roi_640i = (x, y, w, h)
        print(f"Selected ROI for PI 640i: {selected_roi_640i}")

# Toggle recording
def toggle_recording(camera_name):
    global recording, video_writer_1m, video_writer_640i
    with recording_lock:
        recording = not recording
    if recording:
        print(f"Recording started for {camera_name}")
        if camera_name == 'PI 1M' and video_writer_1m is None:
            video_writer_1m = cv2.VideoWriter(f'{frame_data_dir}/video_1m_{int(time.time())}.avi', 
                                              cv2.VideoWriter_fourcc(*'XVID'), 10, (764, 480))
        elif camera_name == 'PI 640i' and video_writer_640i is None:
            video_writer_640i = cv2.VideoWriter(f'{frame_data_dir}/video_640i_{int(time.time())}.avi', 
                                                cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))
    else:
        stop_recording(camera_name)

# Stop recording and save video
def stop_recording(camera_name):
    global video_writer_1m, video_writer_640i
    with recording_lock:
        if camera_name == 'PI 1M' and video_writer_1m is not None:
            video_writer_1m.release()
            print(f"Recording stopped and saved for PI 1M.")
            video_writer_1m = None
        elif camera_name == 'PI 640i' and video_writer_640i is not None:
            video_writer_640i.release()
            print(f"Recording stopped and saved for PI 640i.")
            video_writer_640i = None

# Process frames from PI 1M
def process_pi_1m(ID):
    global frame_buffer_1m, times_computer_1m, running, selected_roi_1m, video_writer_1m
    try:
        w, h = 764, 480
        while running:
            thermal_image = optris.get_multi_thermal_image(ID, w, h)[0]
            temperature_data = (thermal_image - 1000.0) / 10.0
            normalized_image = cv2.normalize(temperature_data, None, 0, 255, cv2.NMAX)
            color_image = cv2.applyColorMap(np.uint8(normalized_image), cv2.COLORMAP_JET)
            cv2.putText(color_image, "Camera: PI 1M (Full frame)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if roi_start_1m and roi_end_1m:
                cv2.rectangle(color_image, roi_start_1m, roi_end_1m, (255, 0, 0), 2)

            img = Image.fromarray(color_image)
            imgtk = ImageTk.PhotoImage(image=img)
            label_img_1m.imgtk = imgtk
            label_img_1m.configure(image=imgtk)

            # Write the frame to video file if recording
            if recording and video_writer_1m:
                video_writer_1m.write(color_image)

            time.sleep(0.1)
    except Exception as e:
        print(f"Error capturing frame from PI 1M: {e}")

# Process frames from PI 640i
def process_pi_640i(ID):
    global frame_buffer_640i, times_computer_640i, running, selected_roi_640i, video_writer_640i
    try:
        w, h = 640, 480
        while running:
            thermal_image = optris.get_multi_thermal_image(ID, w, h)[0]
            temperature_data = (thermal_image - 1000.0) / 10.0
            normalized_image = cv2.normalize(temperature_data, None, 0, 255, cv2.NORM_MINMAX)
            color_image = cv2.applyColorMap(np.uint8(normalized_image), cv2.COLORMAP_JET)
            cv2.putText(color_image, "Camera: PI 640i (Full frame)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if roi_start_640i and roi_end_640i:
                cv2.rectangle(color_image, roi_start_640i, roi_end_640i, (255, 0, 0), 2)

            img = Image.fromarray(color_image)
            imgtk = ImageTk.PhotoImage(image=img)
            label_img_640i.imgtk = imgtk
            label_img_640i.configure(image=imgtk)

            # Write the frame to video file if recording
            if recording and video_writer_640i:
                video_writer_640i.write(color_image)

            time.sleep(0.1)
    except Exception as e:
        print(f"Error capturing frame from PI 640i: {e}")

# GUI setup
root = tk.Tk()
root.title("Thermal Imaging Monitor")
root.geometry("1024x600")

# Camera Feed Frames
frame_1m = tk.Frame(root)
frame_1m.pack(side="left", padx=10)

frame_640i = tk.Frame(root)
frame_640i.pack(side="right", padx=10)

label_img_1m = tk.Label(frame_1m)
label_img_1m.pack()

label_img_640i = tk.Label(frame_640i)
label_img_640i.pack()

# Initialize cameras and start processing
err, ID1, ID2 = initialize_cameras()
if err == 0:
    threading.Thread(target=process_pi_1m, args=(ID1,)).start()
    threading.Thread(target=process_pi_640i, args=(ID2,)).start()

# Start/Stop Recording Buttons
btn_record_1m = tk.Button(root, text="Toggle PI 1M Recording", command=lambda: toggle_recording('PI 1M'))
btn_record_1m.pack(side="bottom", pady=5)

btn_record_640i = tk.Button(root, text="Toggle PI 640i Recording", command=lambda: toggle_recording('PI 640i'))
btn_record_640i.pack(side="bottom", pady=5)

root.mainloop()

