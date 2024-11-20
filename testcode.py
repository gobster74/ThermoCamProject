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

    xml_files = [
        camera_frames['PI 1M'][frame_mode],
        camera_frames['PI 640i'][frame_mode]
    ]

    # PI 1M
    err, ID1 = optris.multi_usb_init(xml_files[0], None, os.path.join(log_dir, f'log_1m_{int(time.time())}.log'))
    if err != 0:
        print(f"Failed to initialize PI 1M: {err}")
        return err, None, None  # Return here if there's an error
    print(f"PI 1M ID: {ID1} Serial: {optris.get_multi_get_serial(ID1)}")

    # PI 640i
    err, ID2 = optris.multi_usb_init(xml_files[1], None, os.path.join(log_dir, f'log_640i_{int(time.time())}.log'))
    if err != 0:
        print(f"Failed to initialize PI 640i: {err}")
        return err, None, None  # Return here if there's an error
    print(f"PI 640i ID: {ID2} Serial: {optris.get_multi_get_serial(ID2)}")

    print("Cameras initialized successfully.")
    return 0, ID1, ID2  

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

def close_camera():
    try:
        optris.terminate()
        print("Cameras terminated successfully")
    except Exception as e:
        print(f"Failed to terminate cameras: {e}")

def toggle_recording(camera_name):
    global recording
    with recording_lock:
        recording = not recording
    if recording:
        print(f"Recording started for {camera_name}")
    else:
        stop_recording(camera_name)

def stop_recording(ID):
    global frame_buffer_1m, frame_buffer_640i, times_computer_1m, times_computer_640i, ID1, ID2
    with recording_lock:
        if ID == 'PI 1M' and frame_buffer_1m:
            print(f"Saving recording for PI 1M with {len(frame_buffer_1m)} frames.")
            save_recording(ID, frame_buffer_1m, times_computer_1m)
            frame_buffer_1m = []
            times_computer_1m = []
        elif ID == 'PI 640i' and frame_buffer_640i:
            print(f"Saving recording for PI 640i with {len(frame_buffer_640i)} frames.")
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

    with open(f'{filename_prefix}.npy', 'wb') as f:
        np.save(f, np.array(frame_buffer))

    with open(f'{filename_prefix}_times.npy', 'wb') as f:
        np.save(f, np.array(timestamps))
    print(f'Recording stopped and files saved for {camera_name}: {filename_prefix}')

def switch_frame():
    global frame_mode
    frame_mode = 'reduced' if frame_mode == 'full' else 'full'
    print(f"Switch requested to {frame_mode} frame mode, reinitializing cameras")

def process_pi_1m(ID):
    global frame_buffer_1m, times_computer_1m, running, selected_roi_1m

    try:
        w, h = 764, 480 

        while running:
            thermal_image = optris.get_multi_thermal_image(ID, w, h)[0]
            temperature_data = (thermal_image - 1000.0) / 10.0
            
            normalized_image = cv2.normalize(temperature_data, None, 0, 255, cv2.NORM_MINMAX)
            color_image = cv2.applyColorMap(np.uint8(normalized_image), cv2.COLORMAP_JET)
            cv2.putText(color_image, "Camera: PI 1M (Full frame)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if roi_start_1m and roi_end_1m:
                cv2.rectangle(color_image, roi_start_1m, roi_end_1m, (255, 0, 0), 2)

            img = Image.fromarray(color_image)
            imgtk = ImageTk.PhotoImage(image=img)
            label_img_1m.imgtk = imgtk
            label_img_1m.configure(image=imgtk)

            with recording_lock:
                if recording and selected_roi_1m:
                    x, y, roi_w, roi_h = selected_roi_1m
                    roi_data = temperature_data[y:y+roi_h, x:x+roi_w]
                    frame_buffer_1m.append(roi_data)
                    times_computer_1m.append(time.time())


    except Exception as e:
        print(f"Error capturing frame from PI 1M: {e}")

def process_pi_640i(ID):
    global frame_buffer_640i, times_computer_640i, running, selected_roi_640i

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

            with recording_lock:
                if recording and selected_roi_640i:
                    x, y, roi_w, roi_h = selected_roi_640i
                    roi_data = temperature_data[y:y+roi_h, x:x+roi_w]
                    frame_buffer_640i.append(roi_data)
                    times_computer_640i.append(time.time())

    except Exception as e:
        print(f"Error capturing frame from PI 640i: {e}")

    except Exception as e:
        print(f"Error capturing frame from PI 640i: {e}")

def start_cameras(ID1,ID2):
    threading.Thread(target=process_pi_1m, daemon=True,args=(ID1,)).start()
    threading.Thread(target=process_pi_640i, daemon=True,args=(ID2,)).start()

def open_recording():
    file_path = filedialog.askopenfilename(title="Select Recording File", filetypes=(("Numpy files", "*.npy"), ("All files", "*.*")))
    if not file_path:
        return  
    try:
        frame_buffer = np.load(file_path)
        timestamps_file = file_path.replace(".npy", "_times.npy")  # Ensure matching extension
        timestamps = np.load(timestamps_file)
        display_loaded_frame(frame_buffer[0])  
        print(f"Loaded recording from {file_path}")
    except Exception as e:
        print(f"Error loading recording: {e}")

def display_loaded_frame(thermal_image):
    temperature_data = (thermal_image - 1000.0) / 10.0
    normalized_image = cv2.normalize(temperature_data, None, 0, 255, cv2.NORM_MINMAX)
    color_image = cv2.applyColorMap(np.uint8(normalized_image), cv2.COLORMAP_JET)
    
    img = Image.fromarray(color_image)
    imgtk = ImageTk.PhotoImage(image=img)
    
    label_img_1m.imgtk = imgtk
    label_img_1m.configure(image=imgtk)


def create_gui():
    global label_img_1m, label_img_640i

    window = tk.Tk()
    window.title("Thermal Camera Control")
    window.geometry("1500x600") 

    frame_display = tk.Frame(window)
    frame_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    label_img_1m = tk.Label(frame_display)
    label_img_1m.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    label_img_1m.bind("<Button-1>", on_mouse_down_1m)
    label_img_1m.bind("<B1-Motion>", on_mouse_drag_1m)
    label_img_1m.bind("<ButtonRelease-1>", on_mouse_up_1m)

    label_img_640i = tk.Label(frame_display)
    label_img_640i.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    label_img_640i.bind("<Button-1>", on_mouse_down_640i)
    label_img_640i.bind("<B1-Motion>", on_mouse_drag_640i)
    label_img_640i.bind("<ButtonRelease-1>", on_mouse_up_640i)

    # Controls at the bottom
    frame_controls = tk.Frame(window)
    frame_controls.pack(side=tk.BOTTOM, fill=tk.X)

    record_button_1m = tk.Button(frame_controls, text="Start/Stop Recording (PI 1M)",
                                  command=lambda: toggle_recording('PI 1M'))
    record_button_1m.pack(side=tk.LEFT, padx=5, pady=5)

    record_button_640i = tk.Button(frame_controls, text="Start/Stop Recording (PI 640i)",
                                    command=lambda: toggle_recording('PI 640i'))
    record_button_640i.pack(side=tk.LEFT, padx=5, pady=5)

    open_recording_button = tk.Button(frame_controls, text="Open Recording", command=open_recording)
    open_recording_button.pack(side=tk.LEFT, padx=5, pady=5)

    switch_frame_button = tk.Button(frame_controls, text="Switch Full/Reduced Frame", command=switch_frame)
    switch_frame_button.pack(side=tk.LEFT, padx=5, pady=5)

    quit_button = tk.Button(frame_controls, text="Quit", command=lambda: on_closing(window))
    quit_button.pack(side=tk.LEFT, padx=5, pady=5)

    window.protocol("WM_DELETE_WINDOW", lambda: on_closing(window))
    window.mainloop()

def on_closing(window):
    global running
    running = False 
    close_camera()
    window.quit()
    window.destroy()

if __name__ == "__main__":
    err, ID1, ID2 = initialize_cameras()
    if err != 0:
        print("Camera initialization failed. Exiting...")
    else:
        start_cameras(ID1, ID2)
        create_gui()