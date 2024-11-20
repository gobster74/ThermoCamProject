import numpy as np
import cv2
import pyOptris as optris
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

recording = False
frame_buffer_1m = []  # Buffer for 1M camera frames
times_computer_1m = []  # Timestamps for 1M camera frames
running = True
toggled = False  # Whether reduced mode is active
click_x, click_y = None, None  

# Configuration files for the 1M camera
full_frame_xml = "17092037f.xml"  # Full frame configuration
reduced_frame_xml = "17092037.xml"  # Reduced frame configuration
crop_size = 100  # Crop area size


def initialize_camera(config_file):
    print(f"Initializing 1M camera with {config_file}...")
    result = optris.usb_init(config_file)
    if result != 0:
        print(f"Failed to initialize 1M camera with {config_file}.")
        return False
    print(f"1M camera initialized successfully with {config_file}.")
    return True

def close_camera():
    optris.terminate()
    print("Camera terminated successfully.")

def process_1m_camera():
    global running, recording, frame_buffer_1m, times_computer_1m, click_x, click_y, toggled

    w, h = optris.get_palette_image_size()  # Image size
    frame_count = 0
    start_time = time.time()

    while running:
        capture_start = time.time()
        frame = optris.get_palette_image(w, h)  # RGB image
        capture_end = time.time()

        timestamp = time.time()  # Capture timestamp
        frame_count += 1

        if recording:  # If recording is active, save frame and time
            frame_buffer_1m.append(frame)
            times_computer_1m.append(timestamp)
            status_label.config(text=f"Saving frame at {timestamp:.2f}s")
            label_img_1m.update_idletasks()  # Update the GUI

        # Update GUI with the current frame
        if toggled and click_x is not None and click_y is not None:
            # Calculate cropping area based on the click position
            start_x = click_x - crop_size // 2
            start_y = click_y - crop_size // 2
            end_x = click_x + crop_size // 2
            end_y = click_y + crop_size // 2

            # Ensure the cropping area is within bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(w, end_x)
            end_y = min(h, end_y)

            # Ensure the cropping area is valid
            if start_x < end_x and start_y < end_y:
                cropped_frame = frame[start_y:end_y, start_x:end_x]
                img = Image.fromarray(cropped_frame)
            else:
                img = Image.fromarray(frame)  # Fallback to full frame if cropping fails
        else:
            img = Image.fromarray(frame)

        imgtk = ImageTk.PhotoImage(image=img)
        label_img_1m.imgtk = imgtk
        label_img_1m.configure(image=imgtk)

        # Log capture timing
        capture_duration = capture_end - capture_start
        print(f"Frame captured in {capture_duration:.6f} seconds")
        
        # Periodically calculate the frame rate
        if frame_count % 50 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"Current frame rate: {fps:.2f} FPS")


def toggle_frame_mode(event):
    global toggled, running

    # Switch the frame mode (toggle between full and reduced)
    toggled = not toggled

    # Close the camera to reinitialize it with new configuration
    running = False
    close_camera()

    # Reinitialize camera with the respective configuration
    config_file = reduced_frame_xml if toggled else full_frame_xml
    if initialize_camera(config_file):
        # Restart processing with new camera settings
        running = True
        threading.Thread(target=process_1m_camera, daemon=True).start()

    print(f"Switched to {'reduced' if toggled else 'full'} frame mode.")

def start_camera():
    threading.Thread(target=process_1m_camera, daemon=True).start()

def start_recording():
    global recording
    recording = True
    status_label.config(text="Recording started...")

def stop_recording():
    global recording
    recording = False
    status_label.config(text="Recording stopped.")

def save_recording():
    if not frame_buffer_1m:
        status_label.config(text="No frames recorded.")
        return
    
    file_path = filedialog.asksaveasfilename(defaultextension=".bin", 
                                             filetypes=[("Binary files", "*.bin")])
    if file_path:
        with open(file_path, 'wb') as f:
            np.save(f, {'frame_buffer': frame_buffer_1m, 'times_computer': times_computer_1m})
        status_label.config(text="Recording saved successfully!")

def capture_click(event):
    global click_x, click_y 
    click_x, click_y = event.x, event.y
    print(f"Click coordinates: ({click_x}, {click_y})")

def create_gui():
    global label_img_1m, status_label

    window = tk.Tk()
    window.title("Thermal Camera Control")
    window.geometry("800x600")

    # Frame for camera display
    frame_display = tk.Frame(window)
    frame_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Label to display camera output for 1M camera
    label_img_1m = tk.Label(frame_display)
    label_img_1m.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Bind the right mouse button click to toggle the frame mode
    label_img_1m.bind("<Button-3>", toggle_frame_mode)
    # Bind left mouse button click to capture click coordinates
    label_img_1m.bind("<Button-1>", capture_click)

    # Buttons for control
    start_button = tk.Button(window, text="Start Recording", command=start_recording)
    start_button.pack(side=tk.LEFT, padx=5, pady=5)

    stop_button = tk.Button(window, text="Stop Recording", command=stop_recording)
    stop_button.pack(side=tk.LEFT, padx=5, pady=5)

    save_button = tk.Button(window, text="Save Recording", command=save_recording)
    save_button.pack(side=tk.LEFT, padx=5, pady=5)

    quit_button = tk.Button(window, text="Quit", command=window.quit)
    quit_button.pack(side=tk.LEFT, padx=5, pady=5)

    # Status label to show current status
    status_label = tk.Label(window, text="Ready")
    status_label.pack(side=tk.BOTTOM, padx=5, pady=5)

    window.protocol("WM_DELETE_WINDOW", lambda: close_camera() or window.quit())
    window.mainloop()

if __name__ == "__main__":
    if not initialize_camera(full_frame_xml):
        print("Camera initialization failed. Exiting...")
    else:
        start_camera()
        create_gui()  # Start the GUI
