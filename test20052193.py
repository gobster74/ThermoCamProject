import pyOptris as optris

def initialize_camera(serial_file):
    try:
        # Initialize the camera
        optris.usb_init(serial_file)
        print(f"Camera with serial {serial_file} initialized successfully.")

        # Attempt to get the image size
        width, height = optris.get_thermal_image_size()
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions returned.")
        print(f"Image dimensions: {width}x{height}")

        # Capture a sample image
        try:
            image, _ = optris.get_thermal_image(width, height)
            if image is not None:
                print("Sample image captured successfully.")
            else:
                print("Failed to capture a sample image.")
        except Exception as e:
            print(f"Error capturing image: {e}")

    except Exception as e:
        print(f"Failed to initialize camera with serial {serial_file}: {e}")

# Path to the XML configuration file for the camera
serial_file = "6060300.xml"

# Initialize the camera
initialize_camera(serial_file)
