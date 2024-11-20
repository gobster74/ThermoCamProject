import cv2

import pyOptris as optris


# USB connection initialisation 
optris.usb_init("6060300f.xml")

# optris.set_palette(9)

w, h = optris.get_palette_image_size()
print("{} x {}".format(w, h))

while True:
    # Get the palette image (RGB image)
    frame = optris.get_palette_image(w, h)
    cv2.imshow("IR streaming", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

optris.terminate()
cv2.destroyAllWindows()
