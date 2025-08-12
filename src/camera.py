# Camera "firmware" for drone

import cv2 as cv
import numpy as np

# Read image from Pictures folder
image = cv.imread('/home/simon/Documents/python-projects/drone-course/Pictures/Drone/capture_5/img_0.jpg')


# Convert to HSV
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)


# Create green mask
green_LB = np.array([20, 40, 40])
green_UB = np.array([80, 255, 255])


# Apply mask
green_mask = cv.inRange(hsv_image, green_LB, green_UB)


# Invert mask
non_green = cv.bitwise_not(green_mask)


# Apply morphologyEx to reduce noise
kernel = np.ones((15, 15), np.uint8)
non_green = cv.morphologyEx(non_green, cv.MORPH_OPEN, kernel)


# Find contours and filter out large objects
contours, heir = cv.findContours(non_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
animal_mask = np.zeros_like(non_green)
animals_count = 0
for cnt in contours:
    if cv.contourArea(cnt) > 6500.0:
        print("Contour size: ", cv.contourArea(cnt))
        cv.drawContours(animal_mask, [cnt], -1, 255, thickness=cv.FILLED, lineType=cv.LINE_8)


# Cut out animals
animals = cv.bitwise_and(image, image, mask=animal_mask)

# Make images ready for viewing
mask      = cv.cvtColor(animal_mask, cv.COLOR_GRAY2BGR)

scale     = 0.20
image     = cv.resize(image, (0, 0), fx=scale, fy=scale)
hsv_image = cv.resize(hsv_image, (0, 0), fx=scale, fy=scale)
cutout    = cv.resize(animals, (0, 0), fx=scale, fy=scale)
mask      = cv.resize(mask, (0, 0), fx=scale, fy=scale)


row1 = np.hstack((image, hsv_image))
row2 = np.hstack((mask, cutout))
view = np.vstack((row1, row2))


cv.imshow('Animals:', view)
cv.waitKey(0)
cv.destroyAllWindows()
