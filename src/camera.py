# Camera "firmware" for drone

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_path = "/home/simon/Documents/python-projects/drone-course/images/capture_13/img_0.jpg"

# Read image from Pictures folder
image = cv.imread(image_path)
dis_image = cv.imread(image_path)

#image = cv.GaussianBlur(image, (5, 5), 0)

# Convert to HSV
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)


# Create green mask
green_LB = np.array([20, 40, 0])
green_UB = np.array([80, 255, 255])


# Apply mask
green_mask = cv.inRange(hsv_image, green_LB, green_UB)


# Invert mask
non_green = cv.bitwise_not(green_mask)


# Apply morphologyEx to reduce noise
kernel = np.ones((15, 15), np.uint8)
non_green = cv.morphologyEx(non_green, cv.MORPH_OPEN, kernel)

stored = []

# Find contours and filter out large objects
contours, heir = cv.findContours(non_green, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
animal_mask = np.zeros_like(non_green)
animals_count = 0
index = 0
for cnt in contours:
    if cv.contourArea(cnt) > 6500.0:
        index += 1
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # If contour area is zero, just use a default point
            cx, cy = cnt[0][0]

        stored.append([cv.contourArea(cnt), cv.arcLength(cnt, True), index])
        cv.putText(dis_image, str(index), (cx, cy), 1, 16, (255, 0, 0), 20)
        cv.drawContours(animal_mask, [cnt], -1, 255, thickness=cv.FILLED, lineType=cv.LINE_8)

stored_vals = np.array(stored)

# Cut out animals
animals = cv.bitwise_and(image, image, mask=animal_mask)

# Make images ready for viewing
mask      = cv.cvtColor(animal_mask, cv.COLOR_GRAY2BGR)


areas = stored_vals[:, 0]
perimeters = stored_vals[:, 1]
indices = stored_vals[:, 2].astype(int)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(dis_image)
plt.axis('off')
plt.title('Input image')

plt.subplot(2, 3, 2)
plt.imshow(hsv_image)
plt.axis('off')
plt.title('HSV converted image')


plt.subplot(2, 3, 3)
plt.scatter(areas, perimeters, c='blue', marker='o')

for x, y, label in zip(areas, perimeters, indices):
    plt.text(x, (y + 50), str(label), fontsize=9, ha='center', va='center', color='black')  # Change "A" as needed

plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.title('Coutour Area vs Perimeter')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.imshow(mask)
plt.axis('off')
plt.title('Mask generated')

plt.subplot(2, 3, 5)
plt.imshow(animals)
plt.axis('off')
plt.title('Contours cutout')

plt.tight_layout()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
