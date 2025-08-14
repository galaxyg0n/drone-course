import cv2 as cv
import numpy as np
from pathlib import Path
import random
import json
import matplotlib.pyplot as plt

def get_data_from_image(img_file):
    img = cv.imread(img_file)

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, (90,166,0), (180,255,255))
    mask = mask > 0
    img[mask] = (0, 0, 0)
    img[~mask] = (255, 255, 255)
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.Canny(img, 100, 200)

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    max_index, biggest_area = 0, 0
    color = (0, 0, 0)

    for idx, contour in enumerate(contours):
        area = cv.contourArea(contour, False)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Only takes the bigges contour, and we asume that is the object
        if area > biggest_area:
            max_index = idx
            biggest_area = area

    contour_img = np.zeros_like(img)
    #show contours
    for cnt in contours:
        cv.drawContours(contour_img, [cnt], -1, color, 2)

    P = cv.arcLength(contours[max_index], True)
    A = cv.contourArea(contours[max_index])
    compactness = (P * P) / A
    print(img_file.stem,f"Compactness: {compactness}")
    #return humoment and compactness
    hu_moments = cv.HuMoments(cv.moments(contours[max_index])).flatten()
    #normalize hu moments
    hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    return hu_moments, compactness


















hu_moments_list = []
compactness_list = []
names = []

# Collect data first
for image_path in Path('template_images').glob('*.png'):
    hu_moments, compactness = get_data_from_image(image_path)
    hu_moments_list.append(hu_moments)
    compactness_list.append(compactness)
    names.append(image_path.stem)

hu_moments_array = np.array(hu_moments_list)  # shape: (n_images, 7)

plt.figure(figsize=(12, 6))

# Hu moments subplot
plt.subplot(1, 3, 1)
for i, name in enumerate(names):
    plt.plot(range(1, 8), hu_moments_array[i], marker='o', label=name)
plt.title('Hu Moments')
plt.xlabel('Moment')
plt.ylabel('Value')
plt.legend()

# Compactness subplot
plt.subplot(1, 3, 2)
plt.bar(names, compactness_list)  # bar chart for clarity
plt.title('Compactness')
plt.xlabel('Image')
plt.ylabel('Value')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
hu_idx = 1
plt.scatter(hu_moments_array[:, hu_idx], compactness_list, color='b')
#names
for i, name in enumerate(names):
    plt.annotate(name, (hu_moments_array[i, hu_idx], compactness_list[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Hu Moment 7 vs Compactness')
plt.xlabel('Hu Moment 7')
plt.ylabel('Compactness')

plt.tight_layout()
plt.show()