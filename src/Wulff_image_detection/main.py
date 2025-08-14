import cv2 as cv
import numpy as np
from pathlib import Path
import random
import json

def make_template_contour(img_file):
    img = cv.imread(img_file)

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, (90,166,0), (180,255,255)) #Grass background colors

    mask = mask > 0
    img[mask] = (0, 0, 0)
    img[~mask] = (255, 255, 255)
    img = cv.GaussianBlur(img, (5, 5), 0)
    img = cv.Canny(img, 100, 200)

    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    max_index, biggest_area = 0, 0

    for idx, contour in enumerate(contours):
        area = cv.contourArea(contour, False)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Only takes the bigges contour, and we asume that is the object
        if area > biggest_area:
            max_index = idx
            biggest_area = area

    return contours[max_index]

def process_image(file):

    img = cv.imread(file)

    #Makes the folders in process
    path = Path("process") / file.stem
    path.mkdir(parents=True, exist_ok=True)

    #resize
    img_original = cv.resize(img, fx=0.2, fy=0.2, dsize=None)
    # Save the image
    cv.imwrite(str(path / "(1)Normal_image.jpg"), img_original)

    img = cv.GaussianBlur(img_original, (9, 9), 0) # Gaussian Blur to smooth the image

    cv.imwrite(str(path / "(2)Gaussian.jpg"), img)

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)   # Convert to HSV color space for thresholding

    lower_hsv, upper_hsv = (100,0,0), (180,255,255) # Grass colors
    mask = cv.inRange(img, lower_hsv, upper_hsv) 
    mask = mask > 0 #Making a binary mask
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR) # Convert back to BGR color space

    img[mask] = (255, 255, 255)
    img[~mask] = (0, 0, 0)

    img_masked = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    # Convert to grayscale

    cv.imwrite(str(path / "(3)Masked.jpg"), img)

    canny_img = cv.Canny(img_masked, 200, 300)      # Detect edges

    cv.imwrite(str(path / "(4)canny_img.jpg"), canny_img)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))    # Create a kernel for morphological operations
    filled = cv.morphologyEx(canny_img, cv.MORPH_CLOSE, kernel, iterations=2) # Close gaps in edges

    cv.imwrite(str(path / "(5)filled.jpg"), filled)

    # Optional: fill inside using findContours + drawContours
    contours, _ = cv.findContours(filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv.contourArea(cnt) < 100:  # adjust threshold
            continue
        cv.drawContours(img_original, [cnt], -1, 255, thickness=2)

        matches = []

        for idx, image_path in enumerate(Path('template_images').glob('*.png')):
            template_contour = make_template_contour(image_path)

            score = cv.matchShapes(cnt, template_contour, cv.CONTOURS_MATCH_I1, 0.0)

            #Match must be below threshold
            if score < 0.3:
                matches.append((idx, image_path.stem, score, cnt))

        #find best match
        if matches:
            best_match = min(matches, key=lambda x: x[2])  # Find best match
            idx, name, score, cnt = best_match  
            org = tuple(cnt[0][0])
            #org = (org[0] - 100, org[1] + )
            text = name + str(round(score, 2))
            cv.putText(img_original, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result
    cv.imwrite(str(path / "(6)result.jpg"), img_original)

    #show result
    cv.imshow('Original Image', img_original)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':

    #get the scripts parent folder
    folder = Path(__file__).parent.parent.parent
    folder = folder / 'Pictures/Portfolie_exam_images/capture_1'
    print(folder)

    for idx, image_path in enumerate(folder.glob('*.jpg')):
        if (idx % 10 == 0):
            process_image(image_path)
