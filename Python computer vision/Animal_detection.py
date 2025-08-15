import numpy as np
import cv2


import matplotlib.pyplot as plt


animal_huMoments_old = [
(np.array([2.36169072e-01, 2.17350011e-02, 1.17960212e-04, 4.64883236e-04, 1.03820211e-07, 6.82483761e-05, 3.27520501e-08]), "Antelope"),
(np.array([3.00897437e-01, 5.98463195e-02, 3.72894655e-03, 1.48597653e-03, 3.38272127e-06, 2.91509190e-04, 8.90339768e-07]), "Zebra"), 
(np.array([3.35128769e-01, 8.46137006e-02, 1.41238550e-03, 8.70711121e-04, 9.65542890e-07, 2.52998571e-04, -8.33788856e-09]), "Lion"),
(np.array([3.35944104e-01, 8.47196546e-02, 2.48696965e-03, 1.41426456e-03, 2.64174337e-06, 3.89449845e-04, -2.36992237e-07]), "Rhino"),
(np.array([2.74433840e-01, 4.35331829e-02, 1.72701051e-03, 3.67493877e-04, 2.58260712e-07, 2.99085496e-05, -1.37891731e-07]), "Elephant"),
(np.array([2.70362332e-01, 4.09247858e-02, 2.94455871e-03, 8.33865451e-04, 1.12140135e-06, 9.91531579e-05, -6.70635815e-07]), "Hippo")]

animal_huMoments = [
(np.array([2.64715307e-01, 3.04745422e-02, 3.68543692e-04, 1.02669871e-03, 6.10939056e-07, 1.79125323e-04, 1.60035037e-07]), "Antelope"),
(np.array([3.25557048e-01, 7.38786639e-02, 4.67947835e-03, 2.23339726e-03, 7.08708626e-06, 5.07668707e-04, 1.37990102e-06]), "Zebra"), 
(np.array([3.71532739e-01, 1.08488313e-01, 4.85084292e-03, 3.27610637e-03, 1.30600294e-05, 1.07906733e-03, 3.36786005e-08]), "Lion"),
(np.array([3.37503056e-01, 8.56412104e-02, 2.33793190e-03, 1.33973649e-03, 2.36156554e-06, 3.70294112e-04, -2.12126020e-07]), "Rhino"),
(np.array([2.71181323e-01, 4.13444790e-02, 1.41532936e-03, 5.06349882e-04, 4.24014765e-07, 6.63479796e-05, -6.28807824e-08]), "Elephant"),
(np.array([2.87423401e-01, 4.95896295e-02, 4.04005449e-03, 1.49108822e-03, 3.43295687e-06, 2.42943985e-04, -1.26822889e-06]), "Hippo")]


#[3.71532739e-01, 1.08488313e-01, 4.85084292e-03, 3.27610637e-03, 1.30600294e-05, 1.07906733e-03, 3.36786005e-08]
#[3.68630683e-01, 1.06982320e-01, 3.43215097e-03, 2.30838685e-03, 6.49748812e-06, 7.55027878e-04, 1.06592739e-08]
#[3.86516267e-01,  1.20874070e-01,  4.13693078e-03,  2.80644518e-03, 9.56126509e-06,  9.72836020e-04, -1.57096387e-07]

#[0.374388284500000, 0.111666105000000, 0.003212716420000, 0.002234580480000, 6.086034125000000e-06, 7.473723410000000e-04, -4.266164399999999e-09]


#[0.384979338000000, 0.119832625500000, 0.002719855635000, 0.001922477050000, 4.407918030000000e-06, 6.644579509999999e-04, 1.502157980000000e-08]

def draw_largest_contour(image, contours):

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw it on a copy of the image
    output_img = image.copy()
    cv2.drawContours(output_img, [largest_contour], -1, (0, 255, 0), 2)

    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # Plot using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(output_rgb)
    plt.title("Largest Contour")
    plt.axis('off')
    plt.show()



def blur_and_filter(img):
    #blur_img = cv2.blur(img, (3,3)) # Box filter - AxB dimension
    filtered_img = cv2.bilateralFilter(img, 8, 150, 150) # diameter, different colours considered similar, distance considered neightbouring
    return filtered_img

def grass_masking(filtered_img):
    hsv_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)
    grass_lower = np.array([30, 30, 30]) # Hue - Saturation - Value  --  maybe lower saturation and value?
    grass_upper = np.array([85, 255, 255])
    grass_mask = cv2.inRange(hsv_img, grass_lower, grass_upper)

    shadow_lower = np.array([0, 0, 0]) # Hue - Saturation - Value  --  maybe lower saturation and value?
    shadow_upper = np.array([180, 255, 50])
    shadow_mask = cv2.inRange(hsv_img, shadow_lower, shadow_upper)

    mask = cv2.bitwise_or(grass_mask, shadow_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # removes small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # fills small holes

    return mask

def white_masking(filtered_img):
    hsv_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)
    white_lower = np.array([0, 0, 250])  
    white_upper = np.array([180, 30, 255]) 
    mask = cv2.inRange(hsv_img, white_lower, white_upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # removes small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # fills small holes
    return mask

def find_edges(masked_img):
    edge_img = cv2.Canny(masked_img, 100, 200)
    return edge_img

def find_contours(edge_img):
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def contour_analysis(contours, contour_min_area, min_dim):
    print("\nAmount of detected contours:")
    print(len(contours))
    min_w = min_dim
    min_h = min_dim

    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimiter = cv2.arcLength(cnt, True)
        
        if area < contour_min_area:
            continue
        
        filtered_contours.append(cnt)


    

    amount = 0
    contour_data = []
    for cnt in filtered_contours:
        moment = cv2.moments(cnt)
        huMoment = cv2.HuMoments(moment)

        if moment["m00"] != 0:
            cx = int(moment["m10"] / moment["m00"])
            cy = int(moment["m01"] / moment["m00"])
        else:
            cx, cy = 0, 0

        amount = amount + 1
        contour_data.append((huMoment.flatten(), (cx,cy), "TBD", cnt))

    print("\nAmount after filtering for size:")
    print(len(contour_data))
    
    return contour_data




def detect_animal_1(contour_data, diff):
    animal_contour_data = []
    amount = 0
    for cnt_data in contour_data:
        huMoment, (cx,cy), contour_name, contour = cnt_data
        print(huMoment.flatten())
        
        for animals in animal_huMoments:
            animal_huMoment, animal_name = animals
            i = 0
            match = True
            print(animal_name)
            for data in huMoment:
                #absdiff = np.abs(data - animal_huMoment[i])
                reldiff = np.abs((data - animal_huMoment[i])/data)
                #print("relative diff: ")
                #print(reldiff)

                if reldiff > diff:
                    match = False
                    break
                i = i + 1

            if match == True:
                animal_contour_data.append((huMoment.flatten(), (cx,cy), animal_name, contour))
                amount = amount + 1
    print("\nAmount of detected animals:")
    print(len(animal_contour_data))
    return animal_contour_data

def log_normalize(huMoment):
    return -np.sign(huMoment) * np.log10(np.abs(huMoment) + 1e-10)
def detect_animal_2(contour_data, diff):
    animal_contour_data = []
    for cnt_data in contour_data:
        huMoment, (cx,cy), contour_name, contour = cnt_data
        data = None
        lowest_comp = 10
        print("\n\n NEW CONTOUR\n")
        print(huMoment.flatten())
        for animals in animal_huMoments:
            animal_huMoment, animal_name = animals
            #weights = np.array([1.0, 0.8, 0.5, 0.5, 0.3, 0.3, 0.1])
            comp = np.linalg.norm((log_normalize(animal_huMoment) - log_normalize(huMoment)))
            print(animal_name)
            print(comp)
            if comp < lowest_comp:
                lowest_comp = comp
                data = (huMoment.flatten(), (cx,cy), animal_name, contour)
                
        if lowest_comp < diff:
            animal_contour_data.append(data)
            print("\nGOT ONE")
            print(data[2])
            print(lowest_comp)

    print("\nAmount of detected animals:")
    print(len(animal_contour_data))
    return animal_contour_data








def draw_animal_contours(background_img, animal_contour_data):
    for animal_cnt_data in animal_contour_data:
        huMoment, (cx,cy), animal_name, contour = animal_cnt_data
        cv2.drawContours(background_img, contour, -1, (0, 0, 255), 10) # contour index, colour, linewidth
        cv2.putText(background_img, animal_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 10)

def draw_only_animals(img, animal_contour_data):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for data in animal_contour_data:
        huMoment, (cx,cy), animal_name, contour = data
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    return cv2.bitwise_and(img, img, mask=mask)
    


        






def plot_images(images):
    fig, ax = plt.subplots(2, 4, figsize=(10, 8))
    ax[0, 0].imshow(cv2.cvtColor(images[0][0], cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title(images[0][1])

    ax[0, 1].imshow(cv2.cvtColor(images[1][0], cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title(images[1][1])

    ax[0, 2].imshow(cv2.cvtColor(images[2][0], cv2.COLOR_BGR2RGB), cmap='gray')
    ax[0, 2].set_title(images[2][1])

    ax[0, 3].imshow(cv2.cvtColor(images[3][0], cv2.COLOR_BGR2RGB), cmap='gray')
    ax[0, 3].set_title(images[3][1])

    ax[1, 0].imshow(cv2.cvtColor(images[4][0], cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title(images[4][1])

    ax[1, 1].imshow(cv2.cvtColor(images[5][0], cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title(images[5][1])

    ax[1, 2].imshow(cv2.cvtColor(images[6][0], cv2.COLOR_BGR2RGB))
    ax[1, 2].set_title(images[6][1])

    ax[1, 3].imshow(cv2.cvtColor(images[7][0], cv2.COLOR_BGR2RGB))
    ax[1, 3].set_title(images[7][1])
    plt.show()


def check_image(image_link):
    img = cv2.imread(image_link, 1)

    filtered_img = blur_and_filter(img)
    grass_mask = grass_masking(filtered_img)
    edge_img = find_edges(grass_mask)

    contours = find_contours(edge_img)


    
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 10) # contour index, colour, linewidth
    
    contour_min_area = 5000 # OLD VALUE = 20
    min_dim = 0.1
    contour_data = contour_analysis(contours, contour_min_area, min_dim)

    mag = 4 # Maximum magnitude of the hu moment difference vector
    animal_contour_data = detect_animal_2(contour_data, mag)

    
    filtered_contours = []
    filtered_contour_img = img.copy()
    for cnt in contour_data:
        huMoment, (cx,cy), contour_name, filtered_contour = cnt
        filtered_contours.append(filtered_contour)
        cv2.drawContours(filtered_contour_img, filtered_contour, -1, (0, 0, 255), 10) # contour index, colour, linewidth

    #draw_largest_contour(img, filtered_contours)


    animals_img = img.copy()
    draw_animal_contours(animals_img, animal_contour_data)

    only_animals_img = draw_only_animals(img, animal_contour_data)

    
    images = [(img, "1 Original"), (filtered_img, "2 Filtered"), 
              (grass_mask, "3 Masked"), (edge_img, "4 Edge detection"), (contour_img, "5 Applied contours"), 
              (filtered_contour_img, "6 Filtered contours"), (animals_img, "8 Detected animals"), 
              (only_animals_img, "8 Only animals")]
    plot_images(images)

#check_image('ANIMAL DETECTION/img_xxx.jpg')
#check_image('ANIMAL DETECTION/capture_1/img_52.jpg')
check_image('ANIMAL DETECTION/capture_1/img_25.jpg')

# check_image('ANIMAL DETECTION/capture_1/img_93.jpg') - works

