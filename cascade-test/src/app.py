import cv2 as cv
import os

# Import pre-build face cascade 
face_cascade = cv.CascadeClassifier(os.getcwd() + '/Cascades/haarcascade_frontalface_default.xml')

# Connect to laptop camera
cam = cv.VideoCapture(0)

# Get the default frame width and height
frame_width  = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

# Function for classifying face and drawing rectangle + text on top
def face_detect(img):
    face_img = img.copy()                                                                                # Takes a copy of the current frame 
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)                 # Harr Cascade classifier to detect face using imported cascade

    for(x, y, w, h) in face_rect:                                                                        # Face coordinates       
        cv.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 255), 2)                                 # Draw red rectangle on detected face
        cv.putText(face_img, 'Nerd alert!', (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0), 2)  # Add text above rectangle

    return face_img


while True:
    ret, frame = cam.read()                                                                              # Read the camera input
    cv.putText(frame, 'Use Q to quit!', (25, 50), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)      # Put info text on the image
    cv.imshow('Cam', face_detect(frame))                                                                 # Show the live image + draw rectangle on faces

    if cv.waitKey(1) == ord('q'):                                                                        # Wait for user input and quit on 'q'
        break

cam.release()                                                                                            
cv.destroyAllWindows()
