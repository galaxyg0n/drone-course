import cv2 as cv

cam = cv.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')

while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv.destroyAllWindows()