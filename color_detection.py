import cv2
import numpy as np

# Capture video from the default camera (usually 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, blue, and yellow lines
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create masks for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine the masks
    mask_combined = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_yellow))

    # Find contours in the combined mask
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour and display a text label
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        color = ""

        if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
            if np.any(mask_red[y:y+h, x:x+w] > 0):
                color = "Red"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
            elif np.any(mask_blue[y:y+h, x:x+w] > 0):
                color = "Blue"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
            elif np.any(mask_yellow[y:y+h, x:x+w] > 0):
                color = "Yellow"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle

        cv2.putText(frame, f"{color} detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Color Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
