import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Function to calculate image noise
def calculate_noise(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray_image)
    std_dev = np.std(gray_image)
    return std_dev / mean


# Function to capture image and update the graph
def capture_image(event):
    global noise_levels

    ret, frame = cap.read()
    if not ret:
        return

    noise = calculate_noise(frame)
    noise_levels.append(noise)

    # Display the captured frame
    ax_frame.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax_noise.clear()
    ax_noise.plot(noise_levels)
    ax_noise.set_xlabel('Capture')
    ax_noise.set_ylabel('Noise Level')
    ax_noise.set_title('Noise Level Variation')

    fig.canvas.draw()


# Initialize webcam
cap = cv2.VideoCapture(0)

noise_levels = []

# Create the Matplotlib figure and axes
fig, (ax_frame, ax_noise) = plt.subplots(2, 1)
plt.subplots_adjust(bottom=0.2)
ax_capture = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_capture = Button(ax_capture, 'Capture')

btn_capture.on_clicked(capture_image)

# Start capturing frames
plt.show()

# Release the webcam
cap.release()
cv2.destroyAllWindows()
