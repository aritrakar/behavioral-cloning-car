import socketio
import eventlet
from flask import Flask
import numpy as np
import base64
import cv2

import torch
from torch import load
from torchvision import transforms

from model import DAVE2Model

MAX_SPEED = 25
MIN_SPEED = 7
speed_limit = MAX_SPEED
model = None

print("Starting server...")
sio = socketio.Server()
app = Flask(__name__)
print("Server started.")


def preprocess_image_with_cv2(encoded_image):
    # Step 1: Decode the base64 image to bytes, then to a numpy array
    image_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(image_data, np.uint8) # np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR to ensure 3 color channels

    # Step 2: Apply preprocessing steps with OpenCV
    img = img[60:135, :, :] # Remember to crop!
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Convert color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize the image

    # Step 3: Convert the preprocessed image to a PyTorch tensor
    img_tensor = transforms.ToTensor()(img)  # Converts to tensor and rearranges color channels to CxHxW

    # Add an extra dimension to simulate the batch_size
    img_tensor = img_tensor.unsqueeze(0).type(torch.float32)  # Shape becomes [1, C, H, W]

    return img_tensor

@sio.on("telemetry")
def telemetry(sid, data):
    global model, speed_limit

    speed = float(data["speed"])
    throttle = float(data["throttle"])
    image_tensor = preprocess_image_with_cv2(data["image"])

    # Show the image_tensor    
    cv2.imshow("Front camera", image_tensor[0].permute(1, 2, 0).numpy())
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the display window
        return

    steering_angle = 0.0
    if model is not None:
        steering_angle = float(model(image_tensor))
    
    if speed > speed_limit:
        speed_limit = MIN_SPEED  # slow down
    else:
        speed_limit = MAX_SPEED

    throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

    print(f"Steering angle: {steering_angle:.5f}. Throttle: {throttle:10.6f}")
    send_control(steering_angle, throttle)

@sio.on("connect")
def connect(sid, environ):
    print("I am connected to the Simulator!", sid)
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data = {
        "steering_angle": steering_angle.__str__(), 
        "throttle": throttle.__str__()
    })

if __name__ == "__main__":
    print("Loading model...")
    model = DAVE2Model()
    model_path = "models/model_no_dropout_30_epochs.pth"
    model.load_state_dict(load(model_path))
    model.eval()
    print("Loaded model.")

    print("Middleware")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
