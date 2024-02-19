import socketio
import eventlet
from flask import Flask
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch import load
import torch.nn.functional as nnF
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur

class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(3*6*64, 100) # Adjust number of input features
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = nnF.elu(self.conv1(x))
        x = nnF.elu(self.conv2(x))
        x = nnF.elu(self.conv3(x))
        x = nnF.elu(self.conv4(x))
        x = nnF.elu(self.conv5(x))

        # x = x.view(-1, 64*2*33) # Flatten the tensor
        # x = torch.flatten(x)
        x = x.view(x.size(0), -1)

        x = nnF.elu(self.fc1(x))
        # x = self.dropout(x)
        x = nnF.elu(self.fc2(x))
        # x = self.dropout(x)
        x = nnF.elu(self.fc3(x))
        # x = self.dropout(x)

        # Output layer
        x = self.fc4(x)

        return x

print("Starting server...")
sio = socketio.Server()
app = Flask(__name__) #'__main__' name 
print("Server started.?")

SPEED_LIMIT = 10
model = None

def preprocess_image_with_cv2(encoded_image):
    # Step 1: Decode the base64 image to bytes, then to a numpy array
    image_data = base64.b64decode(encoded_image)
    nparr = np.frombuffer(image_data, np.uint8) # np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR to ensure 3 color channels

    # Step 2: Apply preprocessing steps with OpenCV
    # Example preprocessing steps:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Convert color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize the image
    img = img / 255.0  # Normalize the image

    # Step 3: Convert the preprocessed image to a PyTorch tensor
    img_tensor = transforms.ToTensor()(img)  # Converts to tensor and rearranges color channels to CxHxW

    # Add an extra dimension to simulate the batch_size
    img_tensor = img_tensor.unsqueeze(0)  # Shape becomes [1, C, H, W]

    return img_tensor

@sio.on('telemetry')
def telemetry(sid, data):
    global model

    speed = float(data['speed'])
    image_tensor = preprocess_image_with_cv2(data['image'])

    # Convert image_tensor to float32
    image_tensor = image_tensor.type(torch.float32)

    steering_angle = 0.0
    if model is not None:
        steering_angle = float(model(image_tensor)) #model predicts a steering angle
    
    throttle = 1.0 - speed/SPEED_LIMIT #adjust throttle based on speed data

    print(f"{steering_angle} {throttle} {speed}")
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('I am connected to the Udacity Simulator!', sid)
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data ={
        'steering_angle': steering_angle.__str__(), 
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    # model = load_model('model.h5')

    print("Loading model...")
    model = NvidiaModel()
    model_path = "model.pth"
    model.load_state_dict(load(model_path))
    model.eval()
    print("Loaded model.")

    print("Middleware")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
