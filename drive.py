import socketio
import eventlet
from flask import Flask
import numpy as np
# from keras.models import load_model
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

# def img_preprocess_tensor(img_tensor):
#     # Assuming img_tensor is a PyTorch tensor in (C, H, W) format
#     # Crop
#     img_tensor = img_tensor[:, 60:135, :]
    
#     # Convert color space if needed - this would require custom implementation or prior conversion
#     img_tensor = rgb_to_yuv(img_tensor)

#     # Apply Gaussian blur
#     img_tensor = gaussian_blur(img_tensor, kernel_size=[3, 3])
    
#     # Resize
#     img_tensor = transforms.Resize((66, 200))(img_tensor)
    
#     # Normalize
#     # img_tensor = img_tensor / 255.0
#     return img_tensor

sio = socketio.Server()
app = Flask(__name__) #'__main__' name 
SPEED_LIMIT = 10
model = None

# def img_preprocess(img): 
#     img = img[60:135,:,:] #shortening height of image [height, width, layer]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #image space used for training NVIDIA neural model 
#     img = cv2.GaussianBlur(img, (3,3), 0) #smoothening image technique
#     img = cv2.resize(img, (200,66)) #resizing image as per specifications of NVIDIA neural model 
#     img = img/255 #normalizing image (reduce variance btw image data without visual impact)
#     return img

# def img_preprocess(img):
#     # Crop the image
#     img = img[60:135,:,:]
#     # Convert color space from RGB to YUV
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     # Apply Gaussian blur
#     img = cv2.GaussianBlur(img, (3,3), 0)
#     # Resize the image
#     img = cv2.resize(img, (200, 66))
#     # Normalize the image
#     img = img / 255.0
#     # Convert the numpy array to a PyTorch tensor
#     img_tensor = torch.tensor(img, dtype=torch.float32)
#     # Permute the tensor to match the [Channels, Height, Width] format
#     img_tensor = img_tensor.permute(2, 0, 1)

#     return img_tensor

def preprocess_image_with_cv2(encoded_image):
    # Step 1: Decode the base64 image to bytes, then to a numpy array
    image_data = base64.b64decode(encoded_image)
    nparr = np.fromstring(image_data, np.uint8)
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

    speed = float(data['speed']) #accessing speed
    # image = Image.open(BytesIO(base64.b64decode(data['image'])))
    # image = np.asarray(image)
    # image = img_preprocess(image)
    # image = np.array([image]) #add extra layer to image array?

    # image = np.array(image)
    # image_tensor = img_preprocess(image)
    # # image = np.array([image]) #add extra layer to image array?
    # image_tensor = torch.unsqueeze(image_tensor, 0)  # Adds the batch dimension

    image_tensor = preprocess_image_with_cv2(data['image'])

    steering_angle = 0.0
    if model is not None:
        steering_angle = float(model(image_tensor)) #model predicts a steering angle
    
    throttle = 1.0 - speed/SPEED_LIMIT #adjust throttle based on speed data

    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('I am connected to the Udacity Simulator!')
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

    print("Starting server...")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    # Connect to the simulator
    connect()
