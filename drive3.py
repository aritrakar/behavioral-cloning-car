import socketio
import eventlet
from flask import Flask
import numpy as np
from keras.models import load_model
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

sio = socketio.Server()
app = Flask(__name__) #'__main__' name 
SPEED_LIMIT = 10
model = None

def img_preprocess(img): 
    img = img[60:-25,:,:] #shortening height of image [height, width, layer]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #image space used for training NVIDIA neural model 
    img = cv2.GaussianBlur(img, (3,3), 0) #smoothening image technique
    img = cv2.resize(img, (200,66))
    img = img/255 #normalizing image (reduce variance btw image data without visual impact)
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    global model

    speed = float(data['speed']) #accessing speed
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image]) #add extra layer to image array?
    print(image)

    steering_angle = 0.0
    if model is not None:
        print("Found model")
        steering_angle = float(model.predict(image))
    
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
    print("Loading model...")
    model = load_model("models/model_old.h5")
    print("Model loaded. Summary:", model.summary())

    print("Starting server...")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
