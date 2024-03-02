# Learning to Drive via Behavioural Cloning

 This project aims to teach a virtual car to drive autonomously by mimicking human driving behavior. Using a deep neural network, the model analyzes driving data collected from human drivers and learns to execute driving actions such as steering (for now, the acceleration is not controlled by the model). This project leverages the power of behavioral cloning, a form of supervised learning, to enable machines to learn complex behaviors directly from data.

 ## Technologies

 1. Python 3.8+
 1. PyTorch (torch and torchvision)
 1. OpenCV
 1. [Udacity Car Simulator](https://github.com/udacity/self-driving-car-sim)

 **Critical dependencies:** These two packages are critical for communicating with the simulator. The exact versions are required.
 1. [python-socketio](https://python-socketio.readthedocs.io/en/stable/) **v4.6.1**
 1. [python-engineio] (https://pypi.org/project/python-engineio/) **v3.13** 

## Getting started

```bash
# Clone the repository
git clone 

# Install the dependncies
pip install -r requirements.txt

# Start the simulator separately

# After the simulator is started, run the following
python drive_pt.py
```

## Technical details

### Data collection

See the Acknowledgements section.

### Image preprocessing

There are two rounds of image preprocessing. The first round is exclusive to training data, wherein different transformations (affine, flip, translation, rotation, brightness adjustment) were applied to the training data. 

The second transformation involves using the YUV color space instead of the RGB color space, applying a Gaussian blur, and resizing the image. These transformations are common for all data.

### Model Architecture
The NVIDIA DAVE-2 CNN model architecture was used. Read more about it [here](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/).

## Results

![video](https://github.com/aritrakar/behavioral-cloning-car/blob/master/demo.mp4)

## Acknowledgements

I got the training data from this [repository](https://github.com/rslim087a/track).
