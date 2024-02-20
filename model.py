import torch.nn as nn
import torch.nn.functional as nnF

class DAVE2Model(nn.Module):
    def __init__(self):
        super(DAVE2Model, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ELU()
            nn.Dropout(0.5)
        )
        
        # Dropping the fully connected layers to a flatten operation
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # nn.Linear(64 * 2 * 33, 100),  # Assuming the input image size leads to this dimension
            nn.Linear(1152, 100),  # Assuming the input image size leads to this dimension
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            # nn.ELU(),
            nn.Linear(10, 1)  # Output layer
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x) # or x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

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

        x = x.view(x.size(0), -1)

        x = nnF.elu(self.fc1(x))
        x = self.dropout(x)
        x = nnF.elu(self.fc2(x))
        x = self.dropout(x)
        x = nnF.elu(self.fc3(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc4(x)

        return x

class NvidiaModelNoDropout(nn.Module):
    def __init__(self):
        super(NvidiaModelNoDropout, self).__init__()

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
    