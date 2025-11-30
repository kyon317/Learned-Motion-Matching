"""
Neural network model definitions
Fully compatible with the original framework architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Compressor(nn.Module):
    """
    Compressor network - compresses pose data into latent variables
    Used to generate latent variables when training the decompressor
    """
    
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Compressor, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        x = F.elu(self.linear0(x))
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)
        return x.reshape([nbatch, nwindow, -1])


class Decompressor(nn.Module):
    """
    Decompressor network - generates pose from features and latent variables
    Input: feature vector + latent variables
    Output: bone positions, rotations, velocities, etc.
    """
    
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Decompressor, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        nbatch, nwindow = x.shape[:2]
        x = x.reshape([nbatch * nwindow, -1])
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x.reshape([nbatch, nwindow, -1])


class Stepper(nn.Module):
    """
    Stepper network - updates features and latent variables
    Input: current features + latent variables
    Output: feature velocity + latent variable velocity
    """
    
    def __init__(self, input_size, hidden_size=512):
        super(Stepper, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Projector(nn.Module):
    """
    Projector network - projects query features onto nearest features + latent variables
    Input: query feature vector
    Output: projected features + projected latent variables
    """
    
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Projector, self).__init__()
        
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

