import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import sys


import os
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader





class TransitionModel(nn.Module):
    def __init__(self, N):
        super(TransitionModel, self).__init__()
        self.N = N
        self.transition_logits = nn.Parameter(torch.tensor([[0.9, 0.1], [0.85, 0.15]]))

    def forward(self, alpha):
        # Apply softmax to get probability matrix
        transition_probs = F.softmax(self.transition_logits, dim=1)
        return torch.matmul(alpha, transition_probs)

    def transition_matrix(self):
        return F.softmax(self.transition_logits, dim=1)
    
    
    
    
    
class EmissionModel(torch.nn.Module):
    def __init__(self, N, M):
        super(EmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.emission_logits = nn.Parameter(torch.tensor([[0.8, 0.2], [0.6, 0.4]]))

    def forward(self, x_t):
        """
        x_t : LongTensor of shape (batch size)
        Get observation probabilities
        """
        emission_probs = F.softmax(self.emission_logits, dim=1)
        out = emission_probs[:, x_t].transpose(0, 1)
        return out

    def emission_matrix(self):
        return F.softmax(self.emission_logits, dim=1)
    
    




# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class HMM(nn.Module):
    def __init__(self, M, N):
        super(HMM, self).__init__()
        self.M = M  # number of possible observations
        self.N = N  # number of states
        self.state_prior_logits = nn.Parameter(torch.tensor([0.3, 0.7]))
        self.transition_model = TransitionModel(self.N)
        self.emission_model = EmissionModel(self.N, self.M)

    def forward(self, combined_input):
        x, T = combined_input[:, :-1], combined_input[:, -1].long()
        batch_size = x.shape[0]
        T_max = x.shape[1]

        # Initialize alpha
        alpha = torch.zeros(batch_size, T_max, self.N)

        # Normalize trained state_priors parameters
        state_priors = F.softmax(self.state_prior_logits, dim=0)

        # Compute initial alpha
        alpha_0 = self.emission_model(x[:, 0]) * state_priors.unsqueeze(0)
        alpha = torch.cat([alpha_0.unsqueeze(1), torch.zeros(batch_size, T_max-1, self.N, device=x.device)], dim=1)

        # Forward algorithm
        for t in range(1, T_max):
            alpha[:, t, :] = self.emission_model(x[:, t]) * self.transition_model(alpha[:, t-1, :].clone())

        # Compute final probabilities
        probs = alpha.sum(dim=2)
        final_probs = torch.gather(probs, 1, T.view(-1, 1) - 1)

        return final_probs

    def predict(self, combined_input):
        x, T = combined_input[:, :-1], combined_input[:, -1].long()
        batch_size, seq_length = x.shape
        predictions = torch.zeros_like(x, dtype=torch.float, device=x.device)

        # Get normalized parameters
        normalized_state_priors = F.softmax(self.state_prior_logits, dim=0)
        normalized_transition_matrix = self.transition_model.transition_matrix()
        normalized_emission_matrix = self.emission_model.emission_matrix()

        # Calculate initial prediction (t=0)
        predictions[:, 0] = (normalized_state_priors * normalized_emission_matrix[:, 1]).sum()

        # Calculate alpha for t=0
        alpha_0 = self.emission_model(x[:, 0]) * normalized_state_priors.unsqueeze(0)
        alpha = torch.cat([alpha_0.unsqueeze(1),
                          torch.zeros(batch_size, seq_length-1, self.N, device=x.device)], dim=1)

        eps = 1e-10  # Small constant to prevent division by zero

        for t in range(1, seq_length):
            # Calculate joint probability
            joint_prob = torch.matmul(alpha[:, t-1, :], normalized_transition_matrix)

            # Calculate emission probability for X_t = 1
            emission_prob_1 = normalized_emission_matrix[:, 1]

            # Calculate numerator
            numerator = (joint_prob * emission_prob_1).sum(dim=1)

            # Calculate denominator with numerical stability
            denominator = alpha[:, t-1, :].sum(dim=1)
            denominator = torch.clamp(denominator, min=eps)  # Prevent division by zero

            # Calculate prediction with numerical stability
            predictions[:, t] = numerator / denominator
            predictions[:, t] = torch.clamp(predictions[:, t], min=0.0, max=1.0)  # Ensure valid probabilities

            # Update alpha for current time step
            alpha[:, t, :] = self.emission_model(x[:, t]) * joint_prob

        binary_predictions = (predictions > 0.5).int()
        return predictions, binary_predictions

    def custom_predict(self, combined_input):
        x, T = combined_input[:, :-1], combined_input[:, -1].long()
        batch_size, seq_length = x.shape
        predictions = torch.zeros_like(x, dtype=torch.float)

        # Set p_1 = 0.5 for all sequences
        predictions[:, 0] = 0.5

        # Compute cumulative sum of 1's for each position
        cumsum_ones = torch.cumsum(x, dim=1)

        # Create a tensor representing (i-1) for each position
        position_indices = torch.arange(1, seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Compute predictions for positions 2 to n
        numerator = 1 + cumsum_ones[:, :-1]
        denominator = 2 + position_indices
        predictions[:, 1:] = numerator / denominator
        binary_predictions = (predictions > 0.5).int()

        return predictions, binary_predictions