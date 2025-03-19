import torch
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


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, lines, Sx):
        self.lines = lines # list of strings
        self.Sx = Sx
        pad_and_one_hot = PadAndOneHot(self.Sx) # function for generating a minibatch from strings
        self.loader = torch.utils.data.DataLoader(self, batch_size=32, num_workers=1, shuffle=True, collate_fn=pad_and_one_hot)
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        line = self.lines[idx].lstrip(" ").rstrip("\n").rstrip(" ").rstrip("\n")
        return line

class PadAndOneHot:
    def __init__(self, Sx, pad_value=0):
        self.Sx = Sx
        self.pad_value = pad_value

    def __call__(self, batch):
        x = []
        batch_size = len(batch)
        for index in range(batch_size):
            x_ = batch[index]
            x.append([int(c) for c in x_])

        x_lengths = [len(x_) for x_ in x]
        T = max(x_lengths)
        for index in range(batch_size):
            x[index] += [self.pad_value] * (T - len(x[index]))
            x[index] = torch.tensor(x[index])

        x = torch.stack(x)
        x_lengths = torch.tensor(x_lengths)

        #print("In PadAndOneHot:")
        #print("x shape:", x.shape)
        #print("x_lengths:", x_lengths)

        return (x, x_lengths)

"""def get_datasets(file_path, batch_size=32, validation_split=0.3, random_state=42):
    
    #read and parse text file here
    #here just read csv file and process all correct sequence for each user 
    lines = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        print(lines)
    # Strip newline characters and any whitespace
    lines = [line.strip() for line in lines]

    # Filter out any empty lines
    lines = [line for line in lines if line]

    Sx = list(Counter(("".join(lines))).keys())
    train_lines, valid_lines = train_test_split(lines, test_size=validation_split, random_state=random_state)
    train_dataset = TextDataset(train_lines, Sx)
    valid_dataset = TextDataset(valid_lines, Sx)
    M = len(Sx)
    
    
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PadAndOneHot(Sx))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=PadAndOneHot(Sx))

    return train_dataset, valid_dataset, train_loader, valid_loader, M, Sx"""
    
    
    
    
"""def get_datasets(file_path, batch_size=32, validation_split=0.3, random_state=42):
    # Read and parse text file
    lines = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Strip newline characters and any whitespace
    lines = [line.strip() for line in lines]

    # Filter out any empty lines
    lines = [line for line in lines if line]

    # Check if there are enough sequences for splitting
    if len(lines) <= 1:
        print(f"Skipping {file_path}: Not enough data to split (only {len(lines)} sequence(s)).")
        return None, None, None, None, 0, []

    # Calculate the unique states (Sx)
    Sx = list(Counter(("".join(lines))).keys())

    # Split into train and validation sets
    try:
        train_lines, valid_lines = train_test_split(lines, test_size=validation_split, random_state=random_state)
    except ValueError as e:
        print(f"Skipping {file_path}: Error during train-test split - {e}")
        return None, None, None, None, 0, []

    # Create datasets
    train_dataset = TextDataset(train_lines, Sx)
    valid_dataset = TextDataset(valid_lines, Sx)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PadAndOneHot(Sx))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=PadAndOneHot(Sx))

    # Number of unique states
    M = len(Sx)

    return train_dataset, valid_dataset, train_loader, valid_loader, M, Sx"""
    
    
    
def get_datasets(file_path, batch_size=32, validation_split=0.3, random_state=42):
    lines = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Strip newline characters and any whitespace
    lines = [line.strip() for line in lines]

    # Filter out any empty lines
    lines = [line for line in lines if line]

    # Check if there are enough sequences for splitting
    if len(lines) <= 1:
        print(f"Skipping {file_path}: Not enough data to split (only {len(lines)} sequence(s)).")
        return None, None, None, None, 0, []

    # Calculate the unique states (Sx)
    Sx = list(Counter(("".join(lines))).keys())

    # Split into train and validation sets
    try:
        train_lines, valid_lines = train_test_split(lines, test_size=validation_split, random_state=random_state)
    except ValueError as e:
        print(f"Skipping {file_path}: Error during train-test split - {e}")
        return None, None, None, None, 0, []

    # Create datasets
    train_dataset = TextDataset(train_lines, Sx)
    valid_dataset = TextDataset(valid_lines, Sx)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PadAndOneHot(Sx))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=PadAndOneHot(Sx))

    # Number of unique states
    M = len(Sx)

    return train_dataset, valid_dataset, train_loader, valid_loader, M, Sx




#this goes into train.py

#train_dataset, valid_dataset, train_loader, valid_loader, M, Sx = get_datasets('box_and_whisker_binary_sequences.txt', batch_size=32, validation_split=0.3, random_state=42)






