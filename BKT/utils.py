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








def calculate_percentage_of_ones(dataset):
    total_chars = 0
    total_ones = 0

    for line in dataset.lines:
        total_chars += len(line)
        total_ones += line.count('1')

    percentage = (total_ones / total_chars) * 100 if total_chars > 0 else 0
    return percentage

# Calculate percentages in train.py
#train_percentage = calculate_percentage_of_ones(train_dataset)
#valid_percentage = calculate_percentage_of_ones(valid_dataset)