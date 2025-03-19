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
from model import HMM
from process_dataset import PadAndOneHot

import os
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from process_dataset import get_datasets
import torch.optim as optim
from sklearn.metrics import roc_auc_score, mean_squared_error

from torch.utils.data import Subset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from sklearn.model_selection import KFold


#get dataset








#training

def generate_data(num_sequences, sequence_length):
    return torch.randint(0, 2, (num_sequences, sequence_length))

def train_and_check_parameters(model, num_epochs, data):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Forward pass
        combined_input = torch.cat([data, torch.full((data.shape[0], 1), data.shape[1])], dim=1)
        loss = -torch.log(model(combined_input)).mean()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

        # Get parameters from forward() method
        with torch.no_grad():
            forward_state_priors = F.softmax(model.state_prior_logits, dim=0)
            forward_transition_matrix = model.transition_model.transition_matrix()
            forward_emission_matrix = model.emission_model.emission_matrix()

        print("Parameters from forward() method:")
        print("State priors:", forward_state_priors)
        print("Transition matrix:\n", forward_transition_matrix)
        print("Emission matrix:\n", forward_emission_matrix)
        
        # Get parameters from predict() method
        _, _ = model.predict(combined_input)

        predict_state_priors = F.softmax(model.state_prior_logits, dim=0)
        predict_transition_matrix = model.transition_model.transition_matrix()
        predict_emission_matrix = model.emission_model.emission_matrix()

        print("\nParameters from predict() method:")
        print("State priors:", predict_state_priors)
        print("Transition matrix:\n", predict_transition_matrix)
        print("Emission matrix:\n", predict_emission_matrix)

        # Check if parameters are the same
        state_priors_same = torch.allclose(forward_state_priors, predict_state_priors)
        transition_matrix_same = torch.allclose(forward_transition_matrix, predict_transition_matrix)
        emission_matrix_same = torch.allclose(forward_emission_matrix, predict_emission_matrix)

        print("\nAre parameters the same?")
        print("State priors:", state_priors_same)
        print("Transition matrix:", transition_matrix_same)
        print("Emission matrix:", emission_matrix_same)
        
        
        
#initialize model + if name=='__main__':

        
# Usage
#M, N = 2, 2
#model = HMM(M, N)
#num_sequences = 100
#sequence_length = 10
#data = generate_data(num_sequences, sequence_length)
#num_epochs = 5

#train_and_check_parameters(model, num_epochs, data)



class Trainer:
    def __init__(self, model, Sx, lr, device):
        self.model = model
        self.lr = lr
        self.Sx = Sx
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max')

    def train_batch(self, batch):
        self.model.train()
        x, lengths = batch[0].to(self.device), batch[1].to(self.device)
        combined_input = torch.cat([x, lengths.unsqueeze(1)], dim=1)
        probs = self.model(combined_input)
        log_probs = torch.log(probs + 1e-10)  # Convert to log probabilities
        loss = -log_probs.mean()
        return loss

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                x, lengths = batch[0].to(self.device), batch[1].to(self.device)
                combined_input = torch.cat([x, lengths.unsqueeze(1)], dim=1)

                # Get model predictions
                predictions, _ = self.model.predict(combined_input)

                # Debug: Check if predictions contain NaN
                if torch.isnan(predictions).any():
                    print(f"NaN found in predictions at batch {batch_idx}")
                    print(f"Prediction shape: {predictions.shape}")
                    print(f"Number of NaNs: {torch.isnan(predictions).sum()}")
                    # Print the actual predictions tensor
                    print("Predictions:", predictions)

                # For each sequence
                for i, length in enumerate(lengths):
                    pred = predictions[i, :length]
                    target = x[i, :length]

                    # Debug: Check if current sequence predictions contain NaN
                    if torch.isnan(pred).any():
                        print(f"NaN found in sequence {i} of batch {batch_idx}")
                        print(f"Sequence length: {length}")
                        print("Sequence predictions:", pred)

                    # For accuracy
                    binary_pred = (pred > 0.5)
                    correct += (binary_pred == target).sum().item()
                    total += length.item()

                    # Collect predictions and targets for AUC and RMSE
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

        # Debug: Final check of collected predictions
        all_predictions_np = np.array(all_predictions)
        if np.isnan(all_predictions_np).any():
            print("NaN found in final collected predictions")
            print(f"Total predictions: {len(all_predictions_np)}")
            print(f"Number of NaNs: {np.isnan(all_predictions_np).sum()}")
            nan_indices = np.where(np.isnan(all_predictions_np))[0]
            print(f"Indices of NaN values: {nan_indices[:10]}...")

    # Compute metrics...

        # Compute overall accuracy
        accuracy = correct / total if total > 0 else 0

        # Compute AUC
        try:
            auc = roc_auc_score(all_targets, all_predictions)
        except ValueError:
            # This can happen if all targets are of one class
            auc = float('nan')

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

        return {
            'accuracy': accuracy,
            'auc': auc,
            'rmse': rmse
        }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get datasets
#train_dataset, valid_dataset, train_loader, valid_loader, M, Sx = get_datasets('skill_builder_data_binary_sequences.txt', batch_size=256, validation_split=0.3)



def print_sequence_predictions(model, dataset, device):
    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(dataset))):
            seq = dataset[i]  # This should return a string
            # Convert the string to a list of integers
            seq = [int(bit) for bit in seq]
            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            T = torch.tensor([len(seq)], dtype=torch.long).to(device)
            combined_input = torch.cat([seq_tensor, T.unsqueeze(1)], dim=1)

            predictions, binary_predictions = model.custom_predict(combined_input)

            print(f"\nSequence {i+1}:")
            print(f"Input: {seq}")
            print("Conditional probabilities and predictions:")
            for t in range(len(seq)):
                if t == 0:
                    prob_str = f"P(X_{t} = 1)"
                else:
                    prob_str = f"P(X_{t} = 1 | X_0 = {seq[0]}"
                    for j in range(1, t):
                        prob_str += f", X_{j} = {seq[j]}"
                    prob_str += ")"
                print(f"  t={t+1}: {prob_str} = {predictions[0, t]:.4f}, Prediction: {binary_predictions[0, t].item()}")











    
    
"""def train_and_eval(skill_file, model_class, trainer_class, device):
    # Load the dataset for this skill
    result = get_datasets(skill_file, batch_size=32, validation_split=0.3, random_state=42)
    
    # Check if the dataset creation failed
    if result[0] is None:  # If `train_dataset` is None, skip this skill
        print(f"Skipping {skill_file}: Dataset creation failed or insufficient data.")
        return {"accuracy": float("nan"), "auc": float("nan"), "rmse": float("nan")}

    train_dataset, valid_dataset, train_loader, valid_loader, M, Sx = result

    # Initialize the model and trainer
    model = model_class(M, 2).to(device)
    trainer = trainer_class(model, Sx, lr=0.001, device=device)

    # Train the model
    num_epochs = 30  # Adjust this for testing or final training
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch in train_loader:
            loss = trainer.train_batch(batch)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f}")

    # Evaluate the model
    evaluation_metrics = trainer.evaluate(valid_loader)
    return evaluation_metrics
"""


def train_and_eval(skill_file, model_class, trainer_class, device, num_folds=5):
    # Load the dataset for this skill
    result = get_datasets(skill_file, batch_size=256, validation_split=0.3, random_state=42)

    # Check if the dataset creation failed
    if result[0] is None:  # If `train_dataset` is None, skip this skill
        print(f"Skipping {skill_file}: Dataset creation failed or insufficient data.")
        return {"accuracy": float("nan"), "auc": float("nan"), "rmse": float("nan")}

    train_dataset, valid_dataset, train_loader, valid_loader, M, Sx = result

    # Combine train and validation datasets into a single dataset for cross-validation
    combined_dataset = train_dataset + valid_dataset
    num_samples = len(combined_dataset)
    if num_samples < 5:
        print(f"Skipping {skill_file}: Not enough samples for 5-fold cross-validation (only {num_samples} sample(s)).")
        return {"accuracy": float("nan"), "auc": float("nan"), "rmse": float("nan")}
    # Initialize 5-Fold Cross-Validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Lists to store metrics for each fold
    fold_accuracies = []
    fold_aucs = []
    fold_rmses = []

    for fold, (train_indices, valid_indices) in enumerate(kf.split(combined_dataset)):
        print(f"\n=== Fold {fold + 1}/{num_folds} ===")

        # Create Subset datasets for this fold
        train_subset = Subset(combined_dataset, train_indices)
        valid_subset = Subset(combined_dataset, valid_indices)

        # Create DataLoaders for this fold
        train_loader_fold = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=PadAndOneHot(Sx))
        valid_loader_fold = DataLoader(valid_subset, batch_size=32, shuffle=False, collate_fn=PadAndOneHot(Sx))

        # Initialize the model and trainer for this fold
        model = model_class(M, 2).to(device)
        trainer = trainer_class(model, Sx, lr=0.01, device=device)

        # Train the model
        num_epochs = 30  # Adjust this for testing or final training
        for epoch in range(num_epochs):
            total_loss = 0
            model.train()
            for batch in train_loader_fold:
                loss = trainer.train_batch(batch)
                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f}")

        # Evaluate the model
        evaluation_metrics = trainer.evaluate(valid_loader_fold)
        print(f"Fold {fold + 1} Metrics: {evaluation_metrics}")

        fold_accuracies.append(evaluation_metrics['accuracy'])
        fold_aucs.append(evaluation_metrics['auc'])
        fold_rmses.append(evaluation_metrics['rmse'])

    # Calculate average metrics across folds
    avg_accuracy = np.nanmean(fold_accuracies)
    avg_auc = np.nanmean(fold_aucs)
    avg_rmse = np.nanmean(fold_rmses)

    return {"accuracy": avg_accuracy, "auc": avg_auc, "rmse": avg_rmse}


# Process all skill files in parallel
"""def process_all_skills(skill_dir, model_class, trainer_class, device, num_workers=4):
    # List all skill files
    skill_files = [
        os.path.join(skill_dir, f) for f in os.listdir(skill_dir)
        if os.path.isfile(os.path.join(skill_dir, f)) and f.endswith('.txt')
    ]

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(train_and_eval, skill_files, [model_class] * len(skill_files), [trainer_class] * len(skill_files), [device] * len(skill_files)),
                total=len(skill_files),
                desc="Processing skills"
            )
        )

    # Collect all metrics
    accuracies = [res['accuracy'] for res in results if not pd.isna(res['accuracy'])]
    aucs = [res['auc'] for res in results if not pd.isna(res['auc'])]
    rmses = [res['rmse'] for res in results if not pd.isna(res['rmse'])]

    # Calculate averages
    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else float("nan")
    average_auc = sum(aucs) / len(aucs) if aucs else float("nan")
    average_rmse = sum(rmses) / len(rmses) if rmses else float("nan")

    return results, average_accuracy, average_auc, average_rmse"""


def process_all_skills(skill_dir, model_class, trainer_class, device, num_workers=4):
    # List all skill files
    skill_files = [
        os.path.join(skill_dir, f) for f in os.listdir(skill_dir)
        if os.path.isfile(os.path.join(skill_dir, f)) and f.endswith('.txt')
    ]

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(train_and_eval, skill_files, [model_class] * len(skill_files), [trainer_class] * len(skill_files), [device] * len(skill_files)),
                total=len(skill_files),
                desc="Processing skills"
            )
        )

    # Collect all metrics
    accuracies = [res['accuracy'] for res in results if not pd.isna(res['accuracy'])]
    aucs = [res['auc'] for res in results if not pd.isna(res['auc'])]
    rmses = [res['rmse'] for res in results if not pd.isna(res['rmse'])]

    # Calculate averages
    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else float("nan")
    average_auc = sum(aucs) / len(aucs) if aucs else float("nan")
    average_rmse = sum(rmses) / len(rmses) if rmses else float("nan")

    return results, average_accuracy, average_auc, average_rmse

# Main execution
"""
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    skill_dir = "datasets/assistments17/"  # Path to your dataset directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process all skills and calculate metrics
    results, average_accuracy, average_auc, average_rmse = process_all_skills(skill_dir, HMM, Trainer, device)

    # Print results
    print("\nMetrics per skill:")
    skill_files = [
        f for f in os.listdir(skill_dir) if os.path.isfile(os.path.join(skill_dir, f)) and f.endswith('.txt')
    ]
    for skill_file, metrics in zip(skill_files, results):
        print(f"{skill_file} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, RMSE: {metrics['rmse']:.4f}")

    print("\nAverage Metrics across all skills:")
    print(f"Accuracy: {average_accuracy:.4f}")
    print(f"AUC: {average_auc:.4f}")
    print(f"RMSE: {average_rmse:.4f}")
"""


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    skill_dir = "datasets/simulated/Simulated/"  # Path to your dataset directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process all skills and calculate metrics
    results, average_accuracy, average_auc, average_rmse = process_all_skills(skill_dir, HMM, Trainer, device)

    # Print results
    print("\nMetrics per skill:")
    skill_files = [
        f for f in os.listdir(skill_dir) if os.path.isfile(os.path.join(skill_dir, f)) and f.endswith('.txt')
    ]
    for skill_file, metrics in zip(skill_files, results):
        print(f"{skill_file} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, RMSE: {metrics['rmse']:.4f}")

    print("\nAverage Metrics across all skills:")
    print(f"Accuracy: {average_accuracy:.4f}")
    print(f"AUC: {average_auc:.4f}")
    print(f"RMSE: {average_rmse:.4f}")