import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from models.utils import match_seq_len

DATASET_DIR = "datasets/datasets/asistments2012/"

class ASSIST2012(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(self.dataset_dir, "preprocessed_df.csv")
        
        # Check if preprocessed files exist
        if os.path.exists(os.path.join(self.dataset_dir, "skill_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "skill_seqs.pkl"), "rb") as f:
                self.skill_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "response_seqs.pkl"), "rb") as f:
                self.response_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "skill_list.pkl"), "rb") as f:
                self.skill_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "user_list.pkl"), "rb") as f:
                self.user_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "skill2idx.pkl"), "rb") as f:
                self.skill2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "user2idx.pkl"), "rb") as f:
                self.user2idx = pickle.load(f)
        else:
            self.skill_seqs, self.response_seqs, self.skill_list, self.user_list, \
            self.skill2idx, self.user2idx = self.preprocess()

        self.num_users = self.user_list.shape[0]
        self.num_q = self.skill_list.shape[0]

        if seq_len:
            self.skill_seqs, self.response_seqs = \
                match_seq_len(self.skill_seqs, self.response_seqs, seq_len)
        
        self.len = len(self.skill_seqs)

    def __getitem__(self, index):
        return self.skill_seqs[index], self.response_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        # Read and sort data by user_id to maintain sequence order
        df = pd.read_csv(self.dataset_path, sep = '\t')\
            .sort_values(by=["user_id"])

        # Get unique users and skills
        user_list = np.unique(df["user_id"].values)
        skill_list = np.unique(df["skill_id"].values)

        # Create mapping dictionaries
        user2idx = {u: idx for idx, u in enumerate(user_list)}
        skill2idx = {s: idx for idx, s in enumerate(skill_list)}

        # Create sequences for each user
        skill_seqs = []
        response_seqs = []
        
        for user in user_list:
            df_user = df[df["user_id"] == user]
            skill_seq = np.array([skill2idx[s] for s in df_user["skill_id"]])
            response_seq = df_user["correct"].values
            skill_seqs.append(skill_seq)
            response_seqs.append(response_seq)

        # Save preprocessed data
        with open(os.path.join(self.dataset_dir, "skill_seqs.pkl"), "wb") as f:
            pickle.dump(skill_seqs, f)
        with open(os.path.join(self.dataset_dir, "response_seqs.pkl"), "wb") as f:
            pickle.dump(response_seqs, f)
        with open(os.path.join(self.dataset_dir, "skill_list.pkl"), "wb") as f:
            pickle.dump(skill_list, f)
        with open(os.path.join(self.dataset_dir, "user_list.pkl"), "wb") as f:
            pickle.dump(user_list, f)
        with open(os.path.join(self.dataset_dir, "skill2idx.pkl"), "wb") as f:
            pickle.dump(skill2idx, f)
        with open(os.path.join(self.dataset_dir, "user2idx.pkl"), "wb") as f:
            pickle.dump(user2idx, f)

        return skill_seqs, response_seqs, skill_list, user_list, skill2idx, user2idx