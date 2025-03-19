import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from random import random, randint
from opacus.accountants import RDPAccountant
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from utils import EarlyStopping
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from opacus.schedulers import ExponentialNoise, LambdaNoise, StepNoise
import matplotlib.pyplot as plt
from opacus.utils.batch_memory_manager import BatchMemoryManager
# For Train MLM
# 15% <MASK>, 80% of 15% are real <MASK>, 10% of 15% are reverse, 10% of 15% are not changed
def Mlm4BertTrain(r_seqs, mask_seqs):
    #|r_seqs| = (bs, n)

    mlm_r_seqs = []
    mlm_idxs = []

    # <PAD> is -1
    for r_seq, mask_seq in zip(r_seqs, mask_seqs):
        r_len = r_seq.size(0)
        # real_r_seq: r_seq with no <PAD>
        real_r_seq = torch.masked_select(r_seq, mask_seq).cpu()
        real_r_seq_len = real_r_seq.size(0)

        mlm_idx = np.random.choice(real_r_seq_len, int(real_r_seq_len*0.15), replace=False)

        for idx in mlm_idx:
            if random() < 0.8: # 15% of 80% are <MASK>
                real_r_seq[idx] = 2 # <MASK> is 2
            elif random() < 0.5: # 15% of 10% are random among the 0 or 1
                real_r_seq[idx] = randint(0, 1)
            # 15% of 10% are same with original

        # cover the PAD(-1)
        pad_len = r_len - real_r_seq_len
        # <PAD> is 3
        pad_seq = torch.full((1, pad_len), 3).squeeze(0) 
        # combine the <PAD>
        pad_r_seq = torch.cat((real_r_seq, pad_seq), dim=-1)
        # append to the mlm_r_seqs
        mlm_r_seqs.append(pad_r_seq)

        # <MASK> idx bool
        # make zero vector with r_len size
        mlm_zeros = np.zeros(shape=(r_len, ))
        # mlm_idx are 1
        mlm_zeros[mlm_idx] = 1
        # append to the mlm_idxs
        mlm_idxs.append(mlm_zeros)

    mlm_r_seqs = torch.stack(mlm_r_seqs)
    mlm_idxs = torch.BoolTensor(mlm_idxs)

    # mlm_r_seqs: masked r_seqs
    # mlm_idx: masked idx
    return mlm_r_seqs, mlm_idxs
    # |mlm_r_seqs| = (bs, n)
    # |mask_seqs| = (bs, n)

# For Test MLM
# The last of seq will be changed to the <MASK>
def Mlm4BertTest(r_seqs, mask_seqs):
    #|r_seqs| = (bs, n)

    mlm_r_seqs = []
    mlm_idxs = []

    for r_seq, mask_seq in zip(r_seqs, mask_seqs):
        r_len = r_seq.size(0)

        real_r_seq = torch.masked_select(r_seq, mask_seq).cpu()
        real_r_seq_len = real_r_seq.size(0)

        # last index of real_r_seq
        mlm_idx = real_r_seq_len - 1
        # last index get a <MASK>, <MASK> is 2
        real_r_seq[mlm_idx] = 2

        pad_len = r_len - real_r_seq_len
        pad_seq = torch.full((1, pad_len), 3).squeeze(0) # <PAD> is 3
        pad_r_seq = torch.cat((real_r_seq, pad_seq), dim=-1)
        mlm_r_seqs.append(pad_r_seq)

        mlm_zeros = np.zeros(shape=(r_len, ))
        mlm_zeros[mlm_idx] = 1
        mlm_idxs.append(mlm_zeros)

    mlm_r_seqs = torch.stack(mlm_r_seqs)
    mlm_idxs = torch.BoolTensor(mlm_idxs)

    return mlm_r_seqs, mlm_idxs
    # |mlm_r_seqs| = (bs, n)
    # |mask_seqs| = (bs, n)

class MonaCoBERT_Trainer():

    def __init__(
        self, 
        model, 
        optimizer, 
        n_epochs, 
        device, 
        num_q, 
        crit, 
        max_seq_len, 
        grad_acc=False, 
        grad_acc_iter=4
        ):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device
        self.num_q = num_q
        self.crit = crit
        self.max_seq_len = max_seq_len
        self.grad_acc = grad_acc #gradient accumulation
        self.grad_acc_iter = grad_acc_iter
    
    def _train(self, train_loader, metric_name, delta, privacy_engine, max_physical_batch_size=16):
        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []
    
        self.model.train()
    
        with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=16,
        optimizer=self.optimizer
    ) as memory_safe_data_loader:
            for data in tqdm(memory_safe_data_loader):
                self.optimizer.zero_grad()
            
                if any(d.shape[0] == 0 for d in data):
                   print(f"Skipping empty batch")
                   continue
            
                q_seqs, r_seqs, pid_seqs, mask_seqs = data
            
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                pid_seqs = pid_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)
            
                real_seqs = r_seqs.clone()
            
                mlm_r_seqs, mlm_idxs = Mlm4BertTrain(r_seqs, mask_seqs)
                mlm_r_seqs = mlm_r_seqs.to(self.device)
                mlm_idxs = mlm_idxs.to(self.device)
            
                y_hat = self.model(
                q_seqs.long(), 
                mlm_r_seqs.long(),
                pid_seqs.long(),
                mask_seqs.long()
                ).to(self.device)
            
                y_hat = y_hat.squeeze()
                y_hat = torch.masked_select(y_hat, mlm_idxs)
                correct = torch.masked_select(real_seqs, mlm_idxs)
            
                loss = self.crit(y_hat, correct)
            
                loss.backward()
                self.optimizer.step()
            
            y_trues.append(correct)
            y_scores.append(y_hat)
            loss_list.append(loss.item())
    
        if privacy_engine:
            eps, alpha = privacy_engine.accountant.get_privacy_spent(delta=delta)
            print(f" | (ε = {eps:.2f}, δ = {delta}), alpha = {alpha}")
    
        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()
    
        auc_score += metrics.roc_auc_score(y_trues, y_scores)
    
        loss_result = np.mean(loss_list)
    
        if metric_name == "AUC":
            return auc_score, eps
        elif metric_name == "RMSE":
            return loss_result, eps


    def _validate(self, valid_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(valid_loader):
                self.model.eval()
                q_seqs, r_seqs, pid_seqs, mask_seqs = data
                
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                pid_seqs = pid_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                real_seqs = r_seqs.clone()

                mlm_r_seqs, mlm_idxs = Mlm4BertTest(r_seqs, mask_seqs)

                mlm_r_seqs = mlm_r_seqs.to(self.device)
                mlm_idxs = mlm_idxs.to(self.device)

                y_hat = self.model(
                    q_seqs.long(),
                    mlm_r_seqs.long(),
                    pid_seqs.long(),
                    mask_seqs.long()
                ).to(self.device)

                y_hat = y_hat.squeeze()

                y_hat = torch.masked_select(y_hat, mlm_idxs)
                correct = torch.masked_select(real_seqs, mlm_idxs)

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result

    def _test(self, test_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, pid_seqs, mask_seqs = data
                
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                pid_seqs = pid_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                real_seqs = r_seqs.clone()

                mlm_r_seqs, mlm_idxs = Mlm4BertTest(r_seqs, mask_seqs)

                mlm_r_seqs = mlm_r_seqs.to(self.device)
                mlm_idxs = mlm_idxs.to(self.device)

                y_hat = self.model(
                    q_seqs.long(),
                    mlm_r_seqs.long(),
                    pid_seqs.long(),
                    mask_seqs.long()
                ).to(self.device)

                y_hat = y_hat.squeeze()

                y_hat = torch.masked_select(y_hat, mlm_idxs)
                correct = torch.masked_select(real_seqs, mlm_idxs)

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result

    # train use the _train, _validate, _test
    def train(self, train_loader, valid_loader, test_loader, config):
        
        if config.crit == "binary_cross_entropy":
            best_valid_score = 0
            best_test_score = 0
            metric_name = "AUC"
        elif config.crit == "rmse":
            best_valid_score = float('inf')
            best_test_score = float('inf')
            metric_name = "RMSE"
        
        train_scores = []
        valid_scores = []
        test_scores = []

        accountant = 'rdp'
        early_stopping = EarlyStopping(metric_name=metric_name,best_score=best_valid_score)
        secure_mode = False
        
        delta = 0.9/len(train_loader)
        #delta = 10.0
       
        num_epochs = self.n_epochs
        #self.model.train()
        nm = 1.5
        max_grad_norm = 4.0
        privacy_engine = PrivacyEngine(secure_mode=secure_mode, accountant = accountant)
        self.model, self.optimizer, train_loader = privacy_engine.make_private(module = self.model, optimizer = self.optimizer, data_loader = train_loader, noise_multiplier = nm, max_grad_norm = max_grad_norm,  poisson_sampling = True, clipping = 'adaptive', target_unclipped_quantile=1.0,
    clipbound_learning_rate=0.2,
    max_clipbound=5.0,
    min_clipbound=1.0,
    unclipped_num_std=1.0,)
        #noise_scheduler = StepNoise(self.optimizer, step_size=5, gamma=0.95)
        epsilons = []
        # Train and Valid Session
        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            # Training Session
            train_score, ep = self._train(train_loader, metric_name, delta, privacy_engine)
            epsilons.append(ep)
            valid_score = self._validate(valid_loader, metric_name)
            test_score = self._test(test_loader, metric_name)

            # train, test record 저장
            train_scores.append(train_score)
            valid_scores.append(valid_score)
            test_scores.append(test_score)

            # early stop
            train_scores_avg = np.average(train_scores)
            valid_scores_avg = np.average(valid_scores)
            early_stopping(valid_scores_avg, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if config.crit == "binary_cross_entropy":
                if test_score >= best_test_score:
                    best_test_score = test_score
            elif config.crit == "rmse":
                if test_score <= best_test_score:
                    best_test_score = test_score

            print("Epoch(%d/%d) result: train_score=%.4f  valid_score=%.4f test_score=%.4f best_test_score=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_score,
                valid_score,
                test_score,
                best_test_score,
            ))

        print("\n")
        print("The Best Test Score(" + metric_name + ") in Testing Session is %.4f" % (
                best_test_score,
            ))
        print("\n")
        
        self.model.load_state_dict(torch.load("../checkpoints/checkpoint.pt"))

        return train_scores, valid_scores, \
            best_valid_score, best_test_score, epsilons

    
