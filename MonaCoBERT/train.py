import numpy as np
import datetime

import torch
from get_modules.get_loaders import get_loaders
from get_modules.get_models import get_models
from get_modules.get_trainers import get_trainers
from utils import get_optimizers, get_crits, recorder, visualizer
from opacus.accountants import RDPAccountant
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from utils import EarlyStopping
from opacus import PrivacyEngine
from define_argparser import define_argparser
import matplotlib.pyplot as plt

def main(config, train_loader=None, valid_loader=None, test_loader=None, num_q=None, num_r=None, num_pid=None, num_diff=None):
    # 0. device setting
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    # 1. get dataset from loader
    # 1-1. use fivefold
    if config.fivefold == True:
        train_loader = train_loader
        valid_loader = valid_loader
        test_loader = test_loader
        num_q = num_q
        num_r = num_r
        num_pid = num_pid
        num_diff = num_diff
    # 1-2. not use fivefold
    else:
        idx = 0
        train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff = get_loaders(config, idx)

    # 2. select models using get_models
    model = get_models(num_q, num_r, num_pid, num_diff, device, config)
    
    # 3. select optimizers using get_optimizers
    optimizer = get_optimizers(model, config)
    
    # 4. select crits using get_crits
    crit = get_crits(config)
    
    # 5. select trainers for models, using get_trainers
    trainer = get_trainers(model, optimizer, device, num_q, crit, config)

    # 6. use trainer.train to train the models
    # the result contain train_scores, valid_scores, hightest_valid_score, highest_test_score
    """train_scores, valid_scores, \
        highest_valid_score, highest_test_score  = trainer.train(train_loader, valid_loader, test_loader, config)"""
    #accountants = ["gdp", "prv", "rdp"]
    #results = {}
    train_scores, valid_scores, highest_valid_score, highest_test_score, epsilons= trainer.train(
        train_loader, valid_loader, test_loader, config
    )
    """for accountant in accountants:
        print(f"\nRunning with {accountant.upper()} accountant...")
        train_scores, valid_scores, highest_valid_score, highest_test_score, epsilons= trainer.train(
        train_loader, valid_loader, test_loader, config, accountant
    )
        results[accountant] = (epsilons)
    plt.figure(figsize=(12, 6))
    
    for accountant, epsilons in results.items():
       epochs = range(1, len(epsilons) + 1)
       plt.plot(epochs, epsilons, label=accountant.upper(), marker='o', markersize=4)
    plt.title("Privacy Budget (Epsilon) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Ensure y-axis starts from 0
    plt.ylim(bottom=0)
    
    # Find the maximum number of epochs across all accountants
    max_epochs = max(len(epsilons) for epsilons in results.values())
    
    # Adjust x-axis to show integer ticks
    plt.xticks(range(0, max_epochs + 1, max(1, max_epochs // 10)))
    
    plt.tight_layout()
    plt.savefig('accountant_comparison_epsilon.png', dpi=300, bbox_inches='tight')
    plt.close()"""
    
    # Ensure y-axis starts from 0
    
        
    # 7. model record
    # for model's name
    today = datetime.datetime.today()
    record_time = str(today.month) + "_" + str(today.day) + "_" + str(today.hour) + "_" + str(today.minute)
    # model's path
    model_path = '../model_records/' + str(highest_test_score) + "_" + record_time + "_" + config.model_fn
    # model save
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    return train_scores, valid_scores, highest_valid_score, highest_test_score, record_time
"""
def run_accountants(trainer, train_loader, valid_loader, test_loader, config):
        accountants = ["gdp", "prv", "rdp"]
        results = {}
        for accountant in accountants:
            print(f"\nRunning with {accountant.upper()} accountant...")
            epsilons, train_scores, valid_scores = trainer.train(
            train_loader, valid_loader, test_loader, config, accountant
        )
            results[accountant] = (epsilons, train_scores, valid_scores)
    
        return results
"""   
"""   
def plot_accountant_comparisons(results):
    plt.figure(figsize=(10, 6))
    
    for accountant, (epsilons, _, _) in results.items():
        epochs = range(1, len(epsilons) + 1)
    plt.plot(epochs, epsilons, label=accountant.upper())
    
    plt.title("Privacy Budget (Epsilon) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('monacobert_accountant_comparison.png')
    plt.close()
""" 
# If you used python train.py, then this will be start first
if __name__ == "__main__":
    # get config from define_argparser
    config = define_argparser() 

    # if fivefold = True
    if config.fivefold == True:

        test_scores_list = []
        
        for idx in range(5):
            train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff = get_loaders(config, idx)
            train_auc_scores, valid_auc_scores, \
                 best_valid_score, test_auc_score,  \
                    record_time = main(config, train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff)
            test_scores_list.append(test_auc_score)
        # mean the test_scores_list
        test_auc_score = sum(test_scores_list)/5
        # for record
        recorder(test_auc_score, record_time, config)
    # if fivefold = False 
    else:
        train_auc_scores, valid_auc_scores, \
             best_valid_score, test_auc_score, record_time = main(config)
        # for record
        recorder(test_auc_score, record_time, config)
        # for visualizer
        visualizer(train_auc_scores, valid_auc_scores, record_time)
    
