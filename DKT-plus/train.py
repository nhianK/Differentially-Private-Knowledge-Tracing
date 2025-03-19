import os
import argparse
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011

from models.dkt_plus import DKTPlus

from models.utils import collate_fn
from opacus import PrivacyEngine
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
import numpy as np


import warnings
warnings.filterwarnings("ignore")


def main(model_name, dataset_name):
    lambda_r = 0.01
    lambda_w1 = 0.003
    lambda_w2 = 3.0
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]

    
    if dataset_name == "Algebra2005":
        print("############")
        dataset = Algebra2005(seq_len)
    else:
        print("dataset not found")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    
    if model_name == "dkt+":
        model = DKTPlus(dataset.num_q, **model_config).to(device)
    
    else:
        print("The wrong model name was used...")
        return

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator(device='cuda:0')
    )

    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(
            os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
        ) as f:
            train_dataset.indices = pickle.load(f)
        with open(
            os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb"
        ) as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(
            os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb"
        ) as f:
            pickle.dump(train_dataset.indices, f)
        with open(
            os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
        ) as f:
            pickle.dump(test_dataset.indices, f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator = torch.Generator(device = 'cuda:0')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn,generator= torch.Generator(device = 'cuda:0')
    )

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)
    #privacu engine
    secure_mode = False
    max_norm = 1.5
    epsilon = 12.0
    delta = 0.9/len(train_dataset)
    privacy_engine = PrivacyEngine(secure_mode = secure_mode, accountant = "rdp")
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(module = model, optimizer = opt, data_loader = train_loader, max_grad_norm = max_norm, target_delta = delta, target_epsilon = epsilon, epochs = num_epochs)
    aucs = []
    loss_means = []

    max_auc = 0

    for i in range(1, num_epochs + 1):
        loss_mean = []

        for data in train_loader:
            q, r, qshft, rshft, m = data

            model.train()

            y = model(q.long(), r.long())
            y_curr = (y * one_hot(q.long(), dataset.num_q)).sum(-1)
            y_next = (y * one_hot(qshft.long(), dataset.num_q)).sum(-1)

            y_curr = torch.masked_select(y_curr, m)
            y_next = torch.masked_select(y_next, m)
            r = torch.masked_select(r, m)
            rshft = torch.masked_select(rshft, m)

            loss_w1 = torch.masked_select(
                    torch.norm(y[:, 1:] - y[:, :-1], p=1, dim=-1),
                    m[:, 1:]
                )
            loss_w2 = torch.masked_select(
                    (torch.norm(y[:, 1:] - y[:, :-1], p=2, dim=-1) ** 2),
                    m[:, 1:]
                )

            optimizer.zero_grad()
            loss = \
            binary_cross_entropy(y_next, rshft) + \
            lambda_r * binary_cross_entropy(y_curr, r) + \
            lambda_w1 * loss_w1.mean() / dataset.num_q + \
            lambda_w2 * loss_w2.mean() / dataset.num_q
            loss.backward()
           
            optimizer.step()
            
            loss_mean.append(loss.detach().cpu().numpy())
            
            printstr = f"\t Epoch {i} , Loss :{loss:.5f}"
        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta)
            printstr+=f"(ep = {epsilon:.2f}, delta = {delta})"
            print(printstr + "\n-------------------\n")

        with torch.no_grad():
            for data in test_loader:
                q, r, qshft, rshft, m = data

                model.eval()

                y = model(q.long(), r.long())
                y_next = (y * one_hot(qshft.long(), dataset.num_q)).sum(-1)

                y_next = torch.masked_select(y_next, m).detach().cpu()
                rshft = torch.masked_select(rshft, m).detach().cpu()

                auc = metrics.roc_auc_score(
                        y_true=rshft.numpy(), y_score=y_next.numpy()
                    )

                loss_mean = np.mean(loss_mean)
                print("----------------test step----------------")
                print(
                        "Epoch: {},   AUC: {},   Loss Mean: {}"
                        .format(i, auc, loss_mean)
                    )

                if auc > max_auc:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                    max_auc = auc

                aucs.append(auc)
                loss_means.append(loss_mean)
                
        
    
    
    
    
    

    """with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt, dkt+, dkvmn, sakt, gkt]. \
            The default model is dkt."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ASSIST2009",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [ASSIST2009, ASSIST2015, Algebra2005, Statics2011]. \
            The default dataset is ASSIST2009."
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name)
