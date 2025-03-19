
import os
import argparse
import json
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from data_loaders.assist2009 import ASSIST2009
#from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.assist2012 import ASSIST2012
#from data_loaders.statics2011 import Statics2011
from models.dkt import DKT
#from models.dkt_plus import DKTPlus
#from models.dkvmn import DKVMN
#from models.sakt import SAKT
#from models.gkt import PAM, MHA
from models.utils import collate_fn
from opacus.layers import DPLSTM
from opacus import PrivacyEngine
import warnings
import matplotlib.pyplot as plt
from opacus.utils.batch_memory_manager import BatchMemoryManager

# Ignore all warnings
warnings.filterwarnings("ignore")





def main(model_name, dataset_name, max_gradient_norm, delta, epsilon, accountant):
    
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

    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2012":
        dataset = ASSIST2012(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
        #return dataset object
    elif dataset_name == "Statics2011":
        dataset = Statics2011(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    
    if model_name == "dkt":
        model = DKT(dataset.num_q, **model_config).to(device)
        #print(dataset.num_q)
    elif model_name == "dkt+":
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
        collate_fn=collate_fn, generator=torch.Generator(device='cuda:0')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn,generator=torch.Generator(device='cuda:0')
    )

    if optimizer == "sgd":
        optimizer = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = Adam(model.parameters(), learning_rate)
    nm = 6.0
    secure_mode = False
    privacy_engine = PrivacyEngine(secure_mode=secure_mode, accountant = accountant)
    model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier = nm,
    max_grad_norm= 0.5)
    
    
    aucs = []
    loss_means = []
    epochs = []
    epsilons_prv = []
    epsilons_rdp = []
    epsilons_gdp = []
    auc_prv = []
    auc_gdp = []
    auc_rdp = []
    max_auc = 0
    ckpt_path = os.path.join("ckpts", model_name, dataset_name)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        epochs.append(i)
        with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=16, 
        optimizer=optimizer
        ) as new_data_loader:
            for data in train_loader:
                q, r, qshft, rshft, m = data
                model.train()
                y = model(q.long(), r.long())
                y = (y * one_hot(qshft.long(), dataset.num_q)).sum(-1)
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)
                #opt.zero_grad()
                optimizer.zero_grad()
                loss = binary_cross_entropy(y, t)
                #print("Loss:",loss)
                loss.backward()
                optimizer.step()
                model.zero_grad()
            
                loss_mean.append(loss.detach().cpu().numpy())
                printstr = (
                f"\t Epoch {i}. Loss: {loss:.6f} "
                )
            
            if privacy_engine:
                epsilon = privacy_engine.get_epsilon(delta)
                printstr += f" | (ε = {epsilon:.2f}, δ = {delta})"
            if accountant == "prv":
                epsilons_prv.append(epsilon)
                
            if accountant== 'gdp':
                epsilons_gdp.append(epsilon)
            else:
                epsilons_rdp.append(epsilon)
            print(printstr)
        with open(os.path.join(ckpt_path, f"epsilons_{accountant}.pkl"), "wb") as f:
            pickle.dump(epsilons_gdp, f)
            
        
        with torch.no_grad():
            for data in test_loader:
                
                q, r, qshft, rshft, m = data

                model.eval()

                y = model(q.long(), r.long())
                y = (y * one_hot(qshft.long(), dataset.num_q)).sum(-1)

                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft, m).detach().cpu()

                auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
                )
                auc_gdp.append(auc)
                with open(os.path.join(ckpt_path, f"auc_{accountant}.pkl"), "wb") as f:
                    pickle.dump(auc_gdp, f)
                loss_mean = np.mean(loss_mean)
                print("TEST STEP")
                print("AUC: {}"
                        .format(auc)
                    )
                printstr = (f"\t Loss: {loss:.6f} "
                            )  
                if auc > max_auc:
                    torch.save(
                    model.state_dict(),
                    os.path.join(
                    ckpt_path, "model.ckpt"))
                    max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)
        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta)
            printstr += f" (ε = {epsilon:.2f}, δ = {delta})"
        print(printstr + "\n----------------------------\n")
        
       
    
        
        
    if accountant=='rdp':
        return epsilons_rdp, epochs
    if accountant == 'gdp':
        return epsilons_gdp, epochs
    else:
        return epsilons_prv, epochs
    

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
    parser.add_argument(
        "--max_gradient_norm",
        type=float,
        default=2.0,
        help="max gradient norm after clipping"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=8e-5,
        help="---"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=5.0,
        help="privacy budget"
    )
    args = parser.parse_args()
    
    accountant = 'gdp'
    gdp_eps, gdp_epochs = main(args.model_name, args.dataset_name, args.max_gradient_norm, args.delta, args.epsilon, accountant)
   
    
   