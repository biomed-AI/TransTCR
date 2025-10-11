"""
training command line:
python ./Scripts/Pretrain/UniTCR_pretrain.py --config ./Configs/TrainingConfig_pretrain.yaml
"""

import scanpy as sc
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import torch
from torch import nn
import numpy as np
import math
import time
import random
import joblib
import pandas as pd
import anndata as ad
from get_data import InputDataSet
from torch.nn import functional as F
from Model_OT import TransTCR_model
import logging
import yaml
import argparse 

from val_epoch_Inter import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--config', type=str, help='the path to the config file (*.yaml)',required=True)
argparser.add_argument('--data_name', type=str,default=None)
argparser.add_argument('--TCR_model', type=str, default="Transformer")
argparser.add_argument('--RNA_model', type=str, default="MLP")
argparser.add_argument('--method', type=str, default="CLIP")
argparser.add_argument('--loss_num', type=int, default=1)
argparser.add_argument('--others', type=str, default="")

argparser.add_argument('--epoch', type=int, default=None)
argparser.add_argument('--batch_size', type=int, default=None)
argparser.add_argument('--lr', type=float, default=None)
argparser.add_argument('--reg_e', type=float, default=0.1)
argparser.add_argument('--ot_emb', type=bool, default=False)
argparser.add_argument('--max_iter', type=int, default=20)
argparser.add_argument('--seed', type=int, default=None)
argparser.add_argument('--cross', type=str, default=None)

argparser.add_argument('--tol', type=float, default=10e-9)
argparser.add_argument('--t', type=int, default=None)
argparser.add_argument('--version', type=int, default=1)
argparser.add_argument('--chain', type=int, default=1)
argparser.add_argument('--project', type=bool, default=False)

args = argparser.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# load the cofig file
config_file = args.config
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_file)

if args.lr != None:
    config['Train']['Trainer_parameter']['learning_rate'] = args.lr
if args.epoch != None:
    config['Train']['Trainer_parameter']['epoch'] = args.epoch
if args.batch_size != None:
    config['Train']['Sampling']['batch_size'] = args.batch_size
if args.seed != None:
    config['Train']['Trainer_parameter']['random_seed'] = args.seed

method = f"{args.data_name}\t{args.RNA_model}\t{args.TCR_model}\t{args.method}\tloss{args.loss_num}({args.others})"
config['Train']['output_dir'] = config['Train']['output_dir']+method

def make_result_csv(name):
    columns = ["data_name", "MLP_ACC", "MLP_F1", "KNN_ACC", "KNN_F1", "NMI", "ARI", "SCIB_NMI", "SCIB_ARI", "SCIB_ASW", "SCIB_avg_bio"]
    data_names = ["donor_1", "donor_2", "donor_3", "donor_4", "avg"]
    
    data = {
        "data_name": data_names,
        "MLP_ACC": [None] * 5,
        "MLP_F1": [None] * 5,
        "KNN_ACC": [None] * 5,
        "KNN_F1": [None] * 5,
        "NMI": [None] * 5,
        "ARI": [None] * 5,
        "SCIB_NMI": [None] * 5,
        "SCIB_ARI": [None] * 5,
        "SCIB_ASW": [None] * 5,
        "SCIB_avg_bio": [None] * 5
    }
    pd.DataFrame(data, columns=columns).to_csv(f"{name}.csv", index=False)
# create directory if it not exist
if not os.path.exists(config['Train']['output_dir']):
    os.makedirs(config['Train']['output_dir'])
    os.makedirs(config['Train']['output_dir']+'/Model_checkpoints')     
    os.makedirs(config['Train']['output_dir']+'/loss')
    os.makedirs(config['Train']['output_dir']+'/result') 
    os.makedirs(config['Train']['output_dir']+'/Embedding_Result')
    make_result_csv(config['Train']['output_dir']+'/result/RNA_result')
    make_result_csv(config['Train']['output_dir']+'/result/TCR_result')
else:
    print("The fold is exited!")

# random seed setting
torch.manual_seed(config['Train']['Trainer_parameter']['random_seed'])
torch.cuda.manual_seed_all(config['Train']['Trainer_parameter']['random_seed'])
np.random.seed(config['Train']['Trainer_parameter']['random_seed'])
random.seed(config['Train']['Trainer_parameter']['random_seed'])
torch.cuda.manual_seed(config['Train']['Trainer_parameter']['random_seed'])
device = "cuda"

# initialize a new DalleTCR model
TransTCR = TransTCR_model(encoderTCR_in_dim = config['Train']['Model_Parameter']['encoderTCR_in_dim'],
                              encoderTCR_out_dim = config['Train']['Model_Parameter']['encoderTCR_out_dim'],
                              encoderprofile_in_dim = config['Train']['Model_Parameter']['encoderprofile_in_dim'], 
                              encoderprofile_hid_dim = config['Train']['Model_Parameter']['encoderprofile_hid_dim'], 
                              encoderprofile_hid_dim2 = config['Train']['Model_Parameter']['encoderprofile_hid_dim2'],
                              encoderprofle_out_dim = config['Train']['Model_Parameter']['encoderprofle_out_dim'],
                              out_dim = config['Train']['Model_Parameter']['out_dim'],
                              head_num = config['Train']['Model_Parameter']['head_num'],
                              device=device,
                              TCR_model=args.TCR_model,
                              RNA_model=args.RNA_model,
                              method = args.method,
                              loss_num = args.loss_num,
                              max_iter = args.max_iter,
                              tol = args.tol,
                              chain = args.chain,
                              version = args.version,
                              project = args.project,
                              ).to(device)
# print(TransTCR);exit()
# setting the learning rate for the model
TransTCR.optimizer = torch.optim.AdamW(TransTCR.parameters(), lr = config['Train']['Trainer_parameter']['learning_rate'])

dataset = InputDataSet(config['dataset']['Training_dataset'],args.data_name,args.RNA_model,args.TCR_model,args.loss_num,data_class="Inter",chain=args.chain)

if args.loss_num == 1:
    pass
elif args.loss_num == 3:
    data_dist_tcr = sc.read_h5ad(config['dataset']['Training_tcr_dist'])
    data_dist_tcr = torch.Tensor(data_dist_tcr.X)
    data_dist = sc.read_h5ad(config['dataset']['Training_profile_dist'])
    data_dist = torch.Tensor(data_dist.X)
else:
    print("Loss_num is Error!");exit()

# initialize the dataloader
dataloader1 = torch.utils.data.DataLoader(dataset, batch_size = config['Train']['Sampling']['batch_size'], shuffle = config['Train']['Sampling']['sample_shuffle'])   
dataloader2 = torch.utils.data.DataLoader(dataset, batch_size = config['Train']['Sampling']['batch_size'], shuffle = False)


# initialize the logger for saving the traininig log
logger = get_logger(config['Train']['output_dir']+'/Inter_training.log',verbosity=1, name=args.data_name)
logger.info(method)
logger.info(pd.Timestamp.now())
logger.info(str(args)+"\n") 

# setting the training epoch
epochs = config['Train']['Trainer_parameter']['epoch']
original_data = sc.read_h5ad(config['dataset']['Training_dataset'])


loss_ptc_list = []
loss_p2p_list = []
loss_t2t_list = []
loss_task1_list = []
strat_time = time.time()

# training the UniTCR
flag = True
t = 0
loss_task_min = -1
for epoch in range(1, epochs + 1):
    loss_ptc_epoch = 0
    loss_ptm_epoch = 0
    loss_p2p_epoch = 0
    loss_t2t_epoch = 0
    loss_task1 = 0
    s = time.time()
    TransTCR.train()
    for idx, b in enumerate(dataloader1): 
        if args.loss_num == 1: 
            similarity_dist_profile = torch.tensor(0)
            similarity_dist_tcr = torch.tensor(0)
        if args.loss_num == 3: 
            similarity_dist_profile = data_dist[b[3]][:,b[3]]
            similarity_dist_tcr = data_dist_tcr[b[3]][:,b[3]]
        loss_ptc, loss_p2p, loss_t2t = TransTCR([b[0].to(device), 
                                            b[1].to(torch.float32).to(device)], 
                                            b[2].to(device), 
                                            similarity_dist_profile.to(device), 
                                            similarity_dist_tcr.to(device),
                                            task_level = 1,
                                            TCR_emb = b[5].to(device),
                                            RNA_emb = b[6].to(device),
                                            reg_e = args.reg_e,
                                            cross = args.cross)
        # loss_b = 0.1 * loss_ptc + 0.8 * loss_p2p + 0.1 * loss_t2t
        # loss_b = 0.5 * loss_ptc + 0.5 * loss_t2t
        if args.loss_num == 1: 
            loss_b = loss_ptc
        if args.loss_num == 3: 
            # loss_b = 0.1 * loss_ptc + 0.8 * loss_p2p + 0.1 * loss_t2t
            loss_b = loss_ptc + loss_p2p + loss_t2t
        TransTCR.optimizer.zero_grad()
        loss_b.backward()
        TransTCR.optimizer.step()
        loss_ptc_epoch += loss_ptc.item()
        if args.loss_num == 1:
            loss_p2p_epoch += 0
            loss_t2t_epoch += 0
        elif args.loss_num == 3: 
            loss_p2p_epoch += loss_p2p.item()
            loss_t2t_epoch += loss_t2t.item()

        
        loss_task1 += loss_b.item()

    loss_ptc_list.append(loss_ptc_epoch)
    loss_p2p_list.append(loss_p2p_epoch)
    loss_t2t_list.append(loss_t2t_epoch)
    loss_task1_list.append(loss_task1)
    

    e = time.time()
    loss_ptc_epoch /= (idx + 1)
    loss_p2p_epoch /= (idx + 1)
    loss_t2t_epoch /= (idx + 1)
    logger.info('Epoch:[{}/{}]\tsteps:{}\tptc_loss:{:.5f}\tp2p_loss:{:.5f}\tt2t_loss:{:.5f}\ttime:{:.3f}'.format(epoch, epochs, idx + 1, loss_ptc_epoch, loss_p2p_epoch, loss_t2t_epoch, e-s))

    # checkpoint saving
    if args.t != None:
        if epoch%10==0 or epoch>epochs-50:
            if loss_task_min<0:
                loss_task_min = loss_task1
            elif loss_task_min<=loss_task1:
                t=t+1
                logger.info(f"Now tolerance is {t}")
                if t>=args.t:
                    break
            elif loss_task_min>loss_task1:
                loss_task_min = loss_task1
                t = 0
                torch.save({'epoch':epoch, 
                            'model_state_dict':TransTCR.state_dict(), 
                            'optimizer_state_dict': TransTCR.optimizer.state_dict(), 
                            'loss': loss_task1 / (idx+1)}, 
                        f"{config['Train']['output_dir']}/Model_checkpoints/best_model.pth")
                logger.info(f"Save model to {config['Train']['output_dir']}/Model_checkpoints/best_model.pth")

    if epoch % 100 == 0 or epoch % epochs == 0:
        torch.save({'epoch':epoch, 
                    'model_state_dict':TransTCR.state_dict(), 
                    'optimizer_state_dict': TransTCR.optimizer.state_dict(), 
                    'loss': loss_task1 / (idx+1)}, 
                f"{config['Train']['output_dir']}/Model_checkpoints/{epoch}_model.pth")
        logger.info(f"Save model to {config['Train']['output_dir']}/Model_checkpoints/{epoch}_model.pth")

logger.info(f'finish training! All time {time.time()-strat_time}s!')

import matplotlib.pyplot as plt

epochs = range(1, len(loss_ptc_list) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_ptc_list, label='PTC Loss', marker='o')
plt.plot(epochs, loss_p2p_list, label='P2P Loss', marker='s')
plt.plot(epochs, loss_t2t_list, label='T2T Loss', marker='^')
plt.plot(epochs, loss_task1_list, label='Total Loss', linestyle='--', color='black')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'loss')
plt.legend()
plt.grid(True)
plt.savefig(f"{config['Train']['output_dir']}/loss/loss_{args.data_name}_Inter.png")
plt.show()


checkpoint = torch.load(f"{config['Train']['output_dir']}/Model_checkpoints/best_model.pth")
TransTCR.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
    TCR_embedding = []
    Profile_embedding = []
    Gaps = []
    for idx, b in enumerate(dataloader2):
        encoderTCR_embedding = TransTCR.get_TCR_emb(b[0].to(device),b[5].to(device))
        encoderprofile_embedding = TransTCR.get_RNA_emb(b[1].to(torch.float32).to(device),b[2].to(device),b[6].to(torch.float32).to(device))

        if args.ot_emb:
            encoderprofile_embedding,encoderTCR_embedding = TransTCR.get_emb_OT(encoderprofile_embedding,encoderTCR_embedding)
        
        TCR_embedding.append(encoderTCR_embedding.cpu())
        Profile_embedding.append(encoderprofile_embedding.cpu())
        tmp_gap = 2 * (1 - (encoderTCR_embedding @ encoderprofile_embedding.T).diag())

        Gaps.append(tmp_gap.cpu())
        
    TCR_embedding = torch.cat(TCR_embedding, dim = 0)
    Profile_embedding = torch.cat(Profile_embedding, dim = 0)
    Gaps = torch.cat(Gaps, dim = 0)

    TCR_adata = ad.AnnData(TCR_embedding.numpy())
    TCR_adata.obs = original_data.obs

    RNA_adata = ad.AnnData(Profile_embedding.numpy())
    RNA_adata.obs = original_data.obs

    RNA_avg,TCR_avg = run_val_epoch_Inter(config,method,RNA_adata,TCR_adata,logger,is_scib=True)
    logger.info(f"RNA_Avg:{RNA_avg}")
    logger.info(f"TCR_Avg:{TCR_avg}")
    

epochs = epoch
torch.save({'epoch':epochs, 
            'model_state_dict':TransTCR.state_dict(), 
            'optimizer_state_dict': TransTCR.optimizer.state_dict(), 
            'loss': loss_task1 / (idx+1)}, 
        f"{config['Train']['output_dir']}/Model_checkpoints/{args.data_name}_epoch{epochs}_model.pth")    

TCR_adata.write(f"{config['Train']['output_dir']}/Embedding_Result/{args.data_name}_TCR_embedding.h5ad")
RNA_adata.write(f"{config['Train']['output_dir']}/Embedding_Result/{args.data_name}_Profile_embedding.h5ad")

adata_gaps = ad.AnnData(Gaps.unsqueeze(1).numpy())
adata_gaps.obs = original_data.obs
adata_gaps.write(f"{config['Train']['output_dir']}/Embedding_Result/{args.data_name}_Gaps.h5ad")

# RNA_avg,TCR_avg = run_val_epoch_Inter(config,method,RNA_adata,TCR_adata,logger,is_scib=True)
logger.info("="*100)
logger.info(f"The {epochs} result:")
logger.info(f"RNA_Avg:{RNA_avg}")
logger.info(f"TCR_Avg:{TCR_avg}")

# all_result = pd.read_csv("./result/Inter/all_result.csv")
# all_result.loc[len(all_result)] = RNA_avg
# all_result.loc[len(all_result)] = TCR_avg
# all_result.to_csv("./result/Inter/all_result.csv",index=None)

logger.info(pd.Timestamp.now())