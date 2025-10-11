import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import joblib
import scanpy as sc
import time
from scipy.sparse import issparse

class InputDataSet(Dataset):
    
    def __init__(self, data_dir, id=None, RNA_model=None, TCR_model=None, loss_num=None, data_class="Intra", chain=1):
        if data_class=="Inter" and id=="donor_all":
            id="all"
        elif data_class!="Intra":
            print("Processing Error!")
            exit()

        data_path = f"../dataset/"

        # obtain the profile data and Tcell information
        data = sc.read_h5ad(data_dir)
        
        # obtain the single cell profile data
        # self.profile = data.layers['scale_data'].toarray()
        self.profile = data.X.toarray() if issparse(data.X) else data.X
        
        # obtain the profile gene name
        self.gene_names = list(data.var.index)

        self.loss_num = loss_num
        self.chain = chain

        # obtain TCR embeddings
        if TCR_model=="Transformer":
            self.TCR_emb = [0]*data.X.shape[0]
        elif TCR_model=="ESM-2":
            if id=="all":
                self.TCR_emb = np.array(pd.read_csv(f"{data_path}/donor_all/donor_all_esm_emb.csv",index_col=0),dtype=np.float32)
            else:
                self.TCR_emb = np.array(pd.read_csv(f"{data_path}/donor{id}/donor{id}_esm_emb.csv",index_col=0),dtype=np.float32)
        elif TCR_model=="TCR-Bert":
            if id=="all":
                self.TCR_emb = np.array(pd.read_csv(f"{data_path}/donor_all/donor_all_TRB_emb.csv",index_col=0),dtype=np.float32)
            else:
                self.TCR_emb = np.array(pd.read_csv(f"{data_path}/donor{id}/donor{id}_TRB_emb.csv",index_col=0),dtype=np.float32)
            if self.chain==2:
                if id=="all":
                    self.TRA_emb = np.array(pd.read_csv(f"{data_path}/donor_all/donor_all_TRA_emb.csv",index_col=0),dtype=np.float32)
                else:
                    self.TRA_emb = np.array(pd.read_csv(f"{data_path}/donor{id}/donor{id}_TRA_emb.csv",index_col=0),dtype=np.float32)
                self.TCR_emb = np.concatenate([self.TRA_emb, self.TCR_emb], axis=1)
        else:
            print("TCR_Model is Error!");exit()
        # print(self.TCR_emb.shape,self.TRA_emb.shape);exit()

        # obtain RNA embeddings
        if RNA_model=="MLP" or RNA_model=="GCN" or RNA_model=="Self-attention":
            self.RNA_emb = [0]*data.X.shape[0]
        elif RNA_model=="scFoundation" or RNA_model=="UCE":
            if id=="all":
                self.RNA_emb = np.load(f"{data_path}/donor_all/donor_all_{RNA_model}_emb.npy")
            else:
                self.RNA_emb = np.load(f"{data_path}/donor{id}/donor{id}_{RNA_model}_emb.npy")
        elif RNA_model=="CellFM":
            if id=="all":
                self.RNA_emb = sc.read_h5ad(f"{data_path}/donor_all/donor_all_{RNA_model}_emb.h5ad").X
            else:
                self.RNA_emb = sc.read_h5ad(f"{data_path}/donor{id}/donor{id}_{RNA_model}_emb.h5ad").X
        else:
            print("RNA_Model is Error!");exit()

        
        # obtain the T cell information
        self.beta_chains = list(data.obs['beta'])
        self.bindings = list(data.obs['binding_name'])
             
        # create the position encoding
        self.position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
        self.position_encoding[:, 0::2] = np.sin(self.position_encoding[:, 0::2])
        self.position_encoding[:, 1::2] = np.cos(self.position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(self.position_encoding)
        self.aa_dict = joblib.load("./dic_Atchley_factors.pkl")
        
        # create the TCR index dict
        # beta
        n = 0
        self.TCR_ids = {}
        for idx, beta in enumerate(self.beta_chains):
            if beta not in self.TCR_ids:
                self.TCR_ids[beta] = n
                n += 1

    def aamapping(self,TCRSeq,encode_dim):
        #the longest sting will probably be shorter than 80 nt
        TCRArray = []
        if len(TCRSeq)>encode_dim:
            # print('Length: '+str(len(TCRSeq))+' over bound!')
            TCRSeq=TCRSeq[0:encode_dim]
        for aa_single in TCRSeq:
            try:
                TCRArray.append(self.aa_dict[aa_single])
            except KeyError:
                # print('Not proper aaSeqs: '+TCRSeq)  
                TCRArray.append(np.zeros(5,dtype='float64'))
        for i in range(0,encode_dim-len(TCRSeq)):
            TCRArray.append(np.zeros(5,dtype='float64'))
        return torch.FloatTensor(np.array(TCRArray)) 

    def add_position_encoding(self,seq):
        mask = (seq == 0).all(dim = 1)
        seq[~mask] += self.position_encoding[:seq[~mask].size()[-2]]
        return seq 
    
    def __getitem__(self, item):
        batch_profile = self.profile[item]
        TCR_beta = self.add_position_encoding(self.aamapping(self.beta_chains[item], 25))

        return TCR_beta, batch_profile, self.TCR_ids[self.beta_chains[item]], item, self.bindings[item], self.TCR_emb[item], self.RNA_emb[item]
        
    def __len__(self, ):
        
        # return the cell number of the dataset
        return self.profile.shape[0]
