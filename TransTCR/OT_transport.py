import ot
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA


def optimal_transport(Xs,Xt,reg_e=1e-1, reg_cl=None,ys=None,method="lpl1_reg",max_iter=20,tol=10e-9,batch_size=128,version = 1):
   
    if method == "emd":
        ot_emd = ot.da.EMDTransport()
        ot_emd.fit(Xs=Xs, Xt=Xt)
        if version == 2:
            return ot_emd.coupling_
        transp_Xs_emd = ot_emd.transform(Xs=Xs)        
        return transp_Xs_emd

    elif method == "sinkhorn":
        ot_sinkhorn = ot.da.SinkhornTransport(reg_e=reg_e,max_iter=max_iter,tol=tol) # 这里为了加快收敛max_iter=10
        ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
        if version == 2:
            return Xs.shape[0]*ot_sinkhorn.coupling_
        if version == 4:
            return Xs.shape[0]*ot_sinkhorn.coupling_.T@Xs
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs,batch_size=batch_size)
        return transp_Xs_sinkhorn

    elif method == "lpl1_reg":
        ot_lpl1 = ot.da.SinkhornLpl1Transport(reg_e=reg_e, reg_cl=reg_cl)
        ot_lpl1.fit(Xs=Xs, ys=ys, Xt=Xt)
        transp_Xs_lpl1 = ot_lpl1.transform(Xs=Xs)
        # check_mapping_quality(ot_lpl1, transp_Xs_lpl1)
        return transp_Xs_lpl1

    elif method == "emd_laplace":
        ot_emd_laplace = ot.da.EMDLaplaceTransport(reg_lap=10, reg_src=10, tol=tol) # 这里为了加快收敛max_iter=20 
        ot_emd_laplace.fit(Xs=Xs, Xt=Xt)
        transp_Xs_emd_laplace = ot_emd_laplace.transform(Xs=Xs)
        return transp_Xs_emd_laplace
        
    elif method == "l1l2_reg":
        ot_l1l2 = ot.da.SinkhornL1l2Transport(reg_e=1e-1, reg_cl=2e0, max_iter=max_iter,
                                      verbose=True)
        ot_l1l2.fit(Xs=Xs, ys=ys, Xt=Xt)
        transp_Xs_l1l2 = ot_l1l2.transform(Xs=Xs)
        return transp_Xs_l1l2
    
