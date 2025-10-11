import pandas as pd
import scanpy as sc
import numpy as np
import yaml
import argparse 
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")

def get_f1(adata):
    train_idx = adata.obs[adata.obs['set'] == 'train'].index
    test_idx = adata.obs[adata.obs['set'] == 'val'].index

    X_train = adata[train_idx].X  
    X_test = adata[test_idx].X 
    y_train = adata[train_idx].obs['binding_name']
    y_test = adata[test_idx].obs['binding_name']

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    # knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    # print("Classification Report:\n")
    # for k,v in report.items():
    #     print(k,v)

    print("ACC",report["accuracy"])
    print("F1-score",report["weighted avg"]["f1-score"])
    # print("mF1-score",report["macro avg"]["f1-score"])

    return report["accuracy"],report["weighted avg"]["f1-score"]

def get_NMI(adata):
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.louvain(adata,resolution=1.0)
    true_labels = adata.obs['binding_name']
    pred_labels = adata.obs['louvain']
    print(f"True {len(np.unique(true_labels))}")
    print(f"Pred {len(np.unique(pred_labels))}")

    ari_score = adjusted_rand_score(true_labels, pred_labels)
    nmi_score = normalized_mutual_info_score(true_labels, pred_labels)

    print(f"ARI: {ari_score}")
    print(f"NMI: {nmi_score}")

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['binding_name', 'louvain'])

    return ari_score,nmi_score
    
def get_MLP_f1(adata):
    train_idx = adata.obs[adata.obs['set'] == 'train'].index
    test_idx = adata.obs[adata.obs['set'] == 'val'].index

    # 获取训练集和测试集的特征和标签
    X_train = adata[train_idx].X  
    X_test = adata[test_idx].X 
    y_train = adata[train_idx].obs['binding_name']
    y_test = adata[test_idx].obs['binding_name']
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    # 使用MLPClassifier替换KNeighborsClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, activation='relu', solver='adam', random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # 生成分类报告
    report = classification_report(y_test, y_pred, output_dict=True)

    # 打印准确率和F1分数
    print("MLP Classifier")
    print("ACC", report["accuracy"])
    print("F1-score", report["weighted avg"]["f1-score"])
    # print("mF1-score", report["macro avg"]["f1-score"])  # 如果需要宏平均F1分数，可以取消注释

    return report["accuracy"],report["weighted avg"]["f1-score"]

def get_scib(
    adata,
    batch_key="donor",
    label_key="binding_name",
    embed_key="X_obsm"
):
    import scib

    adata.obsm["X_obsm"] = adata.X
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed=embed_key,
        isolated_labels_asw_=False,
        silhouette_=True,
        # silhouette_=False,
        hvg_score_=False,
        # graph_conn_=True,
        graph_conn_=False,
        # pcr_=True,
        pcr_=False,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    result_dict = results[0].to_dict()
    
    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}
    print("NMI:",result_dict['NMI_cluster/label'])
    print("ARI:",result_dict['ARI_cluster/label'])
    print("Avg:",result_dict['avg_bio'])

    return result_dict['NMI_cluster/label'],result_dict['ARI_cluster/label'],result_dict['ASW_label'],result_dict['avg_bio']


def run_val_epoch_Intra(data_name,RNA_adata,TCR_adata,is_scib=False):
    # result_path = config['Train']['output_dir']
    # config['Train']['output_dir'] = config['Train']['output_dir']+f"/{method}"

    def run(adata,key):
        print("$"*50,key,"$"*50)
        dics = []
        for fold in range(5):
            print("*"*30,f"{data_name}_fold{fold}","*"*30)
            dic = {"data_name":f"{data_name}_fold{fold}"}
            with open(f"../{data_name}_indices.pkl", 'rb') as f:
                fold_dict = pickle.load(f)

            train_index = fold_dict['train_indices'][fold]
            test_index = fold_dict['test_indices'][fold]
            train_index = [adata.obs.index[k] for k in train_index]
            test_index  = [adata.obs.index[k] for k in test_index]

            adata.obs['set'] = "None"
            adata.obs.loc[train_index, 'set'] = "train"
            adata.obs.loc[test_index, 'set'] = "val"
            print("train/test:",len(train_index),len(test_index))

            mlp_acc,mlp_f1 = get_MLP_f1(adata)
            dic["MLP_ACC"] = mlp_acc
            dic["MLP_F1"]  = mlp_f1

            knn_acc,knn_f1 = get_f1(adata)
            dic["KNN_ACC"] = knn_acc
            dic["KNN_F1"]  = knn_f1

            if len(dics)==0:
                ari,nmi = get_NMI(adata)
                dic["ARI"] = ari
                dic["NMI"] = nmi
                if is_scib:
                    dic["SCIB_NMI"],dic["SCIB_ARI"],dic["SCIB_ASW"],dic["SCIB_avg_bio"] = get_scib(adata)
            else:
                dic["ARI"],dic["NMI"] = dics[0]["ARI"],dics[0]["NMI"]
                if is_scib:
                    dic["SCIB_NMI"],dic["SCIB_ARI"],dic["SCIB_ASW"],dic["SCIB_avg_bio"] = dics[0]["SCIB_NMI"],dics[0]["SCIB_ARI"],dics[0]["SCIB_ASW"],dics[0]["SCIB_avg_bio"]
            # print(dic)
            dics.append(dic)
        # print(dics)
        return dics
    RNA_dics = run(RNA_adata,"RNA")
    TCR_dics = run(TCR_adata,"TCR")

    return RNA_dics,TCR_dics

    # print(result_df)
    # result_df.to_csv(config['Train']['output_dir']+f"/result/result_epoch{epoch}.csv")
def run_val_epoch_Intra42(data_name,RNA_adata,TCR_adata,is_scib=False):
    def run(adata,key):
        print("*"*30,f"{data_name}_{key}","*"*30)
        dic = {"data_name":f"{data_name}"}

        print(adata.obs["set"].value_counts())

        mlp_acc,mlp_f1 = get_MLP_f1(adata)
        dic["MLP_ACC"] = mlp_acc
        dic["MLP_F1"]  = mlp_f1

        knn_acc,knn_f1 = get_f1(adata)
        dic["KNN_ACC"] = knn_acc
        dic["KNN_F1"]  = knn_f1

        if is_scib:
            dic["SCIB_NMI"],dic["SCIB_ARI"],dic["SCIB_ASW"],dic["SCIB_avg_bio"] = get_scib(adata)
        return dic
    RNA_dic = run(RNA_adata,"RNA")
    TCR_dic = run(TCR_adata,"TCR")

    return RNA_dic,TCR_dic
# if __name__=="__main__":
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--config', type=str, help='the path to the config file (*.yaml)',required=True)
#     argparser.add_argument('--data_name', type=str,default="donor_all")
#     argparser.add_argument('--TCR_model', type=str, default="Bert")
#     argparser.add_argument('--RNA_model', type=str, default="MLP")
#     argparser.add_argument('--method', type=str, default="sinkhorn")
#     argparser.add_argument('--loss_num', type=int, default=1)
#     argparser.add_argument('--others', type=str, default="")
#     args = argparser.parse_args()

#     # load the cofig file
#     config_file = args.config
#     def load_config(config_file):
#         with open(config_file) as file:
#             config = yaml.safe_load(file)
#         return config

#     config = load_config(config_file)
#     method = f"{args.data_name}_{args.RNA_model}_{args.TCR_model}_{args.method}_{args.loss_num}({args.others})"
#     epoch = config['Train']['Trainer_parameter']['epoch']

#     path = f"{config['Train']['output_dir']}/Embedding_Result/"
#     RNA_adata = sc.read_h5ad(path+"Profile_embedding.h5ad")
#     TCR_adata = sc.read_h5ad(path+"TCR_embedding.h5ad")

#     run_val_epoch_Intra(epoch,config,method,RNA_adata,TCR_adata)

