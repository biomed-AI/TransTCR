import pandas as pd
import scanpy as sc
import numpy as np
import yaml
import argparse 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")

def get_f1(adata,test_label):
    train_idx = adata.obs[adata.obs['donor'] != test_label].index
    test_idx = adata.obs[adata.obs['donor'] == test_label].index

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
    print(f"GroundTrue labels {len(np.unique(true_labels))}")
    print(f"Cluster labels {len(np.unique(pred_labels))}")

    ari_score = adjusted_rand_score(true_labels, pred_labels)
    nmi_score = normalized_mutual_info_score(true_labels, pred_labels)

    print(f"ARI: {ari_score}")
    print(f"NMI: {nmi_score}")

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['binding_name', 'louvain'])

    return ari_score,nmi_score

def get_MLP_f1(adata,test_label):
    train_idx = adata.obs[adata.obs['donor'] != test_label].index
    test_idx = adata.obs[adata.obs['donor'] == test_label].index

    # 获取训练集和测试集的特征和标签
    X_train = adata[train_idx].X  
    X_test = adata[test_idx].X 
    y_train = adata[train_idx].obs['binding_name']
    y_test = adata[test_idx].obs['binding_name']
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    # 使用MLPClassifier替换KNeighborsClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=100, activation='relu', solver='adam', random_state=42)
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
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
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

def run_val_epoch_Inter(config,method,RNA_adata,TCR_adata,logger,is_scib=False):
    # result_path = config['Train']['output_dir']
    # config['Train']['output_dir'] = config['Train']['output_dir']+f"/{method}"

    def run(adata,key):
        print("$"*50,key,"$"*50)
        batchs = ["donor_1","donor_2","donor_3","donor_4"]
        df = pd.read_csv(config['Train']['output_dir']+f'/result/{key}_result.csv',index_col=0)
        for id in range(4):
            print("="*30,batchs[id],"="*30)

            mlp_acc,mlp_f1 = get_MLP_f1(adata,batchs[id])
            df.loc[batchs[id],"MLP_ACC"] = mlp_acc
            df.loc[batchs[id],"MLP_F1"]  = mlp_f1

            knn_acc,knn_f1 = get_f1(adata,batchs[id])
            df.loc[batchs[id],"KNN_ACC"] = knn_acc
            df.loc[batchs[id],"KNN_F1"]  = knn_f1

        ari,nmi = get_NMI(adata)
        df.loc[batchs[id],"ARI"] = ari
        df.loc[batchs[id],"NMI"] = nmi
        
        if is_scib:
            scib_nmi,scib_ari,scib_asw,scib_avg_bio = get_scib(adata)
            df.loc[batchs[id],"SCIB_NMI"] = scib_nmi
            df.loc[batchs[id],"SCIB_ARI"] = scib_ari
            df.loc[batchs[id],"SCIB_ASW"] = scib_asw
            df.loc[batchs[id],"SCIB_avg_bio"] = scib_avg_bio

        df.loc["avg"] = df.loc[batchs,:].mean(axis=0)
        # print(key,df)
        logger.info(f"{key}_All:\n{str(df)}")
        df.to_csv(config['Train']['output_dir']+f'/result/{key}_result.csv')
        new_dic = df.loc["avg"].to_dict()
        new_dic["method"] = f"{method}_avg{(key)}"
        return new_dic

    print("*"*100,method,"*"*100)
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

#     run_val_epoch_Inter(epoch,config,method,RNA_adata,TCR_adata)

