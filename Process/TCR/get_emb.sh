echo Get embeddings by TCR-BERT 

id=2

python get_emb.py --id $id \
    --data_path ~/project2/TCR-BERT/data/donor_$id.csv \
    --save_path ~/project2/TCR-BERT/emb/ 



# python get_emb.py \
#     --data_path ~/project2/TCR-BERT/data/donor_all.csv \
#     --save_path ~/project2/TCR-BERT/emb/ 