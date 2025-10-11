import os, sys
import tempfile
import subprocess
import json
import logging
import torch
import numpy as np
from transformers import BertModel
from itertools import zip_longest
import pandas as pd

import data_loader as dl
import featurization as ft
import utils


def get_embeddings(model_dir,seqs,seq_pair = None, layers = [-1], method="mean", device=0, batch_size = 256):

    device = utils.get_device(device)
    seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
    try:
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
    except OSError:
        logging.warning("Could not load saved tokenizer, loading fresh instance")
        tok = ft.get_aa_bert_tokenizer(64)
    model = BertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool")
    # print(model);exit()

    chunks = dl.chunkify(seqs, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = dl.chunkify(
            [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    embeddings = []
    with torch.no_grad():
        for seq_chunk in chunks_zipped:
            encoded = tok(
                *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
            )

            # manually calculated mask lengths
            # temp = [sum([len(p.split()) for p in pair]) + 3 for pair in zip(*seq_chunk)]
            input_mask = encoded["attention_mask"].numpy()
            encoded = {k: v for k, v in encoded.items()}

            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )
            if method == "pool":
                embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                continue
            # x.hidden_states contains hidden states, num_hidden_layers + 1 (e.g. 13)
            # Each hidden state is (batch, seq_len, hidden_size)
            # x.hidden_states[-1] == x.last_hidden_state
            # x.attentions contains attention, num_hidden_layers
            # Each attention is (batch, attn_heads, seq_len, seq_len)

            for i in range(len(seq_chunk[0])):
                e = []
                for l in layers:
                    # Select the l-th hidden layer for the i-th example
                    h = (
                        x.hidden_states[l][i].numpy().astype(np.float64)
                    )  # seq_len, hidden
                    # initial 'cls' token
                    if method == "cls":
                        e.append(h[0])
                        continue
                    # Consider rest of sequence
                    if seq_chunk[1] is None:
                        seq_len = len(seq_chunk[0][i].split())  # 'R K D E S' = 5
                    else:
                        seq_len = (
                            len(seq_chunk[0][i].split())
                            + len(seq_chunk[1][i].split())
                            + 1  # For the sep token
                        )
                    seq_hidden = h[1 : 1 + seq_len]  # seq_len * hidden
                    assert len(seq_hidden.shape) == 2

                    if method == "mean":
                        e.append(seq_hidden.mean(axis=0))
                    elif method == "max":
                        e.append(seq_hidden.max(axis=0))
                    elif method == "attn_mean":
                        # (attn_heads, seq_len, seq_len)
                        # columns past seq_len + 2 are all 0
                        # summation over last seq_len dim = 1 (as expected after softmax)
                        attn = x.attentions[l][i, :, :, : seq_len + 2]
                        # print(attn.shape)
                        print(attn.sum(axis=-1))
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Unrecognized method: {method}")
                e = np.hstack(e)
                assert len(e.shape) == 1
                embeddings.append(e)
    if len(embeddings[0].shape) == 1:
        embeddings = np.stack(embeddings)
    else:
        embeddings = np.vstack(embeddings)
    del x
    del model
    # torch.cuda.empty_cache()
    return embeddings


from argparse import ArgumentParser
 
def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../model/tcr-bert")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--data_path",  type=str, default=None)
    parser.add_argument("--save_path",  type=str, default=None)
 
    args = parser.parse_args()
    return args
 
 
def main():
    args = _get_args()
    model_dir = args.checkpoint
    batch_size = args.batch_size
    data_path = args.data_path
    save_path = args.save_path

    # cdr3 = ["CASSSRSSYEQYF","CASSPVTGGIYGYTF"]

    cdr3 = pd.read_csv(data_path,index_col=0)['tcr'].values
    
    emb = get_embeddings(model_dir,cdr3,batch_size=batch_size)
    print(emb.shape)

    if args.id != None:
        pd.DataFrame(emb).to_csv(save_path+f"donor_{args.id}_tcr_bert_emb.csv")
    else:
        pd.DataFrame(emb).to_csv(save_path+f"donor_all_tcr_bert_emb.csv")

if __name__ == '__main__':
    print("Start:")
    main()
    print("End!")
