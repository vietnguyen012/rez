import argparse
from genericpath import exists
import torch
import logging
import os 
from model import BertDot
import numpy as np 
import faiss 
from retrieve_utils import (
    construct_flatindex_from_embeddings,
    index_retrieve, convert_index_to_gpu
)
from tqdm import tqdm 
from collections import defaultdict
from inference import query_inference,doc_inference
logger = logging.Logger(__name__)
import json 
from dataset import load_rank,load_rel

def retrieve_top(args):
    model = BertDot(args)
    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)
    doc_inference(model, args, output_embedding_size)
    query_inference(model,args,output_embedding_size)
    model = None 
    torch.cuda.empty_cache()
    doc_embeddings = np.memmap(args.doc_memmap_path,dtype=np.float32,mode="r")
    doc_ids = np.memmap(args.docid_memmap_path,dtype=np.int32,mode="r")
    doc_embeddings = doc_embeddings.reshape(-1,output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path,dtype=np.float32,mode="r")
    query_embeddings = query_embeddings.reshape(-1,output_embedding_size)
    query_ids = np.memmap(args.queryid_memmap_path,dtype=np.int32,mode="r")

    index = construct_flatindex_from_embeddings(doc_embeddings,doc_ids)
    if torch.cuda.is_available() and not args.not_faiss_cuda:
        index = convert_index_to_gpu(index, list(range(args.n_gpu)), False)
    else:
        faiss.omp_set_num_threads(32)
    nearest_neighbors,_ = index_retrieve(index, query_embeddings, args.topk + 10,"cuda", batch=320)
    nearest_neighbors = nearest_neighbors.detach().cpu().numpy()
    with open(args.output_rank_file, 'w') as outputfile:
        for qid, neighbors in zip(query_ids, nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx+1}\n")

def gen_static_hardnegs(args):
    rank_dict = load_rank(args.output_rank_file)
    rel_dict = load_rel(args.label_path)
    query_ids_set = sorted(rel_dict.keys())
    for k in tqdm(query_ids_set,desc = "gen hard negs"):
        v = rank_dict[k]
        v = list(filter(lambda x:x not in rel_dict[k],v))
        v = v[:args.topk]
        assert len(v) == args.topk 
        rank_dict[k] = v 
    json.dump(rank_dict,open(args.output_hard_path,'w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--not_faiss_cuda", action="store_true")
    parser.add_argument("--use_mean", type=bool, default=True)
    args = parser.parse_args()

    args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.preprocess_dir = './preprocess'
    args.model_path = f'cross-encoder-reranking-model.pt'
    args.output_dir = f"./warmup_retrieve"
    args.label_path = "./preprocess/train-qrel.tsv"

    args.query_memmap_path = "./warmup_retrieve/query.memmap"
    args.queryid_memmap_path = "./preprocess/query_ids.memmap"
    args.doc_memmap_path = "./warmup_retrieve/passage.memmap"
    args.docid_memmap_path = "./preprocess/passage_ids.memmap"

    args.warmup_retrieve_queryid_memmap_path = "./warmup_retrieve/query_ids.memmap"
    args.warmup_retrieve_docid_memmap_path = "./warmup_retrieve/passage_ids.memmap"


    args.output_rank_file = "./warmup_retrieve/rank.tsv"
    args.output_hard_path = "./warmup_retrieve/hard.json"

    logger.info(args)
    os.makedirs(args.output_dir,exist_ok=True)
    retrieve_top(args)
    gen_static_hardnegs(args)