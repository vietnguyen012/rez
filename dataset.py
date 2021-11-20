import torch 
import json 
import numpy as np 
from torch.utils.data import Dataset
from typing import List, Sequence
from transformers import AutoTokenizer
import os 
from collections import defaultdict
from tqdm import tqdm
import random 
def load_rank(rank_path):
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        qid, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)

def load_rel(rel_path):
    reldict = defaultdict(list)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        ids = line.split()
        qid = int(ids[0])
        p_id = [int(id) for id in ids[2:-1]]
        reldict[qid].extend(p_id)
    return dict(reldict)

def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor

class TextTokenIdsCache:
    def __init__(self,data_dir,prefix):
        meta = json.load(open(data_dir+"/"+prefix+"_meta"))
        self.total_number = meta["total_number"]
        self.max_seq_length = meta["embedding_size"]
        self.ids_arr = np.memmap(f"{data_dir}/{prefix}_token_ids.memmap", 
                shape=(self.total_number, self.max_seq_length), 
                dtype=np.dtype(meta['type']), mode="r")
        self.lengths_arr = np.memmap(f"{data_dir}/{prefix}_lengths.memmap", \
            shape=(self.total_number, ), mode='r+', dtype=np.dtype(meta['type']))
    def __len__(self):
        return self.total_number
    
    def __getitem__(self, item):
        return self.ids_arr[item, :self.lengths_arr[item]]

class SequenceDataset(Dataset):
    def __init__(self,args, ids_cache, max_seq_length,is_query=False):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length
        if is_query:
            self.ids_key = np.memmap(args.queryid_memmap_path, \
                shape=(len(self.ids_cache),), mode ='r+', dtype=np.int32)
        else:
            self.ids_key = np.memmap(args.docid_memmap_path, \
                shape=(len(self.ids_cache),), mode ='r+', dtype=np.int32)

    def __len__(self):  
        return len(self.ids_cache)

    def __getitem__(self, item):
        input_ids = self.ids_cache[item].tolist()
        seq_length = min(self.max_seq_length-1, len(input_ids)-1)
        input_ids = [input_ids[0]] + input_ids[1:seq_length] + [input_ids[-1]]
        attention_mask = [1]*len(input_ids)

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": self.ids_key[item],
        }
        return ret_val

def single_get_collate_function(max_seq_length,padding=False):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt 
        length = None 
        if cnt < 10 or padding:
            length = max_seq_length
            cnt += 1
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function


class SubsetSeqDataset:
    def __init__(self,args, subset: List[int], ids_cache, max_seq_length):
        self.subset = sorted(list(subset))
        self.alldataset = SequenceDataset(args,ids_cache, max_seq_length)
        
    def __len__(self):  
        return len(self.subset)

    def __getitem__(self, item):
        return self.alldataset[self.subset[item]]

class TrainInbatchWithHardDataset:
    def __init__(self,args,rel_file,rank_file,queryids_cache,docids_cache,hard_num,max_query_length,max_doc_length):
        self.query_dataset  = SequenceDataset(args,queryids_cache,max_query_length)
        self.doc_dataset = SequenceDataset(args,docids_cache,max_doc_length)
        self.rel_dict = load_rel(rel_file)
        self.qids = sorted(list(self.rel_dict.keys()))
        self.rankdict = json.load(open(rank_file))
        assert hard_num > 0
        self.hard_num = hard_num 

    def __len__(self):
        return len(self.qids)
    def __getitem__(self,item):
        qid = self.qids[item]
        query_data = self.query_dataset[qid]
        passages_data = []
        hard_passagess_data = []
        for pid in self.rel_dict[qid]:
            passage_data = self.doc_dataset[pid]
            hardpids = random.sample(self.rankdict[str(qid)],self.hard_num)
            hard_passages_data = [self.doc_dataset[hardpid] for hardpid in hardpids ]
            passages_data.append(passage_data)
            hard_passagess_data.append(hard_passages_data)
        return {"query_data":query_data, 
                "passages_data":passages_data,
                "hard_passagess_data":hard_passagess_data}


class PassageCache:
    def __init__(self,passages_data,hard_passagess_data,
                    max_doc_length,padding,query_ids,
            rel_dict,query_num,hard_num_per_query):
        self.passages_data = passages_data
        self.hard_passagess_data = hard_passagess_data
        self.doc_collate_func = single_get_collate_function(max_doc_length,padding)
        self.query_ids = query_ids
        self.rel_dict = rel_dict
        self.query_num = query_num
        self.hard_num_per_query = hard_num_per_query
        self.device = "cuda"
    def __len__(self):
        return len(self.passages_data)
    def __getitem__(self,key):
        passage_data = [passage_data[key] for passage_data in self.passages_data]  
        hard_passage_data = sum([passage_data[key] for passage_data in self.hard_passagess_data],[])
        doc_data, doc_ids = self.doc_collate_func(passage_data)
        hard_doc_data, hard_doc_ids = self.doc_collate_func(hard_passage_data)
        rel_pair_mask = [[1 if docid not in [self.rel_dict[qid][key]] else 0
            for docid in doc_ids]
            for qid in self.query_ids]
        hard_pair_mask = [[1 if docid not in [self.rel_dict[qid][key]] else 0
            for docid in hard_doc_ids ]
            for qid in self.query_ids]
        return {
            "input_doc_ids":doc_data["input_ids"].to(self.device),
            "doc_attention_mask": doc_data["attention_mask"].to(self.device),
            "other_doc_ids":hard_doc_data['input_ids'].reshape(self.query_num, self.hard_num_per_query, -1).to(self.device),
            "other_doc_attention_mask":hard_doc_data['attention_mask'].reshape(self.query_num, self.hard_num_per_query, -1).to(self.device),
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask).to(self.device),
            "hard_pair_mask":torch.FloatTensor(hard_pair_mask).to(self.device),
        }


def triple_get_collate_function(max_query_length,max_doc_length,rel_dict,padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    def collate_function(batch):
        query_data,query_ids = query_collate_func([x["query_data"] for x in batch])
        query_num = len(query_data['input_ids'])
        hard_num_per_query = len(batch[0]["hard_passagess_data"][0])
        passages_data = [x["passages_data"] for x in batch]
        hard_passagess_data = [x["hard_passagess_data"] for x in batch]
        passage = PassageCache(passages_data,hard_passagess_data,max_doc_length,padding,query_ids,rel_dict,query_num,hard_num_per_query)
        return {
            "input_query_ids":query_data["input_ids"],
            "query_attention_mask":query_data["attention_mask"],
            "passages": passage
        }
    return collate_function