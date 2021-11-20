from genericpath import exists
import json
import gzip 
import pickle
import subprocess 
import csv
import numpy as np
import os
from os import listdir
from os.path import isfile,join
from tqdm import tqdm 
from transformers import AutoTokenizer
import argparse
import multiprocessing
from transformers import AutoTokenizer
from collections import defaultdict
class DataText():
    def __init__(self,train_path,legal_corpus_path):
        with open(train_path) as t:
            self.train_data = json.load(t)
        self.items = self.train_data["items"]
        self.max_docs = self.max_articles(self.items)
        with open(legal_corpus_path) as l:
            self.legal_corpus = json.load(l)
        
    def max_articles(self,items):
        max_len = -1
        for it in items:
            tmp_len = len(it["relevant_articles"])
            if tmp_len > max_len:
                max_len = tmp_len
        return max_len
    def __len__(self):
        return len(self.items)
    def find_answer_doc(self,law_id,art_id):
        for item in self.legal_corpus:
            if item["law_id"] == law_id:
                articles = item["articles"]
                for art in articles:
                    if art["article_id"] == art_id:
                        return art["title"]+" "+art["text"]

    def __getitem__(self,key):
        question_id = self.items[key]["question_id"]
        question = self.items[key]["question"]
        answer_ids = []
        # answer_docs = []
        for articles in self.items[key]["relevant_articles"]:
            answer_ids.append(articles["law_id"]+"+"+articles["article_id"])
            # answer_docs.append(self.find_answer_doc(articles["law_id"],articles["article_id"]))
        if len(answer_ids) < self.max_docs:
            answer_ids.extend(["-1"]*(self.max_docs-len(answer_ids)))
            # answer_docs.extend([""]*(self.max_docs-len(answer_docs)))

        return {
            "question_id": question_id,
            "question":question,
            "answer_ids":answer_ids,
            # "answer_docs":answer_docs
        }

class legal_corpus:
    def __init__(self,legal_corpus_path):
        with open(legal_corpus_path) as l:
            self.legal_corpus = json.load(l)
        self.lawartid2int = {}
        self.id2lawart={}
        cnt = 0
        for item in self.legal_corpus:
            law_id = item["law_id"]
            for art in item["articles"]:
                self.lawartid2int[law_id+"+"+art["article_id"]] = cnt
                self.id2lawart[law_id+"+"+art["article_id"]] = art["title"] + " " +art["text"]
                cnt+=1
        self.lawartid2int["-1"] = -1
        self.id2lawart["-1"] = ""
    def __len__(self):
        return len(self.lawartid2int)
    def __getitem__(self,id):
        return self.lawartid2int[id]

def pad_input_ids(input_ids,max_length,pad_on_left=False,pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids

def QueryPreprocessingFn(args, line, tokenizer):
    passage = tokenizer.encode(
        line.rstrip(),
        add_special_tokens=True,
        max_length=args.max_query_length,
        truncation=True)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)
    return input_id_b, passage_len

def PassagePreprocessingFn(args, line, tokenizer):
    line = line.strip()
    p_text = line
    # keep only first 10000 characters, should be sufficient for any
    # experiment that uses less than 500 - 1k tokens
    full_text = p_text[:args.max_doc_character]
    passage = tokenizer.encode(
        full_text,
        add_special_tokens=True,
        max_length=args.max_seq_length,
        truncation=True
    )
    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)
    return input_id_b, passage_len

def Mapid2int(args,data_text,legal_corpus_path,output_dir,tokenizer):
    qid2int = defaultdict()
    legal_corpus_id = legal_corpus(legal_corpus_path)
    qint2lint=defaultdict()
    lid2lint = legal_corpus_id.lawartid2int
    data_cnt = len(data_text)
    passage_cnt = len(legal_corpus_id)
    legal_id2text = legal_corpus_id.id2lawart
    
    ids_q_array = np.memmap(
        os.path.join(output_dir, "query_ids.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    token_ids_q_array = np.memmap(
        os.path.join(output_dir, "query_token_ids.memmap"),
        shape=(data_cnt, args.max_query_length), mode='w+', dtype=np.int32)
    token_length_q_array = np.memmap(
        os.path.join(output_dir, "query_lengths.memmap"),
        shape=(data_cnt, ), mode='w+', dtype=np.int32)
    
    ids_p_array = np.memmap(
        os.path.join(output_dir, "passage_ids.memmap"),
        shape=(passage_cnt, ), mode='w+', dtype=np.int32)
    token_ids_p_array = np.memmap(
        os.path.join(output_dir, "passage_token_ids.memmap"),
        shape=(passage_cnt, args.max_seq_length), mode='w+', dtype=np.int32)
    token_length_p_array = np.memmap(
        os.path.join(output_dir, "passage_lengths.memmap"),
        shape=(passage_cnt, ), mode='w+', dtype=np.int32)

    write_q_idx = 0
    for i in range(len(data_text)):
        qid2int[data_text[i]["question_id"]] = i
        query = data_text[i]["question"]
        token_ids, length = QueryPreprocessingFn(args,query, tokenizer)
        ids_q_array[write_q_idx] = i
        token_ids_q_array[write_q_idx, :] = token_ids
        token_length_q_array[write_q_idx] = length
        write_q_idx+=1

        qint2lint[i] = []
        for ans_id in data_text[i]["answer_ids"]:                
            qint2lint[i].append(legal_corpus_id[ans_id])

    write_p_idx = 0
    for item in legal_id2text:
        ids_p_array[write_p_idx] = legal_corpus_id[item]
        legal_text = legal_id2text[item]
        token_ids, length = PassagePreprocessingFn(args,legal_text, tokenizer)
        token_ids_p_array[write_p_idx, :] = token_ids
        token_length_p_array[write_p_idx] = length
        write_p_idx+=1

    meta = {'type': 'int32', 'total_number':write_q_idx ,
            'embedding_size': args.max_query_length}
    with open("./preprocess/query" + "_meta", 'w') as f:
        json.dump(meta, f)

    meta = {'type': 'int32', 'total_number':write_p_idx ,
            'embedding_size': args.max_seq_length}
    with open("./preprocess/passage" + "_meta", 'w') as f:
        json.dump(meta, f)

    lid2lint_path = "./preprocess/lid2lint.pkl"
    with open(lid2lint_path, 'wb') as handle:
        pickle.dump(lid2lint, handle, protocol=4)
    qid2int_path = "./preprocess/qid2int.pkl"
    with open(qid2int_path,"wb") as handle:
        pickle.dump(qid2int,handle,protocol=4)
    qint2lint_path = "./preprocess/qint2lint.pkl"
    with open(qint2lint_path,"wb") as handle:
        pickle.dump(qint2lint,handle,protocol=4)
    
    with open(os.path.join(args.out_data_dir,"train-qrel.tsv"), "w", encoding='utf-8') as qrel_output: 
        out_line_count = 0
        for q2p in qint2lint:
            topicid = int(q2p)
            docids = [str(d) for d in qint2lint[q2p]]
            rel = 1
            qrel_output.write(str(topicid) +
                         "\t0\t" + str("\t".join(docids)) +
                         "\t" + str(rel) + "\n")
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",default='bert-base-cased',type=str)
    parser.add_argument("--max_seq_length",default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument(  "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",)
    parser.add_argument("--threads", type=int, default=16)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    data_text = DataText("train_question_answer.json","legal_corpus.json")
    args = get_arguments()
    args.out_data_dir = "./preprocess"
    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    Mapid2int(args,data_text,"legal_corpus.json",args.out_data_dir,tokenizer)
    