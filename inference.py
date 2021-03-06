import os 
from dataset import SequenceDataset,SubsetSeqDataset,single_get_collate_function,SequenceDataset,TextTokenIdsCache
import subprocess
import numpy as np 
import torch 
import logging
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
logger = logging.Logger(__name__)
from tqdm import tqdm
from retrieve_utils import index_retrieve,construct_flatindex_from_embeddings,convert_index_to_gpu
import faiss

def prediction(model, data_collator, args, test_dataset, embedding_memmap, ids_memmap, is_query):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size*args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            logits = model(is_query=is_query, **inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        ids_memmap[write_index:write_index+write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)

def query_inference(model,args,embedding_size):
    if os.path.exists(args.query_memmap_path):
        print(f"{args.query_memmap_path} exists, skip inference")
        return 
    query_collator = single_get_collate_function(args.max_query_length)
    query_dataset = SequenceDataset(args,
        ids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="query"),
        max_seq_length=args.max_query_length,is_query=True
    )
    query_memmap = np.memmap(args.query_memmap_path, 
        dtype=np.float32, mode="w+", shape=(len(query_dataset), embedding_size))
    queryids_memmap = np.memmap(args.warmup_retrieve_queryid_memmap_path, 
        dtype=np.int32, mode="w+", shape=(len(query_dataset), ))
    try:
        prediction(model, query_collator, args,
                query_dataset, query_memmap, queryids_memmap, is_query=True)
    except:
        subprocess.check_call(["rm", args.query_memmap_path])
        subprocess.check_call(["rm", args.warmup_retrieve_queryid_memmap_path])
        raise

def doc_inference(model, args, embedding_size):
    if os.path.exists(args.doc_memmap_path):
        print(f"{args.doc_memmap_path} exists, skip inference")
        return
    doc_collator = single_get_collate_function(args.max_doc_length)
    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passage")
    subset=list(range(len(ids_cache)))
    doc_dataset = SubsetSeqDataset(
        args,
        subset=subset,
        ids_cache=ids_cache,
        max_seq_length=args.max_doc_length
    )
    assert not os.path.exists(args.doc_memmap_path)
    doc_memmap = np.memmap(args.doc_memmap_path, 
        dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    docid_memmap = np.memmap(args.warmup_retrieve_docid_memmap_path, 
        dtype=np.int32, mode="w+", shape=(len(doc_dataset), ))
    try:
        prediction(model, doc_collator, args,
            doc_dataset, doc_memmap, docid_memmap, is_query=False
        )
    except:
        subprocess.check_call(["rm", args.doc_memmap_path])
        subprocess.check_call(["rm", args.warmup_retrieve_docid_memmap_path])
        raise

class State:
    def __init__(self,query_embedding,index):
        self.query = query_embedding
        self.index = index 

class Beam:
    batch_size:int 
    beam_size: int
    log_probs: torch.tensor
    finished: torch.tensor  
    state: State 
    def __init__(self,model,query_embedding,doc_embedding,index,beam_size,batch_size,device):
        self.query_embedding = torch.tensor(query_embedding).to(device)
        self.doc_embedding = torch.tensor(doc_embedding).to(device)
        init_state = State(self.query_embedding,index)
        self.nearest_neighbors,score = index_retrieve(init_state.index, init_state.query,beam_size,device, batch=batch_size)
        self.index = index 
        self.device = device
        self.log_probs = score
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.finished = torch.zeros(batch_size,beam_size,dtype=torch.bool).to(device)
        self.query_embedding = self.tile_along_beam(self.query_embedding,self.beam_size,device)
        tiled_doc_embedding = self.tile_along_beam(self.doc_embedding,self.beam_size,device)
        tiled_nearest_neighbors = self.tile_along_beam(self.nearest_neighbors,self.beam_size,device)
        tiled_doc_embedding = tiled_doc_embedding[tiled_nearest_neighbors.view(-1)]
        up_query = model.update_query(self.query_embedding,tiled_doc_embedding)
        self.model = model
        self.state = State(up_query,index)
        self.n_step=0
        self.max_step = 3
        self.pred_doc = (torch.ones((batch_size,self.max_step,self.beam_size),dtype=torch.long)*-2).to(device)
        self.terminated_doc_id = torch.tensor(-1,dtype=torch.long).to("cuda")
        for i in range(self.batch_size):
            for j in range(self.beam_size):
                if self.n_step == self.max_step or self.nearest_neighbors[i,j] ==  self.terminated_doc_id:
                    self.finished[i,j] = True
                    self.pred_doc[i,self.n_step,j] = self.nearest_neighbors[i,j]
                else:
                    self.pred_doc[i,self.n_step,j] = self.nearest_neighbors[i,j]

        self.n_step += 1
        self.log_probs_cache = []
        self.log_probs_cache.append(self.log_probs)
        del self.nearest_neighbors,tiled_nearest_neighbors
        import gc
        gc.collect()
    def tile_along_beam(self,x, beam_size,device, dim=0):
        bs = x.size(dim)
        tile_indices = torch.arange(bs).view(bs, 1).repeat(1, beam_size).view(bs*beam_size).to(device)
        return torch.index_select(x, dim, tile_indices)
        
    def done(self):
        return self.finished.all().item()

    def grow(self):
        nearest_neighbors,log_probs = index_retrieve(self.state.index, self.state.query,self.beam_size,self.device,batch=self.batch_size)
        x = log_probs.view(-1,self.beam_size,self.beam_size)
        x = x + self.log_probs.unsqueeze(-1)
        y = x.view(-1,self.beam_size*self.beam_size)
        values,indices = y.topk(self.beam_size,dim=-1)
        self.log_probs = values
        cnt = 0
        selected_query = []
        selected_doc = []
        
        for i in range(self.batch_size):
            for j in range(self.beam_size):
                idx = indices[cnt,j].item()
                m = cnt*self.beam_size +idx//self.beam_size
                n = idx%self.beam_size
                doc_id = nearest_neighbors[m,n].float().long()
                if self.finished[i,j]:
                    selected_query.append(self.state.query[idx])
                    selected_doc.append(doc_id)
                    continue
                if self.n_step == self.max_step or doc_id ==  self.terminated_doc_id:
                    self.finished[i,j] = True
                    selected_query.append(self.state.query[idx])
                    selected_doc.append(doc_id)
                else:
                    self.pred_doc[i,self.n_step,j] = doc_id
                    selected_query.append(self.state.query[idx])
                    selected_doc.append(doc_id)

            cnt+=1

        if self.done():
            return True
        self.log_probs_cache.append(self.log_probs)
        selected_query = torch.stack(selected_query).to(self.device)
        selected_doc = torch.stack(selected_doc).to(self.device).type(torch.long)

        selected_doc_embedding = self.doc_embedding[selected_doc]

        update_query = self.model.update_query(selected_query,selected_doc_embedding)

        
        self.n_step += 1
        self.state = State(update_query,self.index)
        return False

    def best_hypothesis(self):
        return self.pred_doc[:,:,0]
if __name__=='__main__':
    from model import BertDot
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mean", type=bool, default=True)
    parser.add_argument("--not_faiss_cuda", type=bool, default=False)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--doc_memmap_path", type=str, default="./warmup_retrieve/passage.memmap")
    parser.add_argument("--docid_memmap_path", type=str, default="./warmup_retrieve/passage_ids.memmap")
    parser.add_argument("--query_memmap_path", type=str, default="./warmup_retrieve/query.memmap")
    parser.add_argument("--queryid_memmap_path", type=str, default="./warmup_retrieve/query_ids.memmap")
    args = parser.parse_args()
    model = BertDot(args).to("cuda")
    model.load_state_dict(torch.load("data/passage/star_train/models/checkpoint-10000/pytorch_model.bin"))
    output_embedding_size = 768

    doc_embeddings = np.memmap(args.doc_memmap_path, dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path, dtype=np.int32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path, dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, output_embedding_size)
    query_ids = np.memmap(args.queryid_memmap_path, dtype=np.int32, mode="r")

    index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
    if torch.cuda.is_available() and not args.not_faiss_cuda:
        index = convert_index_to_gpu(index, list(range(args.n_gpu)), False)
    else:
        faiss.omp_set_num_threads(32)

    query_embeddings = query_embeddings[:100]
    beam = Beam(model,query_embeddings, doc_embeddings, index, 3, 100, "cuda")

    while not beam.grow():
        pass
    print(beam.best_hypothesis())