import enum 
from torch._C import dtype
import torch 
from transformers import AutoModel 
import torch.nn.functional as F 
from torch.cuda.amp import autocast 
import torch.nn as nn
from transformers import AutoConfig,AutoModel
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(logits,dim=-1):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    softmax_fn = F.softmax
    probs = softmax_fn(logits, dim)
    return probs

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        # c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        # q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = softmax(s, dim=2)       # (batch_size, c_len, q_len)
        s2 = softmax(s, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x
    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class EmbeddingMixin:
    def __init__(self,model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:",self.use_mean)
    def __init_weights(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding,nn.Conv1d)):
            module.weight.data.normal_(mean=0.0,std=0.2)

    def masked_mean(self,t,mask):
        s = torch.sum(t*mask.unsqueeze(-1).float(),axis=1)
        d = mask.sum(axis=1,keepdim=True).float()
        return s/d 
    def masked_mean_or_first(self,emb_all,mask):
        if self.use_mean:
            return self.masked_mean(emb_all[0],mask)
        else:
            return emb_all[0][:,0]
    def query_emb(self,input_ids,attention_mask):
        raise NotImplementedError
    def body_emb(self,input_ids,attention_mask):
        raise NotImplementedError

class BaseModelDot(EmbeddingMixin):
    def _text_encode(self, input_ids, attention_mask):
        return None 

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self._text_encode(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, is_query, *args):
        assert len(args) == 0
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.body_emb(input_ids, attention_mask)

class BertDot(BaseModelDot,nn.Module):
    def __init__(self, model_argobj=None):
        BaseModelDot.__init__(self, model_argobj)
        nn.Module.__init__(self)
        self.config = AutoConfig.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.electra = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        print(self.electra.eval())
        # pretrained_dict = torch.load(args.model_path)
        # model_dict = self.electra.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # self.electra.load_state_dict(pretrained_dict, strict=False)

        self.output_embedding_size = self.electra.config.hidden_size
        self.embeddingHead = nn.Linear(self.electra.config.hidden_size,self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        # self.transformer_model = nn.Transformer(nhead=12, num_encoder_layers=1,d_model=768)
        self.bidafa_layer = BiDAFAttention(self.output_embedding_size)
        self.reduced_dim = nn.Sequential(
            nn.Linear(4*self.output_embedding_size, 2*self.output_embedding_size),
            nn.LayerNorm(2*self.output_embedding_size),
            nn.ReLU(),
            nn.Linear(2*self.output_embedding_size, self.output_embedding_size),
        )
        self.apply(self._init_weights)

    def update_query(self,emb_query,emb_doc):
        query_embedding = emb_query.unsqueeze(0)
        doc_embedding = emb_doc.unsqueeze(0)
        out = self.bidafa_layer(query_embedding,doc_embedding)
        update_query = self.reduced_dim(out)
        return update_query.squeeze(0)

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.electra(input_ids=input_ids,attention_mask=attention_mask)
        return outputs1

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertDot_InBatch(BertDot):
    def forward(self,input_query_ids,query_attention_mask,passages):
        num_query = len(passages.passages_data[0])
        total_loss = 0
        pre_query_embs = None
        pre_doc_embs = None 
        for idx in range(num_query):
            passage =  passages[idx]
            input_doc_ids,doc_attention_mask,other_doc_ids, \
                other_doc_attention_mask,hard_pair_mask,rel_pair_mask = passage["input_doc_ids"],passage["doc_attention_mask"], \
                    passage["other_doc_ids"], passage["other_doc_attention_mask"],passage["hard_pair_mask"],passage["rel_pair_mask"]
            if idx == 0:
                loss,query_embs,doc_embs = self.inbatch_train(self.query_emb, self.body_emb,
                        input_query_ids, query_attention_mask,
                        input_doc_ids, doc_attention_mask, 
                        other_doc_ids, other_doc_attention_mask,
                        rel_pair_mask, hard_pair_mask,True,None,None)
                pre_query_embs = query_embs
                pre_doc_embs = doc_embs
                del query_embs,doc_embs
                gc.collect()
            else:
                loss, query_embs,doc_embs = self.inbatch_train(None, self.body_emb,
                    None,None,
                    input_doc_ids, doc_attention_mask, 
                    other_doc_ids, other_doc_attention_mask,None,
                    hard_pair_mask,False,pre_query_embs,pre_doc_embs)
                pre_query_embs = query_embs
                pre_doc_embs = doc_embs
            total_loss += loss
        return (total_loss/num_query,) 

    def inbatch_train(self,query_encode_func, doc_encode_func,
            input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            other_doc_ids=None, other_doc_attention_mask=None,
            rel_pair_mask=None, hard_pair_mask=None,is_first=False,pre_query_embs=None,pre_doc_embs=None):

        if is_first:
            query_embs = query_encode_func(input_query_ids, query_attention_mask)
        else:
            query_embs = self.update_query(pre_query_embs,pre_doc_embs)

        doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)

        batch_size = query_embs.shape[0]
        with autocast(enabled=False):
            batch_scores = torch.matmul(query_embs, doc_embs.T)
            # print("batch_scores", batch_scores)
            single_positive_scores = torch.diagonal(batch_scores, 0)
            # print("positive_scores", positive_scores)
            positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)
            if rel_pair_mask is None:
                rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)                
            # print("mask", mask)
            batch_scores = batch_scores.reshape(-1)
            logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                    batch_scores.unsqueeze(1)], dim=1)  
            # print(logit_matrix)
            lsm = F.log_softmax(logit_matrix, dim=1)
            loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)
            # print(loss)
            # print("\n")
            first_loss, first_num = loss.sum(), rel_pair_mask.sum()

        if other_doc_ids is None:
            return (first_loss/first_num,)

        # other_doc_ids: batch size, per query doc, length
        other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
        other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)
        other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)
        other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)
        with autocast(enabled=False):
            other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)
            other_batch_scores = other_batch_scores.reshape(-1)
            positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)
            other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                    other_batch_scores.unsqueeze(1)], dim=1)  
            # print(logit_matrix)
            other_lsm = F.log_softmax(other_logit_matrix, dim=1)
            other_loss = -1.0 * other_lsm[:, 0]
            # print(loss)
            # print("\n")
            if hard_pair_mask is not None:
                hard_pair_mask = hard_pair_mask.reshape(-1)
                other_loss = other_loss * hard_pair_mask
                second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
            else:
                second_loss, second_num = other_loss.sum(), len(other_loss)

        return (first_loss+second_loss)/(first_num+second_num),query_embs,doc_embs
