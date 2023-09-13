import torch.nn as nn
import torch
import numpy as np


class ScaledDotAttention(nn.Module):
    def __init__(self, d_k, attn_pdorp):
        super(ScaledDotAttention, self).__init__()
        self.d_k = d_k
        self.attn_pdrop = nn.Dropout(attn_pdorp)
    

    def forward(self, q_head, k_head, v_head, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k)
        # |k| : (batch_size, n_heads, k_len, d_k)
        # |v| : (batch_size, n_heads, v_len, d_v)
        # |attn_mask| : (batch_size, n_heads, q_len, k_len)

        attn_score = torch.matmul(q_head, k_head.transpose(-1, -2)) / (self.d_k ** 0.5) #(batch, n_head, q_len, k_len)
        attn_score.masked_fill_(attn_mask, -1e9) # 掩码操作
        # |attn_scroe| : (batch_size, n_heads, q_len, k_len)

        attn_weights = nn.Softmax(dim=-1)(attn_score)  # 最后一维是Q*K
        attn_weights = self.attn_pdrop(attn_weights)

        output = torch.matmul(attn_weights, v_head)
        # |output| : (batch_size, n_heads, q_len, d_v)
        return output, attn_weights
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_pdrop):
        """
        dmodel: Embeding Size
        n_heads: num of head
        attn_pdrop: 注意力机制中Q和K进行softmax和加入dropout
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model//n_heads  # Q K V的维度，为了后面concat之后维度==d_model
        self.WQ = nn.Linear(d_model, d_model) # output:n_heads*self._d_q = d_model
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        
        self.scaled_dot_prodcut_attn = ScaledDotAttention(self.d_k, attn_pdrop)

        self.fc = nn.Linear(d_model, d_model)
    

    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len(=seq_len), d_model)
        # |K| : (batch_size, k_len(=seq_len), d_model)
        # |V| : (batch_size, v_len(=seq_len), d_model)
        # |attn_mask| : (batch_size, q_len, k_len)
        batch_size = Q.size(0)
        q_head = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_head = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_head = self.WV(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)

        
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)  #复制维度

        #注意力机制
        attn, attn_weight = self.scaled_dot_prodcut_attn(q_head, k_head, v_head, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        attn = attn.transpose(1,2).contiguous().view(batch_size, -1, self.d_v*self.n_heads)
        # |attn| : (batch_size,q_len, d_model)
        output = self.fc(attn)
        
        return output, attn_weight
    

class PositionWiseFeedForwardNetwork(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model,d_ff):
        self.super(PositionWiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)
    
    def forward(self, input):
        output = self.gelu(self.linear1(input))
        output = self.linear2(output)
        return output
    

class DecodeLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_pdorp, resid_pdrop):
        super(DecodeLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, attn_pdorp)
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout1 = nn.Dropout(resid_pdrop)
        self.dropout2 = nn.Dropout(resid_pdrop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-5)

    
    def forward(self, input, attn_mask):
        attn, attn_weight = self.mha(input, input, input, attn_mask)
        attn = self.dropout1(attn)
        attn = self.layernorm1(attn+input)  # attention部分

        ffn = self.ffn(attn)
        ffn = self.dropout2(ffn)
        ffn = self.layernorm2(ffn+attn)  # 前馈神经网络部分

        return ffn, attn_weight
    

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layers, embd_pdrop, pad_id,
                d_model, n_heads, d_ff, attn_pdorp, resid_pdrop):
        super(TransformerDecoder,self).__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, d_model)  # 单词编码
        self.pos_embedding = nn.Embedding(seq_len+1, d_model)
        self.layer = nn.modules([DecodeLayer(d_model, n_heads, d_ff, attn_pdorp, resid_pdrop) 
                                 for _ in range(n_layers)])
        self.dropout = nn.Dropout(embd_pdrop)
        nn.init.normal_(self.embedding.weight, std=0.02)

    
    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len)
        
        # 先做position embedding, 已经padding成相同长度了，只用转换为true，false
        positions = torch.arange(input.size(1), device=input.device, dtype=torch.long).repeat(input.size(0), 1)+1
        positions_pad_mask = positions.eq(self.pad_id)
        positions.masked_fill_(positions_pad_mask, 0)
        # |positions| : (batch_size, seq_len)












if __name__=="__main__":
    pass