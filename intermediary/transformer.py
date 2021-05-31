import torch.nn as nn
import torch
# import torch.nn.functional as F


class TextSentiment(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=2048, num_class=1024):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, text):
        embedded = self.layer(text)
        # return torch.round(self.fc(embedded))
        return self.fc(embedded)


class TextSentiment_res(nn.Module):
    def __init__(self, in_dim=2048, embed_dim=2048, num_class=1024, bn=False):
        super().__init__()
        self.head = nn.Linear(in_dim, embed_dim)
        self.layer = nn.Sequential(
            ResBlock(embed_dim=embed_dim, bn=bn),
            # ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, text):
        text = self.head((text))
        embedded = self.layer(text)
        # return torch.round(self.fc(embedded))
        return self.fc(embedded)



class TextSentiment_deep(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=2048, num_class=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.ReLU(True)
        )
        # block = [nn.Linear(embed_dim, embed_dim),nn.ReLU(True),nn.BatchNorm1d(embed_dim)]
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),nn.ReLU(True),
            # nn.Linear(embed_dim, embed_dim),nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid())

    def forward(self, text):
        text = self.head(text)
        embedded = self.layer(text)
        # return torch.round(self.fc(embedded))
        return self.fc(embedded)


class ResBlock(nn.Module):
    def __init__(self, embed_dim=2048, bn=False):
        super().__init__()
        m = []
        for _ in range(2):
            m.append(nn.Linear(embed_dim, embed_dim))
            m.append(nn.ReLU(True))
            if bn: m.append(nn.BatchNorm1d(embed_dim))
        self.layer = nn.Sequential(*m)

    def forward(self, x):
        res = self.layer(x)
        return x + res

class BasicBlock(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048, bn=False):
        super().__init__()
        m = []
        m.append(nn.Linear(in_dim, out_dim))
        m.append(nn.ReLU(True))
        if bn: m.append(nn.BatchNorm1d(out_dim))
        self.layer = nn.Sequential(*m)

    def forward(self, x):
        return self.layer(x)



class DnText_res(nn.Module):
    def __init__(self, in_dim=2048, embed_dim=2048, num_class=1024, bn=False):
        super().__init__()
        self.head = nn.Linear(in_dim, embed_dim)
        self.layer = nn.Sequential(
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn),
            ResBlock(embed_dim=embed_dim, bn=bn)
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, text):
        text = self.head(text)
        embedded = self.layer(text)
        embedded = embedded + text
        # return torch.round(self.fc(embedded))
        return self.fc(embedded)



class DnText_linear(nn.Module):
    def __init__(self, in_dim=2048, embed_dim=2048, num_class=1024, bn=False):
        super().__init__()
        self.head = nn.Linear(in_dim, embed_dim)
        layer = [BasicBlock(in_dim=embed_dim, out_dim=embed_dim, bn=bn) for i in range(40) ]
        self.layer = nn.Sequential(
            *layer
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, text):
        text = self.head(text)
        embedded = self.layer(text)
        embedded = embedded + text
        # return torch.round(self.fc(embedded))
        return self.fc(embedded)


class UNet(nn.Module):
    def __init__(self, in_dim=2048, embed_dim=2048, num_class=1024, num_res=1, bn=False):
        super().__init__()
        self.head = BasicBlock(in_dim=in_dim, out_dim=embed_dim, bn=bn)
        self.layer_i1 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim, out_dim=embed_dim, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim, bn=bn) for _ in range(num_res)]
        )
        self.layer_i2 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim, out_dim=embed_dim//2, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//2, bn=bn) for _ in range(num_res)]
        )
        self.layer_i3 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim//4, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//4, bn=bn) for _ in range(num_res)]
        )
        self.layer_i4 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim//4, out_dim=embed_dim//8, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//8, bn=bn) for _ in range(num_res)]
        )
        self.layer_o4 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//8, bn=bn) for _ in range(num_res)] + 
            [BasicBlock(in_dim=embed_dim//8, out_dim=embed_dim//4, bn=bn)]
        )
        self.layer_o3 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//4, bn=bn) for _ in range(num_res)] + 
            [BasicBlock(in_dim=embed_dim//4, out_dim=embed_dim//2, bn=bn)]
        )
        self.layer_o2 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//2, bn=bn) for _ in range(num_res)] +
            [BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim, bn=bn)]
        )
        # self.layer_o1 =  nn.Sequential(
        #     BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim, bn=bn),
        #     ResBlock(embed_dim=embed_dim, bn=bn)
        # )
        self.tail = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):     #(in_dim, out_dim)
        x = self.head(x)
        i1 = self.layer_i1(x) # 2048, 2048
        i2 = self.layer_i2(i1) # 2048, 1024
        i3 = self.layer_i3(i2) # 1024, 512
        i4 = self.layer_i4(i3) # 512, 216
        out = self.layer_o4(i4) # 216, 512
        out = self.layer_o3(out+i3) # 512, 1024
        out = self.layer_o2(out+i2) # 1024, 2048
        return self.tail(i1+out)




class AlbertSelfAttention(nn.Module):
    def __init__(self, hidden_size=4096, num_attention_heads=64, attention_probs_dropout_prob=0, output_attentions=False):
        super(AlbertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # return x.permute(0, 2, 1, 3)
        return x.permute(1, 0, 2)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(1, 0, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else context_layer
        return outputs




class UNetAttention(nn.Module):
    def __init__(self, in_dim=2048, embed_dim=2048, num_class=1024, num_res=1, bn=False):
        super().__init__()
        self.head = BasicBlock(in_dim=in_dim, out_dim=embed_dim, bn=bn)
        self.layer_i1 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim, out_dim=embed_dim, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim, num_attention_heads=64)]
        )
        self.layer_i2 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim, out_dim=embed_dim//2, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//2, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim//2, num_attention_heads=32)]
        )
        self.layer_i3 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim//4, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//4, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim//4, num_attention_heads=16)]
        )
        self.layer_i4 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim//4, out_dim=embed_dim//8, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//8, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim//8, num_attention_heads=8)]
        )
        self.layer_o4 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//8, bn=bn) for _ in range(num_res)] + 
            # [AlbertSelfAttention(hidden_size=embed_dim//8, num_attention_heads=8)] +
            [BasicBlock(in_dim=embed_dim//8, out_dim=embed_dim//4, bn=bn)]
        )
        self.layer_o3 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//4, bn=bn) for _ in range(num_res)] + 
            # [AlbertSelfAttention(hidden_size=embed_dim//4, num_attention_heads=16)] +
            [BasicBlock(in_dim=embed_dim//4, out_dim=embed_dim//2, bn=bn)]
        )
        self.layer_o2 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//2, bn=bn) for _ in range(num_res)] +
            # [AlbertSelfAttention(hidden_size=embed_dim//2, num_attention_heads=32)] +
            [BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim, bn=bn)] #+ 
            + [AlbertSelfAttention(hidden_size=embed_dim, num_attention_heads=64)]
        )
        # self.layer_o1 =  nn.Sequential(
        #     BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim, bn=bn),
        #     ResBlock(embed_dim=embed_dim, bn=bn)
        # )
        self.tail = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):     #(in_dim, out_dim)
        x = self.head(x)
        i1 = self.layer_i1(x) # 2048, 2048
        i2 = self.layer_i2(i1) # 2048, 1024
        i3 = self.layer_i3(i2) # 1024, 512
        i4 = self.layer_i4(i3) # 512, 216
        out = self.layer_o4(i4) # 216, 512
        out = self.layer_o3(out+i3) # 512, 1024
        out = self.layer_o2(out+i2) # 1024, 2048
        return self.tail(i1+out)



class UNetAttention_v2(nn.Module):
    def __init__(self, in_dim=2048, embed_dim=2048, num_class=1024, num_res=1, bn=False):
        super().__init__()
        self.head = BasicBlock(in_dim=in_dim, out_dim=embed_dim, bn=bn)
        self.layer_i1 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim, out_dim=embed_dim, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim, num_attention_heads=64)]
        )
        self.layer_i2 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim, out_dim=embed_dim//2, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//2, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim//2, num_attention_heads=32)]
        )
        self.layer_i3 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim//4, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//4, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim//4, num_attention_heads=16)]
        )
        self.layer_i4 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim//4, out_dim=embed_dim//8, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//8, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim//8, num_attention_heads=8)]
        )
        self.layer_i5 =  nn.Sequential(*
            [BasicBlock(in_dim=embed_dim//8, out_dim=embed_dim//16, bn=bn)] + 
            [ResBlock(embed_dim=embed_dim//16, bn=bn) for _ in range(num_res)] #+ 
            # [AlbertSelfAttention(hidden_size=embed_dim//8, num_attention_heads=8)]
        )
        self.layer_o5 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//16, bn=bn) for _ in range(num_res)] + 
            # [AlbertSelfAttention(hidden_size=embed_dim//8, num_attention_heads=8)] +
            [BasicBlock(in_dim=embed_dim//16, out_dim=embed_dim//8, bn=bn)]
        )
        self.layer_o4 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//8, bn=bn) for _ in range(num_res)] + 
            # [AlbertSelfAttention(hidden_size=embed_dim//8, num_attention_heads=8)] +
            [BasicBlock(in_dim=embed_dim//8, out_dim=embed_dim//4, bn=bn)]
        )
        self.layer_o3 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//4, bn=bn) for _ in range(num_res)] + 
            # [AlbertSelfAttention(hidden_size=embed_dim//4, num_attention_heads=16)] +
            [BasicBlock(in_dim=embed_dim//4, out_dim=embed_dim//2, bn=bn)]
        )
        self.layer_o2 =  nn.Sequential(*
            [ResBlock(embed_dim=embed_dim//2, bn=bn) for _ in range(num_res)] +
            # [AlbertSelfAttention(hidden_size=embed_dim//2, num_attention_heads=32)] +
            [BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim, bn=bn)] #+ 
            + [AlbertSelfAttention(hidden_size=embed_dim, num_attention_heads=64)]
        )
        # self.layer_o1 =  nn.Sequential(
        #     BasicBlock(in_dim=embed_dim//2, out_dim=embed_dim, bn=bn),
        #     ResBlock(embed_dim=embed_dim, bn=bn)
        # )
        self.tail = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):     #(in_dim, out_dim)
        x = self.head(x)
        i1 = self.layer_i1(x) # 2048, 2048
        i2 = self.layer_i2(i1) # 2048, 1024
        i3 = self.layer_i3(i2) # 1024, 512
        i4 = self.layer_i4(i3) # 512, 216
        i5 = self.layer_i5(i4) # 512, 216
        out = self.layer_o5(i5)
        out = self.layer_o4(out+i4) # 216, 512
        out = self.layer_o3(out+i3) # 512, 1024
        out = self.layer_o2(out+i2) # 1024, 2048
        return self.tail(i1+out)


#  class TransformerModel(nn.Module):
    
#     def __init__(self):
#         super(TransformerModel, self).__init__()
#         self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12, num_decoder_layers=12)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.transformer.bias.data.zero_()
#         self.transformer.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src, dst):
#         B, _ = src.shape
#         src = src.view(B, 4, 512).permute(1,0,2)
#         dst = dst.view(B, 2, 512).permute(1,0,2)
#         out = self.transformer(src, dst)
#         out = out.permute(1,0,2).view(B,-1)

#         return out




class Rrelu(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 1] = 0
        return grad_input


class RReLU(nn.Module):

    def __init__(self):
        super(RReLU, self).__init__()

    def forward(self, x):
        out = Rrelu.apply(x)
        return out


import math
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ninp=512, nhead=16, nhid=512, nlayers=16, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ninp//2)

        # self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        B, _ = src.shape
        src = src.view(B, 4, 512).permute(1,0,2)
        src_mask = self.generate_square_subsequent_mask(4).cuda()
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = output.permute(1,0,2)
        # print('output.shape: ', output.shape)
        return output.reshape(B,-1)


# class PositionalEncoding(nn.Module):
    
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)        