import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, seq_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.seg_num_x = self.seq_len // self.patch_len
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.patch_len // 2),
                               stride=1, padding=self.patch_len // 2, padding_mode="zeros", bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        batch,n_vars,temp = x.shape
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, n_vars, self.seq_len) + x
        
        x = x.reshape(batch, n_vars, self.seg_num_x, self.patch_len)
        
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x),batch,n_vars,x.shape[2]
    
    


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, d_model, d_core, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_core_head = d_core // n_heads
        self.gen1 = nn.Linear(d_model, d_model)
        self.gen2 = nn.Linear(d_model, self.d_head*self.n_heads)
        self.gen3 = nn.Conv1d(in_channels=self.d_head*self.n_heads,
                              out_channels=self.d_core_head*self.n_heads,
                              kernel_size=1,
                              groups=self.n_heads)

        self.gen4 = nn.Linear(d_core, d_core)
        self.gen5 = nn.Linear(d_model + d_core, d_model)
        self.gen6 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, B, N, D, x_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        batch_size, channels, d_model = x_glb.shape
        H = self.n_heads
        #multihead = F.gelu(self.gen1(x_glb))
        multihead = F.gelu(self.gen2(x_glb)).view(batch_size,channels,H,-1)
        multihead_core = multihead.reshape(batch_size*channels, H*self.d_head, 1)
        multihead_core = F.gelu(self.gen3(multihead_core)).view(batch_size,channels,H,-1)
        combined_mean = multihead_core.reshape(batch_size,channels,H*self.d_core_head)
        combined_mean = self.gen4(combined_mean)
        
        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        combined_glb_cat = torch.cat([x_glb, combined_mean], -1)
        combined_glb_cat = F.gelu(self.gen5(combined_glb_cat))
        combined_glb_cat = self.gen6(combined_glb_cat)
        combined_glb_cat = torch.reshape(combined_glb_cat,
                                   (combined_glb_cat.shape[0] * combined_glb_cat.shape[1], combined_glb_cat.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + combined_glb_cat
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, b, n, d, x_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, b, n, d, x_mask=x_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.period_len
        self.patch_num = int(configs.seq_len // configs.period_len)
        
        
        self.en_embedding = EnEmbedding(configs.n_vars, configs.d_model, self.patch_len, self.seq_len, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                     AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_core,
                    configs.n_heads,
                    configs.d_ff,
                    configs.dropout
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.n_vars, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out,b,n,d = self.en_embedding(x_enc.permute(0,2,1))
        enc_out = self.encoder(enc_out, b, n, d)
        enc_out = torch.reshape(
            enc_out, (-1, n, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        
         # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        