
################################################################
# this is translated from the one-step shortcut flow-matching paper's jax code. 
# it receives visual inputs and discrete lables. 
################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TrainConfig:
    def __init__(self, dtype):
        self.dtype = dtype

    def kern_init(self, name='default', zero=False):
        if zero or 'bias' in name:
            return lambda m: nn.init.constant_(m, 0)
        return lambda m: nn.init.xavier_uniform_(m, gain=math.sqrt(2))

    def default_config(self):
        return {
            'kernel_init': self.kern_init(),
            'bias_init': self.kern_init('bias', zero=True),
            'dtype': self.dtype,
        }

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, tc, frequency_embedding_size=256):
        super(TimestepEmbedder, self).__init__()
        self.hidden_size = hidden_size
        self.tc = tc
        self.frequency_embedding_size = frequency_embedding_size
        self.dense1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.tc.kern_init('time_bias')(self.dense1.bias)
        self.tc.kern_init('time_bias')(self.dense2.bias)

    def forward(self, t):
        x = self.timestep_embedding(t)
        x = self.dense1(x)
        x = F.silu(x)
        x = self.dense2(x)
        return x

    def timestep_embedding(self, t, max_period=10000):
        t = t.float()
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, stop=half, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = embedding.to(self.tc.dtype)
        return embedding

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, tc):
        super(LabelEmbedder, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.tc = tc
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.embedding_table.weight, 0.02)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, hidden_size, tc, bias=True):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.tc = tc
        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.tc.kern_init('patch')(self.conv.weight)
        if bias:
            self.tc.kern_init('patch_bias', zero=True)(self.conv.bias)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.conv(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

class MlpBlock(nn.Module):
    def __init__(self, mlp_dim, tc, out_dim=None, dropout_rate=None, train=False):
        super(MlpBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.tc = tc
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.train = train
        actual_out_dim = out_dim if out_dim is not None else mlp_dim
        self.dense1 = nn.Linear(mlp_dim, actual_out_dim)
        self.dense2 = nn.Linear(actual_out_dim, mlp_dim)
        self.tc.default_config()['kernel_init'](self.dense1.weight)
        self.tc.default_config()['kernel_init'](self.dense2.weight)
        self.tc.default_config()['bias_init'](self.dense1.bias)
        self.tc.default_config()['bias_init'](self.dense2.bias)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.train)
        output = self.dense2(x)
        output = F.dropout(output, p=self.dropout_rate, training=self.train)
        return output

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, tc, mlp_ratio=4.0, dropout=0.0, train=False):
        super(DiTBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tc = tc
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.train = train
        self.dense = nn.Linear(hidden_size, 6 * hidden_size)
        self.tc.default_config()['kernel_init'](self.dense.weight)
        self.tc.default_config()['bias_init'](self.dense.bias)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.mlp = MlpBlock(mlp_dim=int(hidden_size * mlp_ratio), tc=tc, dropout_rate=dropout, train=train)

    def forward(self, x, c):
        c = F.silu(c)
        c = self.dense(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.split(c, 6, dim=-1)

        x_norm = self.norm1(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        B, N, C = x_modulated.shape
        qkv = self.qkv(x_modulated).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x + (gate_msa[:, None] * x)

        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        x = x + (gate_mlp[:, None] * self.mlp(x_modulated2))
        return x

class FinalLayer(nn.Module):
    def __init__(self, patch_size, out_channels, hidden_size, tc):
        super(FinalLayer, self).__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.tc = tc
        self.dense = nn.Linear(hidden_size, 2 * hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_dense = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.tc.kern_init(zero=True)(self.dense.weight)
        self.tc.kern_init('bias', zero=True)(self.dense.bias)
        self.tc.kern_init('final', zero=True)(self.final_dense.weight)
        self.tc.kern_init('final_bias', zero=True)(self.final_dense.bias)

    def forward(self, x, c):
        c = F.silu(c)
        c = self.dense(c)
        shift, scale = torch.split(c, 2, dim=-1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.final_dense(x)
        return x

class DiT(nn.Module):
    def __init__(self, patch_size, hidden_size, depth, num_heads, mlp_ratio, out_channels, class_dropout_prob, num_classes, ignore_dt=False, dropout=0.0, dtype=torch.float32):
        super(DiT, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.ignore_dt = ignore_dt
        self.dropout = dropout
        self.dtype = dtype
        self.tc = TrainConfig(dtype=dtype)
        self.patch_embed = PatchEmbed(patch_size, hidden_size, tc=self.tc)
        self.pos_embed = nn.Parameter(torch.zeros(1, (256 // patch_size) ** 2, hidden_size), requires_grad=False)
        self.timestep_embedder = TimestepEmbedder(hidden_size, tc=self.tc)
        self.label_embedder = LabelEmbedder(num_classes, hidden_size, tc=self.tc)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, tc=self.tc, mlp_ratio=mlp_ratio, dropout=dropout, train=False) for _ in range(depth)])
        self.final_layer = FinalLayer(patch_size, out_channels, hidden_size, tc=self.tc)
        self.logvar_embed = nn.Embedding(256, 1)
        nn.init.constant_(self.logvar_embed.weight, 0)

    def forward(self, x, t, dt, y, train=False, return_activations=False):
        activations = {}
        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size

        if self.ignore_dt:
            dt = torch.zeros_like(t)

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = x.to(self.dtype)

        te = self.timestep_embedder(t)
        dte = self.timestep_embedder(dt)
        ye = self.label_embedder(y)
        c = te + ye + dte

        activations['patch_embed'] = x
        activations['pos_embed'] = self.pos_embed
        activations['time_embed'] = te
        activations['dt_embed'] = dte
        activations['label_embed'] = ye
        activations['conditioning'] = c

        for i, block in enumerate(self.blocks):
            x = block(x, c)
            activations[f'dit_block_{i}'] = x

        x = self.final_layer(x, c)
        activations['final_layer'] = x

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=num_patches_side, w=num_patches_side, p1=self.patch_size, p2=self.patch_size)
        assert x.shape == (batch_size, input_size, input_size, self.out_channels)

        t_discrete = (t * 256).long()
        logvars = self.logvar_embed(t_discrete) * 100

        if return_activations:
            return x, logvars, activations
        return x
