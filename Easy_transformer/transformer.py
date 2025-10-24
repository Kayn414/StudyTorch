import einops
from fancy_einsum import einsum
from dataclasses import dataclass
from easy_transformer import EasyTransformer
import torch
import torch.nn as nn
import numpy as np
import math
from easy_transformer.utils import get_corner, gelu_new, tokenize_and_concatenate
import tqdm.auto as tqdm



reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randn(shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    random_input = torch.randint(100, 1000, shape).cuda()
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output


def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    if isinstance(input_name, str): 
        reference_input = cache_dict[input_name]
    else:
        reference_input = input_name
    print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    print("Output shape:", output.shape)
    reference_output = gpt2_layer(reference_input)
    print("Reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct")
    return output


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual):
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + self.cfg.layer_norm_eps).sqrt()
        norm = residual / scale
        norm = norm * self.w + self.b
        if self.cfg.debug: 
            print('Normalized:', norm.shape)
        return norm


class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        if self.cfg.debug: 
            print('tokens:', tokens.shape)
        embed = self.W_E[tokens, :]
        if self.cfg.debug: 
            print('embed:', embed.shape)
        return embed


class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        pos_embed = self.W_pos[:seq_len, :]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=batch)
        if self.cfg.debug: 
            print('pos_embed:', pos_embed.shape)
        return pos_embed


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device='cuda'))

    def forward(self, normalized_resid_pre):
        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        attention_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attention_scores = attention_scores / math.sqrt(self.cfg.d_head)
        attention_scores = self.apply_causal_mask(attention_scores)
        pattern = attention_scores.softmax(dim=-1)
        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V
        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)
        output = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
        if self.cfg.debug: 
            print('attn output:', output.shape)
        return output

    def apply_causal_mask(self, attention_scores):
        mask = torch.triu(torch.ones(attention_scores.size(-2), attention_scores.size(-1), device=attention_scores.device), diagonal=1).bool()
        attention_scores.masked_fill_(mask, self.IGNORE)
        return attention_scores


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        if self.cfg.debug: 
            print('mlp_out:', mlp_out.shape)
        return mlp_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)
        resid_mid = resid_pre + attn_out
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        if self.cfg.debug: 
            print('resid_post:', resid_post.shape)
        return resid_post


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        if self.cfg.debug: 
            print('logits:', logits.shape)
        return logits


class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        if self.cfg.debug: 
            print('Initial residual:', residual.shape)
        for i, block in enumerate(self.blocks):
            if self.cfg.debug: 
                print(f'\n--- Block {i} ---')
            residual = block(residual)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        return logits


def main():
    print("=" * 80)
    print("Testing DemoTransformer Implementation")
    print("=" * 80)
    
    cfg = Config(debug=True)
    print(f"\nConfiguration:")
    print(f"  d_model: {cfg.d_model}")
    print(f"  n_heads: {cfg.n_heads}")
    print(f"  n_layers: {cfg.n_layers}")
    print(f"  d_mlp: {cfg.d_mlp}")
    print(f"  d_vocab: {cfg.d_vocab}")
    print(f"  n_ctx: {cfg.n_ctx}")
    
    print("\n" + "=" * 80)
    print("Initializing Model...")
    print("=" * 80)
    model = DemoTransformer(cfg).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    batch_size = 2
    seq_len = 35
    tokens = torch.randint(0, cfg.d_vocab, (batch_size, seq_len)).cuda()
    
    print("\n" + "=" * 80)
    print("Forward Pass")
    print("=" * 80)
    print(f"\nInput tokens shape: {tokens.shape}")
    
    with torch.no_grad():
        logits = model(tokens)
    
    print(f"\n" + "=" * 80)
    print("Output Summary")
    print("=" * 80)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {cfg.d_vocab}]")
    
    predictions = logits.argmax(dim=-1)
    print(f"\nPredicted token IDs shape: {predictions.shape}")
    print(f"Sample predictions (first sequence, first 10 tokens): {predictions[0, :10].tolist()}")
    
    print("\n" + "=" * 80)
    print("Testing with different sequence length")
    print("=" * 80)
    
    cfg_test = Config(debug=False)
    model_test = DemoTransformer(cfg_test).cuda()
    
    test_seq_len = 50
    test_tokens = torch.randint(0, cfg.d_vocab, (1, test_seq_len)).cuda()
    print(f"Input shape: {test_tokens.shape}")
    
    with torch.no_grad():
        test_logits = model_test(test_tokens)
    print(f"Output shape: {test_logits.shape}")
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()