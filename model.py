

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_vocab, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum("pe,bse->bsp", self.W, x)


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir="fwd"):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)

        if dir == "fwd":
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == "bwd":
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir="fwd"):
        if (dir == "fwd") or (dir == "both"):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == "bwd") or (dir == "both"):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model, weight_scale=1):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) * weight_scale)

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x


# Attention
class Attention(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size (113 or 3)
    i : number of heads
    h : embedding size of each heads
    n_ctx : token size
    """

    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_Q = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_V = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_O = nn.Parameter(
            torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model)
        )
        self.register_buffer("mask", torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (
            1 - self.mask[: x.shape[-2], : x.shape[-2]]
        )
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = torch.einsum("biph,biqp->biqh", v, attn_matrix)
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = torch.einsum("df,bqf->bqd", self.W_O, z_flat)
        return out


class Dense(nn.Module):
    def __init__(self, d_in, d_out, act_type, weight_scale=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in))
        torch.nn.init.normal_(self.W, mean=0, std=weight_scale / np.sqrt(d_in))

    def set_weight_ratio(self, weight_ratio):
        self.W = nn.Parameter(self.W * weight_ratio)

    def set_weight_ratio_l2(self, weight_ratio):
        self.W = nn.Parameter(self.W * torch.sqrt(weight_ratio))

    def forward(self, x):
        return x @ self.W.T


# for Transformer
class MLPBlock(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size (114 or 3)
    i : number of heads
    h : embedding size of each heads
    """

    def __init__(self, d_model, d_mlp, act_type):
        super().__init__()
        # bias & layer norm are removed.
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        assert act_type in ["ReLU", "GeLU"]

    def forward(self, x):
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        if self.act_type == "ReLU":
            x = F.relu(x)
        elif self.act_type == "GeLU":
            x = F.gelu(x)
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out
        return x

    def set_weight_ratio(self, weight_ratio):
        self.W_in = nn.Parameter(self.W_in * weight_ratio)
        self.W_out = nn.Parameter(self.W_out * weight_ratio)


class TransformerBlock(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size
    i : number of heads
    h : embedding size of each heads
    """

    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLPBlock(d_model, d_mlp, act_type)
        self.layer_norm = LayerNorm(d_model, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_mid(
            x + self.hook_attn_out(self.attn((self.hook_resid_pre(x))))
        )
        x = self.layer_norm(x)
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x

    def set_weight_ratio(self, weight_ratio):
        self.attn.set_weight_ratio(weight_ratio)
        self.mlp.set_weight_ratio(weight_ratio)


class InputEmbedder(nn.Module):
    """Input embedder."""

    def __init__(self, conf):

        """Initialize the input embedder.

    Args:
      num_classes: Total number of output classes.
      emb_dim: Dimensionality of example and label embeddings.
      example_encoding: How to encode example inputs.
        'resnet': simple resnet encoding
        'linear': flatten and pass through a linear layer
        'embedding': pass through an embedding layer
      flatten_superpixels: Whether to flatten the output of the resnet (instead
        of taking a mean over superpixels).
      example_dropout_prob: Dropout probability on example embeddings. Note that
        these are applied at both train and test.
      concatenate_labels: Whether to concatenate example and label embeddings
        into one token for each (example, label) pair, rather than being fed to
        the transformer as two separate tokens.
      use_positional_encodings: Whether to use positional encoding.
      positional_dropout_prob: Positional dropout probability.
      name: Optional name for the module.
    """
        super(InputEmbedder, self).__init__()
        self._num_classes = conf.d_vocab
        self._emb_dim = conf.d_emb
        self.p_dim = conf.p_dim

        self.Emb = nn.Linear(self._emb_dim, self._emb_dim)

        self.label_embs = nn.Parameter(
            torch.randn(self._num_classes, self._emb_dim)
        )

    def forward(self, examples, labels, is_training=True):
        """Call to the input embedder.

        Args:
          examples: input sequence of shape
            [batch_size, seq_len, height, width, channels]
          labels: input sequence of shape [batch_size, seq_len]
          is_training: if is currently training.

        Returns:
          outputs: output of the transformer tower
            of shape [batch_size, seq_len, channels].
        """
        # Encode the example inputs into shape (B, SS, E)
        B, SS, D = examples.shape
        # pos encoding
        pos_enc = F.one_hot(torch.arange(SS), num_classes=self.p_dim).repeat(B,1,1).to(examples.device)
        h_example = torch.cat([examples, pos_enc], dim=2)

        # Embed the labels.
        n_emb_classes = self._num_classes
        labels_to_embed = labels
        embs = self.get_parameter("label_embs")
        h_label = embs[labels_to_embed]  # (B, SS, E)
        hh = torch.empty(
            (h_example.shape[0], h_example.shape[1] * 2 - 1, h_example.shape[2]),
            dtype=h_example.dtype,
        ).to(h_example.device)
        hh[:, 0::2] = h_example
        hh[:, 1::2] = h_label[:, :-1]

        return hh


class Transformer(nn.Module):
    def __init__(self, embedder, config):
        super().__init__()
        num_layers = config.num_layers
        d_model = config.d_emb
        d_mlp = config.d_emb * 4
        d_head = config.d_emb // config.num_heads
        num_heads = config.num_heads
        n_ctx = config.n_ctx
        act_type = config.act_type
        use_cache = config.use_cache
        use_ln = config.use_ln
        self.cache = {}
        self.use_cache = use_cache
        d_vocab = config.d_vocab

        self.embedder = embedder
        # self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]
                )
                for i in range(num_layers)
            ]
        )
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    def forward(self, x, labels):
        x = self.embedder(x, labels,)
        # x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules() if "hook" in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks("fwd")
            hp.remove_hooks("bwd")

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")
            if incl_bwd:
                hp.add_hook(save_hook_back, "bwd")

class TransformerICL(nn.Module):
    def __init__(self, embedder, config):
        super().__init__()
        num_layers = config.num_layers
        d_model = config.d_emb
        d_mlp = config.d_emb * 4
        d_head = config.d_emb // config.num_heads
        num_heads = config.num_heads
        n_ctx = config.n_ctx
        act_type = config.act_type
        use_cache = config.use_cache
        use_ln = config.use_ln
        self.cache = {}
        self.use_cache = use_cache
        d_vocab = config.d_vocab

        self.embedder = embedder
        # self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList(
            [
                Attention(d_model, num_heads, d_head, n_ctx),
                Attention(d_model, num_heads, d_head, n_ctx),
                Dense(d_model, d_model, act_type),
                Dense(d_model, d_model, act_type),
                Dense(d_model, d_model, act_type),
            ]
        )
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    def forward(self, x, labels):
        x = self.embedder(x, labels)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules() if "hook" in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks("fwd")
            hp.remove_hooks("bwd")

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")
            if incl_bwd:
                hp.add_hook(save_hook_back, "bwd")