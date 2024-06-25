import matplotlib.pyplot as plt
import torch

def visalize_attention(model, layer_i):
    attn_matrix = model.get_attention_matrix(layer_i)
    num_heads = attn_matrix.size(0)
    fig, ax = plt.subplots(num_heads, 1, figsize=(5*num_heads, 5))
    if num_heads == 1:
        ax = [ax]
    for j in range(num_heads):
        ax[j].imshow(attn_matrix[j].detach().cpu().numpy(), cmap="Blues")
    plt.tight_layout()
    return fig

def example_label_extract_attention(model, layer_i,  n_ctx=8):
    attn_matrix = model.get_attention_matrix(layer_i)
    num_heads = attn_matrix.size(0)
    example_idx = torch.arange(0, n_ctx, 2).unsqueeze(0)
    label_idx = torch.arange(1, n_ctx, 2).unsqueeze(0)
    log = {}
    for j in range(num_heads):
        attn = attn_matrix[j].detach().cpu()
        label_attn = attn[:, label_idx].detach().cpu().numpy()[:, 0, :].mean(axis=1)
        ex_attn = attn[:, example_idx].detach().cpu().numpy()[:, 0, :].mean(axis=1)
        # print(label_attn.shape, ex_attn.shape)
        for i in range(n_ctx):
            log[f"atten_value/layer_{layer_i}_head_{j}_token_{i}_from_label"] = label_attn[i]
            log[f"atten_value/layer_{layer_i}_head_{j}_token_{i}_from_ex"] = ex_attn[i]
    # wandb_log(log)
    return log
        
def cal_entropy_attention(model, layer_i):
    attn_matrix = model.get_attention_matrix(layer_i)
    num_heads = attn_matrix.size(0)
    entropy = []
    for j in range(num_heads):
        attn = attn_matrix[j].detach().cpu()
        entropy.append(-torch.sum(attn * torch.log(attn + 1e-10)))
    return torch.stack(entropy).mean()