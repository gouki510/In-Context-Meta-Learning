import matplotlib.pyplot as plt


def visalize_attention1(model):
    attn_matrix = model.get_attention_matrix1()
    num_heads = attn_matrix.size(0)
    fig, ax = plt.subplots(num_heads, 1, figsize=(5*num_heads, 5))
    if num_heads == 1:
        ax = [ax]
    for j in range(num_heads):
        ax[j].imshow(attn_matrix[j].detach().cpu().numpy(), cmap="Blues")
    plt.tight_layout()
    return fig

def visalize_attention2(model):
    attn_matrix = model.get_attention_matrix2()
    num_heads = attn_matrix.size(0)
    fig, ax = plt.subplots(num_heads, 1, figsize=(5*num_heads, 5))
    if num_heads == 1:
        ax = [ax]
    for j in range(num_heads):
        ax[j].imshow(attn_matrix[j].detach().cpu().numpy(), cmap="Blues")
    plt.tight_layout()
    return fig
