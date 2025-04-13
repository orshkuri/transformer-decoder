from __future__ import annotations
import torch
from torch import optim
from model.transformer import TransformerLM
from utils import data
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

HEB = "heb"
SHAKE = "shake"

# Change file_type to HEB or SHAKE to train on the Hebrew or Shakespeare dataset
file_type = HEB


def load_checkpoint():
    with open(f"config_{file_type}.json", "r") as f:
        config = json.load(f)
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    data_path = config["data_path"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_size = config["embed_size"]
    mlp_hidden_size = embed_size * 4

    checkpoint_dir = './checkpoints/'

    learning_rate = config["learning_rate"]
    gradient_clipping = config["gradient_clipping"]
    weight_decay = config["weight_decay"]

    num_batches_to_train = config["num_batches_to_train"]

    tokenizer, tokenized_data = data.load_data(data_path, load_file_type=file_type)

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
        device=device,
        return_attention=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95], weight_decay=weight_decay)
    if device == torch.device('cpu'):
        checkpoint = torch.load(checkpoint_dir + file_type + ".pth", map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_dir + file_type + ".pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    n_training_seqs = checkpoint['n_training_seqs']
    loss_history = checkpoint['loss_history']
    return model, optimizer, n_training_seqs, loss_history, tokenized_data, tokenizer, seq_len


def plot_attention_matrix(attention_matrix, words, layer, head):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a heatmap with the attention matrix
    sns.heatmap(attention_matrix, xticklabels=words, yticklabels=words, cmap='gray', ax=ax)

    # Rotate the x-axis labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title(f'Layer {layer + 1}, Head {head + 1}')

    # Display the plot
    plt.tight_layout()
    if not os.path.exists(f'heatmaps/{file_type}'):
        os.makedirs(f'heatmaps/{file_type}')
    if not os.path.exists(f'heatmaps/{file_type}/layer_{layer}'):
        os.makedirs(f'heatmaps/{file_type}/layer_{layer}')
    plt.savefig(f'heatmaps/{file_type}/layer_{layer}/head_{head}.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, n_training_seqs, loss_history, tokenized_data, tokenizer, seq_len = load_checkpoint()
    # get amount of parameters
    n_params = sum(p.numel() for p in model.parameters())

    print("Parameter count: %.2fM" % (n_params / 1e6))
    print(f"n_training_seqs: {n_training_seqs}")
    print(f"final loss: {loss_history[-1]}")

    plt.plot(np.arange(len(loss_history)) * 10, loss_history, label='loss')
    plt.title('Loss history')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    # plt.savefig(f'plots/loss_history_{file_type}.png')

    sampled = tokenizer.detokenize(
        model.better_sample_continuation(tokenizer.tokenize("אמר "), 48,
                                         temperature=0.5,
                                         topK=5))
    print(sampled)
    tokens = tokenizer.tokenize(sampled)
    input_ = torch.tensor([tokens], dtype=torch.long)
    print(f'amount of tokens: {input_.shape}')
    print(f'tokens: {input_}')

    logits, attentions_per_layer = model(input_)
    print(f'attention shape: {attentions_per_layer.shape}')
    print(f'Amount of layers: {len(attentions_per_layer)}')

    # Plot the attention matrices
    # for layer in range(len(attentions_per_layer)):
    #     for head in range(len(attentions_per_layer[0])):
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(attentions_per_layer[layer][head][0].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    #         plt.title(f'Attention matrix for layer {layer}, head {head}')
    #         plt.savefig(f'heatmaps/attention_{file_type}_layer_{layer}_head_{head}.png')
    #         plt.close()

    num_layers = attentions_per_layer.shape[0]
    num_heads = attentions_per_layer.shape[1]
    seq_len = attentions_per_layer.shape[3]

    for layer in range(num_layers):
        for head in range(num_heads):
            print(f'Layer {layer}, Head {head}')
            attention_matrix = attentions_per_layer[layer][head][0].detach().cpu().numpy()
            plot_attention_matrix(attention_matrix, sampled, layer, head)
