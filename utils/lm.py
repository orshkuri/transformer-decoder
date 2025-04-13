from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    b, n = batch.size()
    inputs = batch[:, :n - 1]  # x1, ..., x_n-1
    labels = batch[:, 1:]  # x2, ..., x_n
    return (inputs, labels)


def compute_loss(logits, gold_labels):
    # logits size is (batch, seq_len, vocab_size)
    # gold_labels size is (batch, seq_len)
    b, n, v = logits.size()

    # Reshape logits to (batch * seq_len, vocab_size)
    logits = logits.reshape(-1, v)

    # Reshape gold_labels to (batch * seq_len)
    gold_labels = gold_labels.reshape(-1)

    # Compute the loss and ignore padding index 0
    loss = F.cross_entropy(logits, gold_labels, ignore_index=0)

    return loss


# def compute_loss(logits, gold_labels):
#     # logits size is (batch, seq_len, vocab_size)
#     # gold_bales size is (batch, seq_len)
#     # NOTE remember to handle padding (ignore them in loss calculation!)
#     # NOTE cross-entropy expects other dimensions for logits
#     # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.
#     b, n, v = logits.size()
#     # print(f"logits: {logits.size()}")
#     # print(f"gold_labels: {gold_labels.size()}")
#     logits = logits.permute(0, 2, 1)
#     # logits = logits.reshape(-1, v)
#     # gold_labels = gold_labels.reshape(-1)
#
#     loss = F.cross_entropy(logits, gold_labels, ignore_index=0)
#     # print(f"loss: {loss}")
#
#     return loss

