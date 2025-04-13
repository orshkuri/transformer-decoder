from __future__ import annotations
from torch import nn
import torch
import torch.nn.functional as F
from model import attention, mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, return_attention: bool = False):
        super().__init__()
        self.return_attention = return_attention
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len, return_attention)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, inputs):
        if self.with_residuals:
            # Residual connection and LayerNorm before the attention layer
            x = inputs
            norm_x = self.layer_norm_1(x)
            if self.return_attention:
                attn_x, attentions = self.causal_attention(norm_x)
            else:
                attn_x = self.causal_attention(norm_x)
            x = x + attn_x  # Apply dropout to the attention output

            # Residual connection and LayerNorm before the MLP layer
            norm_x = self.layer_norm_2(x)
            mlp_x = self.mlp(norm_x)
            x = x + mlp_x  # Apply dropout to the MLP output
        else:
            x = inputs
            x = self.layer_norm_1(x)
            if self.return_attention:
                attn_x, attentions = self.causal_attention(x)
            else:
                x = self.causal_attention(x)
            # TODO: add dropout
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            # TODO: add dropout

        if self.return_attention:
            return x, attentions
        return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size) # TODO set the right values
        self.position_embeddings = nn.Embedding(max_context_len, embed_size) # TODO set the right values
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        b, n = x.size()

        tok_embeddings = self.token_embeddings(x)

        # Create position ids of shape (b x n)
        positions = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        pos_embeddings = self.position_embeddings(positions)

        # # Create a mask for padding tokens
        # pad_mask = (x == 0)
        #
        # embeddings = tok_embeddings + pos_embeddings
        #
        # # Zero out the embeddings of padding tokens
        # embeddings[pad_mask] = 0.0

        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            device=None,
            return_attention: bool = False):
        super().__init__()
        self.return_attention = return_attention
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals, return_attention) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len
        if device is not None:
            self.device = device
        else:
            device = "cpu"

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)

        # list where each element correspond to a layer, and contains all the attention matrices for that layer.
        if self.return_attention:
            attentions_per_layer = []

        for layer in self.layers:
            if self.return_attention:
                x, attentions = layer(x)
                attentions_per_layer.append(attentions)
            else:
                x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)

        if self.return_attention:
            attentions_per_layer = [torch.stack(attentions) for attentions in attentions_per_layer]
            attentions_per_layer = torch.stack(attentions_per_layer)
            return logits, attentions_per_layer
        return logits

    def init_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                torch.nn.init.zeros_(m.bias)
                torch.nn.init.ones_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        print(f"feed_to_lm: {feed_to_lm}")
        print('-------------------')
        print(f"max_tokens_to_generate: {max_tokens_to_generate}")
        print('-------------------')
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32).to(self.device))
                # print(f"logits: {logits}")
                # print('-------------------')
                logits_for_last_token = logits[0][-1]
                # print(f"logits_for_last_token: {logits_for_last_token}")
                # print('-------------------')
                distribution_for_last_token = F.softmax(logits_for_last_token)
                # print(f"distribution_for_last_token: {distribution_for_last_token}")
                # print('-------------------')
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                # print(f"sampled_token: {sampled_token}")
                # print('-------------------')
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int = 5) -> list[int]:
        # TODO implement this.
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.

        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                if self.return_attention:
                    logits, attentions = self(torch.tensor([feed_to_lm], dtype=torch.long).to(self.device))
                else:
                    logits = self(torch.tensor([feed_to_lm], dtype=torch.long).to(self.device))
                # get the logits divided by the temperature
                logits_for_last_token = logits[0, -1] / temperature

                p = F.softmax(logits_for_last_token, dim=-1)
                p_topk, top_k_indices = torch.topk(p, topK)
                sampled_index = torch.multinomial(p_topk, num_samples=1).item()
                sampled_token = top_k_indices[sampled_index]
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated
