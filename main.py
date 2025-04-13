from __future__ import annotations

HEB = "heb"
SHAKE = "shake"

if __name__ == '__main__':
    import torch
    from torch import nn
    from torch import optim
    from model.transformer import TransformerLM
    from utils import data, lm
    import json
    import os

    # Change file_type to HEB or SHAKE to train on the Hebrew or Shakespeare dataset
    file_type = HEB

    with open(f"configs/config_{file_type}.json", "r") as f:
        config = json.load(f)
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    data_path = config["data_path"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_size = config["embed_size"]
    mlp_hidden_size = embed_size * 4

    checkpoint_dir = './checkpoints/'
    save_every = 1000


    def save_checkpoint(model, optimizer, n_training_seqs, loss_history):
        state = {
            'n_training_seqs': n_training_seqs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history
        }
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, checkpoint_dir + file_type + ".pth")


    def load_checkpoint(model, optimizer):
        checkpoint = torch.load(checkpoint_dir + file_type + ".pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        n_training_seqs = checkpoint['n_training_seqs']
        loss_history = checkpoint['loss_history']
        return model, optimizer, n_training_seqs, loss_history


    learning_rate = config["learning_rate"]
    gradient_clipping = config["gradient_clipping"]
    weight_decay = config["weight_decay"]

    num_batches_to_train = config["num_batches_to_train"]

    if os.path.exists(f"tokenizer_{file_type}.json"):  # Load tokenizer if it exists
        tokenizer, tokenized_data = data.load_data(data_path, load_file_type=file_type)
    else:  # Otherwise, create and save it
        tokenizer, tokenized_data = data.load_data(data_path)
        tokenizer.save(f"tokenizer_{file_type}.json")
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
        device=device
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95], weight_decay=weight_decay)

    # Load checkpoint if exists
    n_training_seqs = 0
    loss_history = []
    if os.path.exists(checkpoint_dir + file_type + ".pth"):
        model, optimizer, n_training_seqs, loss_history = load_checkpoint(model, optimizer)

    model.train()

    num_batches = 0
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train: break
            num_batches = num_batches + 1
            n_training_seqs += batch.shape[0]

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # check if at least one sample contains padding token
            if torch.any(batch_x == tokenizer.pad_id()):
                print("NOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            logits = model(batch_x)

            # print(logits.shape)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1
            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                loss_history.append(loss.item())
                if num_batches % 1000 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.better_sample_continuation(tokenizer.tokenize("Hello"), 500,
                                                             temperature=0.5,
                                                             topK=5))
                            # model.sample_continuation(tokenizer.tokenize("Hello"), 500))
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")

            # Save checkpoint
            if num_batches % save_every == 0:
                save_checkpoint(model, optimizer, n_training_seqs, loss_history)
