# 🧠 Transformer‑based Text Generation

A custom‑built **Transformer decoder** model for text generation, trained on **Modern Hebrew** and **Shakespearean English**. This project was implemented entirely from scratch as part of a graduate‑level Deep Learning course, showcasing a deep understanding of attention mechanisms, language modeling, and sequence generation.

---

## 🚀 Features

- **From‑scratch Transformer decoder** (no HuggingFace, no external frameworks)
- **Multilingual training**: Modern Hebrew and Shakespearean English
- Implements:
  - Causal self‑attention  
  - Tokenization and padding  
  - Sampling with temperature and top‑k  
  - Checkpointing and resume training
- Utility modules for loss computation, batching, data handling, and visualization

---

## 🗂️ Project Structure

```text
├── model/                  # Transformer architecture
│   ├── transformer.py      # Decoder block & forward pass
│   ├── attention.py        # Multi‑head causal self‑attention
│   └── mlp.py              # Position‑wise feed‑forward network
├── utils/                  # Training pipeline & helpers
│   ├── data.py             # Dataset loading & preprocessing
│   ├── lm.py               # Language‑model training loop
│   └── heatmaps.py         # Attention‑weight visualization
├── configs/                # Hyper‑parameter configs
│   ├── config_heb.json     # Hebrew dataset config
│   └── config_shake.json   # Shakespeare dataset config
├── checkpoints/            # Saved model weights
├── heb-data/               # Hebrew corpus
├── data/                   # Additional corpora
│   └── input.txt
├── tokenizer_*.json        # Saved tokenizers
├── main.py                 # Entry point for training
├── report.pdf              # Final project report
├── transformers-assignment.pdf  # Course assignment
└── req.txt                 # Python dependencies
```

---

## 📦 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/transformer-textgen.git
   cd transformer-textgen
   ```

2. **Install the requirements**

   ```bash
   pip install -r req.txt
   ```

### 🧰 Requirements

Minimal working setup (excerpt from `req.txt`):

```text
torch>=1.10
numpy
```

Optional utilities:

```text
tqdm      # progress bars
```

> 💡 *Tip:* After creating your environment, run `pip freeze > req.txt` to save the full package list.

---

## 🏃‍♂️ How to Train

Edit `main.py` to choose a dataset:

```python
file_type = "heb"   # or "shake"
```

Then start training:

```bash
python main.py
```

Checkpoints are saved automatically to `./checkpoints/` every *N* steps. Training will resume from the latest checkpoint if one is found.

---

## 📖 Sample Output

### Shakespeare (early epoch)

```text
Shall I compare thee to a summer’s day?
Thou art more lovely and more temperate.
```

### Hebrew (early epoch)

```text
היום הזה הוא התחלה חדשה מלאה באור ותקווה.
```

---

## 📚 Acknowledgments

This project was completed as part of a graduate‑level Deep Learning course. The code emphasizes educational value, reproducibility, and a bottom‑up understanding of Transformer models.

---

## 🧾 License

This repository is provided for **educational and academic use only**.

---

## 🤝 Contributions

Issues and pull requests are welcome! If you discover bugs or have suggestions, feel free to open an issue or submit a PR.
