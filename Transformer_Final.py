import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("/kaggle/input/datasets/sahajs6200/hinglish/Combined.csv")
df = df.dropna(subset=["processed_text"])

# =========================
# 2. Label Encoding
# =========================
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["label"].map(label_map)

# =========================
# 3. NOVELTY #1: Token Augmentation (existing)
# =========================
def transform_text(text):
    words = text.split()
    new_words = []
    for w in words:
        parts = w.split("_")
        if len(parts) == 2:
            new_words.append(parts[0])
            new_words.append("<" + parts[1] + ">")
        else:
            new_words.append(w)
    return " ".join(new_words)

df["processed_text"] = df["processed_text"].apply(transform_text)

# =========================
# 4. Train-Test Split
# =========================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["processed_text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# =========================
# 5. Tokenizer + Special Tokens
# =========================
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

special_tokens = ["<EN>", "<HI>", "<POS>", "<NEG>"]
tokenizer.add_tokens(special_tokens)

# Map special token string → token ID (used later for gate + warm init)
special_token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens}
print("Special token IDs:", special_token_ids)

# =========================
# 6. Tokenization
# =========================
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=64)
test_encodings  = tokenizer(list(test_texts),  truncation=True, padding=True, max_length=64)

# =========================
# 7. Dataset Class
# =========================
class HinglishDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HinglishDataset(train_encodings, train_labels)
test_dataset  = HinglishDataset(test_encodings,  test_labels)

# =====================================================================
# 8. NOVELTY #2: Warm-Initialized Sentiment Token Embeddings
#    Seed words for each special token — averaged into its embedding
# =====================================================================
SEED_WORDS = {
    "<POS>": ["good", "great", "excellent", "happy", "love", "wonderful", "amazing", "fantastic"],
    "<NEG>": ["bad",  "terrible", "awful", "hate", "horrible", "disgusting", "worst", "sad"],
    "<EN>":  ["the", "is", "and", "this", "that", "with", "from", "about"],
    "<HI>":  ["yaar", "bhai", "acha", "theek", "nahi", "hai", "kya", "mera"],
}

def warm_init_special_tokens(model, tokenizer, seed_words_map):
    """
    Replace the random embeddings of special tokens with the mean
    of their corresponding seed-word embeddings.
    """
    embedding_layer = model.embeddings.word_embeddings
    with torch.no_grad():
        for special_tok, seeds in seed_words_map.items():
            seed_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(" ".join(seeds))
            )
            # Filter out unknowns
            seed_ids = [sid for sid in seed_ids if sid != tokenizer.unk_token_id]
            if not seed_ids:
                continue
            seed_embeddings = embedding_layer.weight[seed_ids]   # (N, hidden)
            mean_embedding  = seed_embeddings.mean(dim=0)        # (hidden,)

            special_id = tokenizer.convert_tokens_to_ids(special_tok)
            embedding_layer.weight[special_id] = mean_embedding

    print("✅ Warm initialization of special token embeddings complete.")

# =====================================================================
# 9. NOVELTY #3: Language-Aware Attention Gating Model
#
#    Architecture:
#      XLM-RoBERTa encoder
#           ↓  (all token hidden states)
#      Lang-Gate MLP  ← detects <EN>/<HI> tag positions → scalar gate per token
#           ↓  (gated weighted pooling)
#      Gated Sentence Representation
#           ↓
#      Linear Classifier (3 classes)
# =====================================================================
class LanguageGatedSentimentModel(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels,
                 en_token_id, hi_token_id):
        super().__init__()
        self.encoder      = encoder
        self.num_labels   = num_labels
        self.en_token_id  = en_token_id
        self.hi_token_id  = hi_token_id

        # Gate MLP: takes [token_hidden ; lang_flag] → scalar gate score
        # lang_flag is a 2-dim one-hot: [is_EN, is_HI]
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),           # one scalar per token
            nn.Sigmoid()                 # gate ∈ (0, 1)
        )

        # Final classifier on gated pooled representation
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # ── Encoder ──────────────────────────────────────────────────
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state   # (B, T, H)

        # ── Build language flag for each token position ───────────────
        # Shape: (B, T, 2) — [is_EN_tag, is_HI_tag]
        is_en = (input_ids == self.en_token_id).float().unsqueeze(-1)   # (B,T,1)
        is_hi = (input_ids == self.hi_token_id).float().unsqueeze(-1)   # (B,T,1)
        lang_flags = torch.cat([is_en, is_hi], dim=-1)                  # (B,T,2)

        # ── Gate MLP ─────────────────────────────────────────────────
        gate_input  = torch.cat([hidden_states, lang_flags], dim=-1)    # (B,T,H+2)
        gate_scores = self.gate_mlp(gate_input).squeeze(-1)             # (B,T)

        # Mask out padding tokens
        gate_scores = gate_scores * attention_mask.float()              # (B,T)

        # ── Gated Weighted Pooling ────────────────────────────────────
        # Normalise gate scores over non-padding positions
        gate_weights = gate_scores / (gate_scores.sum(dim=1, keepdim=True) + 1e-9)
        gated_repr   = (hidden_states * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B,H)

        # ── Classification ────────────────────────────────────────────
        logits = self.classifier(self.dropout(gated_repr))              # (B, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        # Return a simple namespace so HuggingFace Trainer is happy
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

# =========================
# 10. Instantiate Model
# =========================
base_encoder = AutoModel.from_pretrained("xlm-roberta-base")
base_encoder.resize_token_embeddings(len(tokenizer))

model = LanguageGatedSentimentModel(
    encoder      = base_encoder,
    hidden_size  = 768,
    num_labels   = 3,
    en_token_id  = special_token_ids["<EN>"],
    hi_token_id  = special_token_ids["<HI>"]
)

# Apply warm initialization AFTER model is built
warm_init_special_tokens(base_encoder, tokenizer, SEED_WORDS)

# =========================
# 11. Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# =========================
# 12. Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir                  = "./results",
    num_train_epochs            = 4,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    learning_rate               = 2e-5,
    logging_steps               = 100,
    load_best_model_at_end      = True
)

# =========================
# 13. Trainer
# =========================
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = test_dataset,
    compute_metrics = compute_metrics
)

# =========================
# 14. Train
# =========================
trainer.train()

# =========================
# 15. Evaluate
# =========================
results = trainer.evaluate()
print(results)

# =========================
# 16. Confusion Matrix
# =========================
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)
cm = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix:\n", cm)

# =========================
# 17. Error Analysis
# =========================
for i in range(len(test_texts)):
    if y_pred[i] != test_labels.iloc[i]:
        print("Text:",      test_texts.iloc[i])
        print("Actual:",    test_labels.iloc[i])
        print("Predicted:", y_pred[i])
        print("------")