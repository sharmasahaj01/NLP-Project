from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import json, re
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from wordfreq import zipf_frequency
import jellyfish
from rapidfuzz import fuzz

app = Flask(__name__)
CORS(app)

# ================================================================
# PREPROCESSING PIPELINE (same as your dataset pipeline)
# ================================================================

# ── Step 1: Normalize repeated chars ────────────────────────────
def normalize_repeated_chars(word):
    return re.sub(r'(.)\1{2,}', r'\1\1', word)

# ── Step 2: Tokenize ─────────────────────────────────────────────
def tokenize_text(text):
    return str(text).split()

# ── Step 3: Normalize tokens ─────────────────────────────────────
def normalize_tokens(tokens):
    return [normalize_repeated_chars(w) for w in tokens]

# ── Step 4: Language tagging ─────────────────────────────────────
def detect_language(word):
    word = word.lower()
    if len(word) <= 2:
        return word + "_HI"
    freq = zipf_frequency(word, "en")
    return word + "_EN" if freq > 3 else word + "_HI"

def language_tagging(tokens):
    return [detect_language(w) for w in tokens]

# ── Step 5: Hinglish sentiment tagging ───────────────────────────
HINGLISH_SENTIMENT = {
    "mast": "POS", "jhakaas": "POS", "bindaas": "POS",
    "sahi": "POS", "badhiya": "POS", "achha": "POS",
    "acha": "POS", "accha": "POS",
    "bakwas": "NEG", "faltu": "NEG", "bekaar": "NEG",
    "bekar": "NEG", "ghatiya": "NEG", "kharab": "NEG"
}

def slang_sentiment_tagging(tokens):
    tagged = []
    for word in tokens:
        base = word.split("_")[0]
        if base in HINGLISH_SENTIMENT:
            tagged.append(base + "_" + HINGLISH_SENTIMENT[base])
        else:
            tagged.append(word)
    return tagged

# ── Step 6: Stopword removal ──────────────────────────────────────
def load_stopwords(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        print(f"⚠️  Stopwords file not found at {file_path}, skipping.")
        return set()

STOPWORDS = load_stopwords(".././Files/hinglish_stop.txt")

def remove_stopwords(tokens):
    return [t for t in tokens if t.split("_")[0] not in STOPWORDS]

# ── Step 7: Transform text for model (your existing function) ────
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

# ── Full pipeline ─────────────────────────────────────────────────
def full_preprocess(raw_text):
    tokens = tokenize_text(raw_text)
    tokens = normalize_tokens(tokens)
    tokens = language_tagging(tokens)
    tokens = slang_sentiment_tagging(tokens)
    tokens = remove_stopwords(tokens)

    tagged_text    = " ".join(tokens)           # "class_EN environment_EN acha_HI tha_HI"
    model_input    = transform_text(tagged_text) # "class <EN> environment <EN> acha <HI> tha <HI>"

    return tagged_text, model_input

# ================================================================
# MODEL
# ================================================================
class LanguageGatedSentimentModel(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels, en_token_id, hi_token_id):
        super().__init__()
        self.encoder     = encoder
        self.num_labels  = num_labels
        self.en_token_id = en_token_id
        self.hi_token_id = hi_token_id
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs       = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        is_en         = (input_ids == self.en_token_id).float().unsqueeze(-1)
        is_hi         = (input_ids == self.hi_token_id).float().unsqueeze(-1)
        lang_flags    = torch.cat([is_en, is_hi], dim=-1)
        gate_input    = torch.cat([hidden_states, lang_flags], dim=-1)
        gate_scores   = self.gate_mlp(gate_input).squeeze(-1)
        gate_scores   = gate_scores * attention_mask.float()
        gate_weights  = gate_scores / (gate_scores.sum(dim=1, keepdim=True) + 1e-9)
        gated_repr    = (hidden_states * gate_weights.unsqueeze(-1)).sum(dim=1)
        logits        = self.classifier(self.dropout(gated_repr))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

# ── Load model ────────────────────────────────────────────────────
print("Loading model...")
with open(".././hinglish_model/config.json") as f:
    config = json.load(f)

tokenizer    = AutoTokenizer.from_pretrained(".././hinglish_model/tokenizer")
base_encoder = AutoModel.from_pretrained(".././hinglish_model/encoder")
base_encoder.resize_token_embeddings(len(tokenizer))

model = LanguageGatedSentimentModel(
    encoder      = base_encoder,
    hidden_size  = config["hidden_size"],
    num_labels   = config["num_labels"],
    en_token_id  = config["en_token_id"],
    hi_token_id  = config["hi_token_id"]
)
model.load_state_dict(torch.load(".././hinglish_model/model_weights.pt", map_location="cpu"))
model.eval()
print("✅ Model loaded!")

# ================================================================
# API ENDPOINTS
# ================================================================
@app.route('/predict', methods=['POST'])
def predict():
    raw_text = request.json.get('text', '').strip()
    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    # Run full preprocessing pipeline
    tagged_text, model_input = full_preprocess(raw_text)

    # Tokenize for model
    inputs = tokenizer(
        model_input, return_tensors="pt",
        truncation=True, padding=True, max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs  = torch.softmax(outputs.logits, dim=1)[0]
    labels = ["negative", "neutral", "positive"]
    scores = {l: round(probs[i].item() * 100, 1) for i, l in enumerate(labels)}
    pred   = max(scores, key=scores.get)

    return jsonify({
        "sentiment":   pred,
        "confidence":  scores[pred],
        "scores":      scores,
        "tagged_text": tagged_text,    # ← "class_EN environment_EN acha_HI tha_HI"
        "model_input": model_input     # ← "class <EN> environment <EN> acha <HI> tha <HI>"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(port=5000, debug=False)