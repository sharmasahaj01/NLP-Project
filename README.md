# Hinglish Sentiment Analysis — Linguistically Enhanced Transformer

A transformer-based framework for sentiment analysis on **Hinglish (code-mixed Hindi-English)** text, built on **XLM-RoBERTa** with linguistic enhancements: token augmentation, linguistically-seeded embedding initialization, and a novel **language-aware attention gating** mechanism. The system also extends into topic modeling and keyword extraction to turn raw student feedback into interpretable insights.

📄 **Full Research Report:** [View on Google Drive](https://drive.google.com/file/d/1RcOBuJWnu2wAOBiIaiCfK_7IpDg9ViQy/view?usp=sharing)

---

## 🚀 Highlights

- **83.08% accuracy** on a 34,563-sample Hinglish sentiment dataset — outperforming mBERT (72.89%), XLM-RoBERTa vanilla (78.34%), and MuRIL (81.68%)
- Custom **phonetic + fuzzy-matching transliteration normalization** to handle inconsistent Hinglish spellings (e.g., "acha" / "accha" / "achha")
- **Language-aware attention gating** that dynamically weights tokens based on Hindi/English identity
- Extended pipeline with **LDA topic modeling** and **TF-IDF keyword extraction** for holistic feedback analysis
- Interactive **Flask dashboard** for real-time sentiment visualization

---

## 🏗️ Architecture

```
Raw Hinglish Text
       │
       ▼
Preprocessing & Augmentation
  ├─ Transliteration Normalization (Phonetic + Fuzzy Matching)
  └─ Token Augmentation (<EN>, <HI>, <POS>, <NEG>)
       │
       ▼
XLM-RoBERTa Encoder (Linguistically-Seeded Embeddings)
       │
       ▼
Language-Aware Attention Gating → Weighted Pooling
       │
       ▼
Fully Connected Layer → Softmax
       │
       ▼
Sentiment Output (Positive / Neutral / Negative)
```

---

## 📊 Results

| Model            | Accuracy | F1-score | Precision | Recall |
|------------------|:--------:|:--------:|:---------:|:------:|
| mBERT            | 72.89    | 72.88    | 72.88     | 72.89  |
| XLM-R (Vanilla)  | 78.34    | 78.36    | 78.37     | 78.34  |
| MuRIL            | 81.68    | 81.65    | 81.63     | 81.68  |
| **Our Model**    | **83.08**| **82.95**| **83.07** | **83.08** |

Per-class F1: **Positive 86.1%**, **Negative 84.7%**, **Neutral 78.4%** (neutral remains the hardest class due to overlap with positive/negative cues).

---

## 📁 Repository Structure

```
├── Dashboard/            # Flask-based dashboard for visualization
├── Dataset/              # Raw and processed Hinglish sentiment dataset
├── Files/                # Supporting project files
├── Keyword_Detection/    # TF-IDF based keyword & bigram extraction
├── Topic_extraction/     # LDA-based topic modeling
├── hinglish_model/       # Core model: token augmentation, embeddings, attention gating
├── pre_processing/       # Transliteration normalization (phonetic + fuzzy matching)
├── DataCleaning.py       # Text cleaning & stopword removal
├── DataFormat.py         # Dataset formatting utilities
├── FeatureExtract.py     # Feature extraction pipeline
├── Transformer_Final.py  # Final transformer model training/inference
├── WC.py                 # Word cloud generation
├── transformer-comparison.ipynb   # Baseline vs. proposed model comparison notebook
└── Combined.csv           # Combined dataset file
```

---

## ⚙️ Tech Stack

**Python** · **PyTorch** · **HuggingFace Transformers** · **XLM-RoBERTa** · **Pandas** · **NumPy** · **scikit-learn** · **Flask**

---

## 🧠 Methodology

1. **Preprocessing** — lowercasing, noise removal, Hinglish-specific stopword removal, phonetic/fuzzy transliteration normalization
2. **Token Augmentation** — special `<EN>`, `<HI>`, `<POS>`, `<NEG>` markers injected into the input sequence
3. **Linguistically-Seeded Embeddings** — special token embeddings initialized as the average of semantically relevant seed word embeddings (instead of random init)
4. **Language-Aware Attention Gating** — token representations concatenated with language indicator vectors and passed through a gating network to compute dynamic importance weights, used for weighted pooling before classification
5. **Post-processing** — LDA topic extraction + TF-IDF keyword/bigram analysis for interpretable feedback insights

---

## 📈 Dataset

| Class    | Train | Test | Total  |
|----------|:-----:|:----:|:------:|
| Negative | 8,772 | 2,143| 10,915 |
| Neutral  | 9,673 | 2,442| 12,115 |
| Positive | 9,205 | 2,328| 11,533 |
| **Total**| **27,650** | **6,913** | **34,563** |

80:20 train-test split, evaluated on accuracy, precision, recall, and macro F1-score.

---

## 🔮 Future Work

- Larger / more recent pre-trained transformer backbones
- Improved phonetic and contextual transliteration modeling
- Contextual embeddings for informal Hinglish slang
- Multimodal sentiment analysis (text + image/audio/emoji)
- Aspect-based sentiment analysis (per-teacher, per-topic sentiment)
- Real-time, compressed deployment for large-scale use

---

## 👥 Authors

- Aditya — NSUT — aditya-ug23@nsut.ac.in
- Goutam Jain — NSUT — goutam.jain.ug23@nsut.ac.in
- Sahaj Sharma — NSUT — sahaj.sharma.ug23@nsut.ac.in

Netaji Subhas University of Technology (NSUT), Delhi, India

---

## 📄 Citation / Report

Full methodology, mathematical formulation, and detailed results are available in the project report:
👉 [Read the full report here](https://drive.google.com/file/d/1RcOBuJWnu2wAOBiIaiCfK_7IpDg9ViQy/view?usp=sharing)
