# =========================
# 1. Imports
# =========================
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv("./../final_dataset2.csv")
df = df.dropna(subset=["processed_text"])

# =========================
# 3. Clean Text for LDA (REMOVE _EN/_HI TAGS)
# =========================
def clean_for_lda(text):
    words = text.split()
    clean_words = []

    for w in words:
        if "_" in w:
            clean_words.append(w.split("_")[0])
        else:
            clean_words.append(w)

    return " ".join(clean_words)

df["lda_text"] = df["processed_text"].apply(clean_for_lda)

# =========================
# 4. Tokenization
# =========================
df["tokens"] = df["lda_text"].apply(lambda x: x.lower().split())

# =========================
# 5. Remove Stopwords
# =========================
df["tokens"] = df["tokens"].apply(
    lambda words: [word for word in words if word not in STOPWORDS and len(word) > 2]
)

# =========================
# 6. Create Dictionary
# =========================
dictionary = corpora.Dictionary(df["tokens"])

# remove rare & very common words
dictionary.filter_extremes(no_below=5, no_above=0.5)

# =========================
# 7. Create Corpus
# =========================
corpus = [dictionary.doc2bow(text) for text in df["tokens"]]

# =========================
# 8. Train LDA Model
# =========================
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,       # try 4–6 if needed
    passes=10,
    random_state=42
)

# =========================
# 9. Print Topics
# =========================
print("\n===== Extracted Topics =====\n")

for idx, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")
    print()

# =========================
# 10. Get Dominant Topic per Feedback (FIXED)
# =========================
def get_dominant_topic(text):
    bow = dictionary.doc2bow(text)

    if len(bow) == 0:
        return -1   # no topic assigned

    topics = lda_model.get_document_topics(bow)

    return max(topics, key=lambda x: x[1])[0]

df["dominant_topic"] = df["tokens"].apply(get_dominant_topic)

# =========================
# 11. Save Output
# =========================
df.to_csv("lda_output_new1.csv", index=False)

print("\nSaved results to lda_output.csv")