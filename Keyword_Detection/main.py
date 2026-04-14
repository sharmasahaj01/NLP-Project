# =========================
# 1. Imports
# =========================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv("./../final_dataset2.csv")
df = df.dropna(subset=["processed_text"])

# =========================
# 3. Clean Text for Keywords (REMOVE _EN/_HI)
# =========================
def clean_text(text):
    words = text.split()
    clean_words = []

    for w in words:
        if "_" in w:
            clean_words.append(w.split("_")[0])
        else:
            clean_words.append(w)

    return " ".join(clean_words)

df["clean_text"] = df["processed_text"].apply(clean_text)

# =========================
# 4. TF-IDF Keyword Extraction
# =========================
vectorizer = TfidfVectorizer(
    max_features=20,
    ngram_range=(1,2),   # includes phrases
    min_df=5
)

X = vectorizer.fit_transform(df["clean_text"])

keywords = vectorizer.get_feature_names_out()

print("\n===== TF-IDF Keywords =====\n")
print(keywords)

# =========================
# 5. Frequency-Based Keywords (Issues)
# =========================
all_words = " ".join(df["clean_text"]).split()

word_freq = Counter(all_words)

top_words = word_freq.most_common(20)

print("\n===== Frequent Words =====\n")
for word, freq in top_words:
    print(word, ":", freq)

# =========================
# 6. Bigram Issues (VERY IMPORTANT)
# =========================
vectorizer_bigram = TfidfVectorizer(
    ngram_range=(2,2),   # only bigrams
    max_features=20,
    min_df=5
)

X_bigram = vectorizer_bigram.fit_transform(df["clean_text"])

bigrams = vectorizer_bigram.get_feature_names_out()

print("\n===== Common Issues (Bigrams) =====\n")
print(bigrams)

# =========================
# 7. Save Results
# =========================
keywords_df = pd.DataFrame({
    "keywords": keywords
})

bigrams_df = pd.DataFrame({
    "issues": bigrams
})

keywords_df.to_csv("keywords.csv", index=False)
bigrams_df.to_csv("issues.csv", index=False)

print("\nSaved keywords and issues to CSV")