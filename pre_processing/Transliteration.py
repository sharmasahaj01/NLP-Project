import re
import pandas as pd
from rapidfuzz import fuzz
import jellyfish
from wordfreq import zipf_frequency

df = pd.read_csv("cleaned_dataset.csv")
def normalize_repeated_chars(word):
    # replace characters repeated more than 2 times
    word = re.sub(r'(.)\1{2,}', r'\1\1', word)
    return word

def tokenize_text(text):
    text = str(text)
    tokens = text.split()
    return tokens

def normalize_tokens(tokens):
    normalized_tokens = []
    for word in tokens:
        word = normalize_repeated_chars(word)
        normalized_tokens.append(word)
    return normalized_tokens

def tokenize_and_normalize(text):
    tokens = tokenize_text(text)
    tokens = normalize_tokens(tokens)
    return tokens

df["tokens"] = df["clean_text"].apply(tokenize_and_normalize)
# print(df.head())

#TRANSLITERATION
vocabulary = set()
for tokens in df["tokens"]:
    for word in tokens:
        vocabulary.add(word)

# phonetic way
phonetic_dict = {}
for word in vocabulary:
    code = jellyfish.metaphone(word)
    if code not in phonetic_dict:
        phonetic_dict[code] = []
    phonetic_dict[code].append(word)

def get_canonical_word(words):
    return min(words, key=len)

normalization_map = {}

for code, words in phonetic_dict.items():
    canonical = get_canonical_word(words)
    for word in words:
        if fuzz.ratio(word, canonical) > 80:
            normalization_map[word] = canonical

def transliteration_normalize(tokens):
    normalized_tokens = []
    for word in tokens:
        if word in normalization_map:
            normalized_tokens.append(normalization_map[word])
        else:
            normalized_tokens.append(word)
    return normalized_tokens

df["normalized_tokens"] = df["tokens"].apply(transliteration_normalize)
# print(df.head())

#LANGUAGE TAGGING
def detect_language(word):
    # normalize word
    word = word.lower()
    # ignore very short tokens
    if len(word) <= 2:
        return word + "_HI"
    # check English frequency
    freq = zipf_frequency(word, "en")
    # threshold for English detection
    if freq > 3:
        return word + "_EN"
    else:
        return word + "_HI"


def language_tagging(tokens):
    tagged_tokens = []
    for word in tokens:
        tagged_word = detect_language(word)
        tagged_tokens.append(tagged_word)

    return tagged_tokens
df["language_tokens"] = df["normalized_tokens"].apply(language_tagging)
# print(df.head())

#HINGLISH SENTIMENT TAGGING
hinglish_sentiment = {
    # positive words
    "mast": "POS",
    "jhakaas": "POS",
    "bindaas": "POS",
    "sahi": "POS",
    "badhiya": "POS",
    "achha": "POS",
    "acha": "POS",
    "accha": "POS",

    # negative words
    "bakwas": "NEG",
    "faltu": "NEG",
    "bekaar": "NEG",
    "bekar": "NEG",
    "ghatiya": "NEG",
    "kharab": "NEG"
}

def slang_sentiment_tagging(tokens):
    tagged_tokens = []
    for word in tokens:
        base_word = word.split("_")[0]
        if base_word in hinglish_sentiment:
            sentiment = hinglish_sentiment[base_word]
            tagged_tokens.append(base_word + "_" + sentiment)
        else:
            tagged_tokens.append(word)
    return tagged_tokens

df["sentiment_tokens"] = df["language_tokens"].apply(slang_sentiment_tagging)
# print(df.head())

#Stopword Removal
def load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)
    return stopwords

hinglish_stopwords = load_stopwords("../Files/hinglish_stop.txt")
def remove_stopwords(tokens):
    filtered_tokens = []
    for token in tokens:
        word = token.split("_")[0]
        if word not in hinglish_stopwords:
            filtered_tokens.append(token)
    return filtered_tokens

df["final_tokens"] = df["sentiment_tokens"].apply(remove_stopwords)
#JOINED
df["processed_text"] = df["final_tokens"].apply(lambda x: " ".join(x))

df_final = df[["processed_text", "label"]]

df_final.to_csv("final_dataset2.csv", index=False)
print("Preprocessing Done")