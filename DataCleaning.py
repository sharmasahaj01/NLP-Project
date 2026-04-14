import re
import pandas as pd

def clean_text(text):
    # convert to lowercase
    text = str(text)
    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # remove hashtags (#topic)
    text = re.sub(r'#\w+', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df = pd.read_csv("hinglish_student_feedback_dataset_20000.csv")
df = df.dropna(subset=["text"])
df["clean_text"] = df["text"].apply(clean_text)
df_clean = df[["clean_text","label"]]

df_clean.to_csv("cleaned_dataset.csv", index=False)
# df.to_csv("processed_dataset.csv", index=False)
# print(df[["text","clean_text"]].head())