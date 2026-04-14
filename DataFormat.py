import pandas as pd

df = pd.read_csv("hinglish_student_feedback_dataset_20000.csv", header=None)
df.columns = ["text", "label"]
print(df.head())
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
df["label"] = df["label"].map(label_map)
# print(df.head())
df = df[["text","label"]]
df.to_csv("clean_dataset.csv", index=False)
print(df["label"].value_counts())

print(df.sample(10))