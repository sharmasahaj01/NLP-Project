import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("Combined.csv")
df = df.dropna(subset=["processed_text"])
print(df.head())

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)
X = vectorizer.fit_transform(df["processed_text"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print(classification_report(y_test, y_pred))

text = ["teacher_EN explanation_HI accha_HI tha_EN bohot_HI"]
vector = vectorizer.transform(text)
prediction = lr_model.predict(vector)
print(prediction)