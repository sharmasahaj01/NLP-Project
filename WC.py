import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
df = pd.read_csv("Combined.csv")
df = df.dropna(subset=["processed_text"])
# Clean text (remove _EN/_HI)
df["clean_text"] = df["processed_text"].apply(
    lambda x: " ".join([w.split("_")[0] for w in x.split()])
)

# Remove noise words
stop_words = set([
    "tha", "thi", "aur", "me", "par", "hai", "the", "to", "ka", "ki"
])

words = " ".join(df["clean_text"]).split()
filtered_words = [w for w in words if w not in stop_words]

# Generate word cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="viridis",   # looks professional
    max_words=100
).generate(" ".join(filtered_words))

# Plot and save
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

plt.savefig("wordcloud.png", dpi=300, bbox_inches='tight')
plt.show()