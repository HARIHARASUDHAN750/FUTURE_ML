import pandas as pd
import nltk
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('punkt')

df = pd.read_csv("customer_support_tickets.csv")

print("\n🔍 Top 5 Ticket Types:")
print(df['Ticket Type'].value_counts().head(5))

print("\n🔍 Top 5 Ticket Subjects:")
print(df['Ticket Subject'].value_counts().head(5))

# Analyze frequent words in descriptions
text = " ".join(df['Ticket Description'].dropna()).lower()
tokens = nltk.word_tokenize(text)
freq = Counter(tokens)

print("\n🔤 Top 10 frequent words in Ticket Descriptions:")
for word, count in freq.most_common(10):
    print(f"{word}: {count}")

# Bar chart
df['Ticket Type'].value_counts().head(5).plot(kind='bar', title='Top 5 Ticket Types')
plt.xlabel('Ticket Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
