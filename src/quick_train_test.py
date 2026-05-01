# import pandas as pd

# df = pd.read_csv("../data/sentiment_class.csv")

# print(df.head())
# print(df.shape)
# print(df.columns)

# print(df["sentiment"].value_counts())

# mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
# df['sentiment'] = df['sentiment'].apply(lambda x: mapping.get(x, -1))

# df = df[df["sentiment"].isin([1, -1])]

# print(df["sentiment"].value_counts())

# corpus = df["text"].tolist()

import pandas as pd

df = pd.read_csv("../data/binary_class.csv")

print(df.head())
print(df.shape)
print(df.columns)

# Vectorization

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X, y = df["text"], df["sentiment"]

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

x_train_vect = vectorizer.fit_transform(X_train)
x_test_vect = vectorizer.transform(X_test)

print(x_train_vect.shape)
print(x_test_vect.shape)

# Model training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(random_state=0)
model = clf.fit(x_train_vect, y_train)

# Prediction
test_pred = model.predict(x_test_vect)

acc = accuracy_score(y_test, test_pred)
print(f"Accuracy - {acc}")



