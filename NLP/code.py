import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# nltk.download_shell()

messages = [line.rstrip() for line in open("SMSSpamCollection")]

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

print(df.head(), df.describe())

print(df.groupby("label").describe())

df["length"] = df["message"].apply(len)

df["length"].plot.hist(bins=150)
plt.show()

df.hist(column="length", by="label", bins=60)
plt.show()


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]


# df["message"].apply(text_process)


# bow_transformer = CountVectorizer(analyzer=text_process).fit(df["message"])

# messages_bow = bow_transformer.transform(df["message"])

# tfidf = TfidfTransformer().fit(messages_bow)

# spam_detect_model = MultinomialNB().fit(tfidf, df["label"])

msg_train, msg_test, label_train, label_test = train_test_split(
    df["message"], df["label"], test_size=0.3)

pipeline = Pipeline([("bow", CountVectorizer(analyzer=text_process)),
                     ("tfidf", TfidfTransformer()), ("classifier", MultinomialNB())])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

print(classification_report(label_test, predictions))
