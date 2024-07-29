#Deep Learning libraries
import os
from keras import backend as K
#Graphing libraries
import matplotlib.pyplot as plt
import seaborn as sns
#NLP libraries
import nltk
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument
#Machine learning libraries
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#Helper libraries
import multiprocessing
import numpy as np
import pandas as pd
import math
from bs4 import BeautifulSoup
import re
import ssl
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt')
nltk.download('stopwords')


def clean(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = text.replace('„','')
    text = text.replace('“','')
    text = text.replace('"','')
    text = text.replace('\'','')
    text = text.replace('-','')
    text = text.lower()
    return text


def remove_stopwords(content):
    for word in stopwords:
        content = content.replace(' '+word+' ',' ')
    return content

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 3:
                continue
            tokens.append(word.lower())
    return tokens

def acc(true, pred):
    acc = 0
    for x,y in zip(true, pred):
        if(x == y):
            acc += 1
    return acc/len(pred)

center_dir = 'datasets/Center Data'
left_dir = '/Users/bccca/PycharmProjects/pythonProject/datasets/Left Data'
right_dir = 'datasets/Right Data'
# Create an empty list to store the data
data = []

# Loop over the files in the Center Data folder and extract their content
for filename in os.listdir(center_dir):
    with open(os.path.join(center_dir, filename), 'r') as f:
        content = f.read()
        data.append({'text': content, 'bias': 'Center'})

# Loop over the files in the Left Data folder and extract their content
for filename in os.listdir(left_dir):
   with open(os.path.join(left_dir, filename), 'r') as f:
       content = f.read()
       data.append({'text': content, 'bias': 'Left'})

# Loop over the files in the Right Data folder and extract their content
for filename in os.listdir(right_dir):
    with open(os.path.join(right_dir, filename), 'r') as f:
        content = f.read()
        data.append({'text': content, 'bias': 'Right'})

# Create a Pandas DataFrame from the data
df = pd.DataFrame(data)
#df['word_count'] = df['text'].apply(lambda x:len(str(x).split()))
#print(df['word_count'])
df = df.iloc[np.random.permutation(len(df))]
df['bias'] = df['bias'].replace(['Center','Left', 'Right'],[0, 1, 2])
#print(df)
print(df['bias'].value_counts())


stopwords = set(stopwords.words('english'))
#print(stopwords)
df['text'] = df['text'].apply(clean)
df['text'] = df['text'].apply(remove_stopwords)

train, test = train_test_split(df, test_size=0.2)

#print(test)

train_tagged = train.apply(
   lambda r: TaggedDocument(words=tokenize_text(r['text']), tags= [r.bias]), axis=1)
test_tagged = test.apply(
   lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.bias]), axis=1)

#print(test_tagged[1800])

cores = multiprocessing.cpu_count()
models = [
    # PV-DBOW
    Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, sample=0, min_count=2, workers=cores),
    # PV-DM
    Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, sample=0,    min_count=2, workers=cores)
]

for model in models:
  model.build_vocab(train_tagged.values)
  model.train(utils.shuffle(train_tagged.values),
    total_examples=len(train_tagged.values),epochs=30)

models[0].save("doc2vec_articles_0.model")
models[1].save("doc2vec_articles_1.model")


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    classes, features = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return features, classes
# PV_DBOW encoded text


train_x_0, train_y_0 = vec_for_learning(models[0], train_tagged)
test_x_0, test_y_0 = vec_for_learning(models[0], test_tagged)
# PV_DM encoded text
train_x_1, train_y_1 = vec_for_learning(models[1], train_tagged)
test_x_1, test_y_1 = vec_for_learning(models[1], test_tagged)

np.unique(train_y_0, axis=0)

#print(train_x_0)

# Machine learning algorithm apply part
#Naive bayes
# bayes_0 = GaussianNB()
# bayes_1 = GaussianNB()
#
# bayes_0.fit(train_x_0,train_y_0)
# bayes_1.fit(train_x_1,train_y_1)
# #Helper function for calculating accuracy on the test set.
# print("naive bayes results: \n")
# print(acc(test_y_0,bayes_0.predict(test_x_0)))
# print(acc(test_y_1,bayes_1.predict(test_x_1)))
#
# #Random forest classifier
#
#
# # Create random forests with 100 decision trees
# forest_0 = RandomForestClassifier(n_estimators=50)
# forest_1 = RandomForestClassifier(n_estimators=50)
#
# forest_0.fit(train_x_0,train_y_0)
# forest_1.fit(train_x_1,train_y_1)
# print("random forest results: \n")
# print(acc(test_y_0,forest_0.predict(test_x_0)))
# print(acc(test_y_1,forest_1.predict(test_x_1)))
#
# #support vector classifier
svc_0 = SVC()
svc_1 = SVC()

svc_0.fit(train_x_0, train_y_0)
svc_1.fit(train_x_1, train_y_1)
print("svc results: \n")
print(acc(test_y_0, svc_0.predict(test_x_0)))
print(acc(test_y_1, svc_1.predict(test_x_1)))

new_data1 = pd.read_csv('/Users/bccca/PycharmProjects/pythonProject/datasets/ChatGPT Responses - Political Spectrum Quiz.csv')
new_data2 = pd.read_csv('/Users/bccca/PycharmProjects/pythonProject/datasets/ChatGPT Responses - Political Compass Test.csv')

print(new_data1)