from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from os import listdir 
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D,MaxPooling1D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers.embeddings import Embedding

def load_doc(filename) :
    file = open(filename,"r")
    text = file.read()
    file.close()
    return text 

vocab = Counter()
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


def clean_doc(doc,vocab):
    tokens = doc.split()
    tokens = [w.lower() for w in tokens]
    words = [word for word in tokens if word.isalpha()]
    tokens = [w for w in words if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

def process_docs(directory,vocab,is_train):
    documents = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + "/" + filename
        doc = load_doc(path)
        tokens = clean_doc(doc,vocab)
        documents.append(tokens)
    return documents


posi_doc = process_docs("/home/sreekesh/python/NLP/txt_sentoken/neg", vocab,True)
neg_doc = process_docs("/home/sreekesh/python/NLP/txt_sentoken/pos",vocab,True)
train_docs = posi_doc + neg_doc

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_docs)

encoded_docs =tokenizer.texts_to_sequences(train_docs)

max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(max_length)

ytrain =np.array([0 for _ in range(900)] + [1 for _ in range(900)])

positive_docs = process_docs('/home/sreekesh/python/NLP/txt_sentoken/pos', vocab, False)
negative_docs = process_docs('/home/sreekesh/python/NLP/txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs

encoded_docs = tokenizer.texts_to_sequences(test_docs)
print(encoded_docs)
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

ytest =np.array([0 for _ in range(100)] + [1 for _ in range(100)])

v_size = len(tokenizer.word_index) + 1     

model = Sequential()
model.add(Embedding(v_size, 100 , input_length = max_length))
model.add(Conv1D(filters=32,kernel_size=8,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10,input_dim=10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(Xtrain,ytrain,epochs=10)

_,accu = model.evaluate(Xtest,ytest)
print("accuracy : {}".format(accu*100))

from numpy import loadtxt 

model.save("/home/sreekesh/python/NLP/nat2mo.h5")

def process_docs(vocab):
	documents = list()
	doc = load_doc("/home/sreekesh/python/NLP/review/reviewb.txt")
	# clean doc
	tokens = clean_doc(doc, vocab)
	# add to list
	documents.append(tokens)
	return documents

pre = process_docs(vocab)
encoded_rev =tokenizer.texts_to_sequences(pre)

print(encoded_rev)

rev = pad_sequences(encoded_rev, maxlen=max_length, padding='post')
print(rev)
revf = np.array(rev)

f = model.predict(revf)
print(f)

if(f <=[[0.5]]):
    print("\n movie less than average ")
else:
    print("\n movie is good to watch ")