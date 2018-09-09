
# coding: utf-8

# - Lexicon - Dictionary of words
# - Bag of words - Every word has an ID (Index) (One-Hot array)
# - Here input vectors have to be of same length
# - Stemming - (Not necessarily a real Word)
# - Lematizing - (Is a Real Word)

# #### NLP Libraries

# In[7]:


import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer


# In[8]:


lemmatizer = WordNetLemmatizer()
hm_lines = 100000


# #### Pre-Processing

# In[9]:


def create_lexicon(pos,neg):

    lexicon = []
    with open(pos,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(neg,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        #print(w_counts[w])
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2


# In[10]:


def sample_handling(sample,lexicon,classification):

    featureset = []

    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features,classification])

    return featureset


# #### Train-Test-Split

# In[11]:


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y


# #### Return length of Lexicon

# If the Dataset is too large you can pickle it and store it so that you don't have to do it again

# In[12]:


if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('/Users/gaddamnitish/Desktop/TensorFlow/SentDex/pos.txt','/Users/gaddamnitish/Desktop/TensorFlow/SentDex/neg.txt')
    # if you want to pickle this data:
    with open('/Users/gaddamnitish/Desktop/TensorFlow/SentDex/sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)


# Basically means when we are sending Data to the Neural Network, the length of every string of ours is 423 characters
