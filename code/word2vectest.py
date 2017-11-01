# -*- coding: utf-8 -*-

import gensim

# n_similarity(ws1, ws2)
#   Compute cosine similarity between two sets of words.

# word_vec(word, use_norm=False)
    # Accept a single word as input. Returns the word’s representations in vector space, as a 1D numpy array.

print("loading stopwords")
stopwords = open("stopwords.txt", 'r').readlines()
print(stopwords)
print("stopwords loaded")

# Load Google's pre-trained Word2Vec model.
print("loading pretrained word2vec model")
model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)  
print("pretrained word2vec model loaded")

print(model.most_similar(positive=['woman', 'king'], negative=['man']))
