# -*- coding: utf-8 -*-
import gensim

print("loading stopwords")
stopwords = [word.strip() for word in open("stopwords.txt", 'r').readlines()]
print(stopwords)
print("stopwords loaded")

# Load Google's pre-trained Word2Vec model.
print("loading pretrained word2vec model")
model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
print("pretrained word2vec model loaded")
    #print(model.most_similar(positive=['woman', 'king'], negative=['man']))
