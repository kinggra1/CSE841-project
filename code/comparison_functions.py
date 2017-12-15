from numpy import dot
from numpy.linalg import norm
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import string


# parses a string into a list of lowercase, alphanumeric words
def parse_into_wordlist(sentence):
    result = [filter(str.isalnum, word.lower()) for word in sentence.split()]
    return filter(lambda x: not x == '', result)

def vector_cosine_similarity(list1, list2):
    return dot(list1, list2)/(norm(list1)*norm(list2))

def character_bins(sentence1, sentence2):
    counts1 = {}
    counts2 = {}
    for letter in string.ascii_lowercase:
        counts1[letter] = 0
        counts2[letter] = 0

    for letter in sentence1.lower():
        if not str.isalpha(letter):
            continue
        counts1[letter] += 1

    for letter in sentence2.lower():
        if not str.isalpha(letter):
            continue
        counts2[letter] += 1

    vec1 = [counts1[x] for x in string.ascii_lowercase]
    vec2 = [counts2[x] for x in string.ascii_lowercase]

    return vector_cosine_similarity(vec1, vec2)

def cos_similarity(sentence1, sentence2, model, apply_stopwords=False, stopwords=None):
    word2vec_keys = model.vocab

    sentence1 = parse_into_wordlist(sentence1)
    sentence2 = parse_into_wordlist(sentence2)

    sentence1_w2v = []
    sentence2_w2v = []
    if (apply_stopwords):
        sentence1_w2v = [word for word in sentence1 if (word in word2vec_keys and word not in stopwords)]
        sentence2_w2v = [word for word in sentence2 if (word in word2vec_keys and word not in stopwords)]

    if (sentence1_w2v == [] or sentence2_w2v == []):
        sentence1_w2v = [word for word in sentence1 if word in word2vec_keys]
        sentence2_w2v = [word for word in sentence2 if word in word2vec_keys]

    return model.n_similarity(sentence1_w2v, sentence2_w2v)

def cos_similarity_tfidf(sentence1, sentence2, model, apply_stopwords=False, stopwords=None, idfs=None, tf_idf=None):
    word2vec_keys = model.vocab

    sentence1 = parse_into_wordlist(sentence1)
    sentence2 = parse_into_wordlist(sentence2)

    sentence1_w2v = []
    sentence2_w2v = []
    if (apply_stopwords):
        sentence1_w2v = [word for word in sentence1 if (word in word2vec_keys and word not in stopwords)]
        sentence2_w2v = [word for word in sentence2 if (word in word2vec_keys and word not in stopwords)]

    if (sentence1_w2v == [] or sentence2_w2v == []):
        sentence1_w2v = [word for word in sentence1 if word in word2vec_keys]
        sentence2_w2v = [word for word in sentence2 if word in word2vec_keys]

    percentile = np.percentile(idfs.values(), 30)

    sentence1_w2v = sorted(sentence1_w2v, key=lambda w: idfs[w] if w in idfs else 100, reverse=True)
    for index in range(len(sentence1_w2v)):
        word = sentence1_w2v[index]


        # approximation of importance from idf findings from https://arxiv.org/pdf/1512.00765.pdf
        weight = 1-(index/18)
        if (weight < 0.1):
            weight = 0.1


        #weight = 1
        # if word in idfs:
        #   weight = tf_idf[word]


        #if (word in idfs and idfs[word] < percentile):
        #    print(word)
        #    weight = 0

        # we've already confirmed this word is in our model keys, so let's weight it here
        sentence1_w2v[index] = model.word_vec(word)/np.linalg.norm(model.word_vec(word))*weight

    sentence2_w2v = sorted(sentence2_w2v, key=lambda w: idfs[w] if w in idfs else 100, reverse=True)
    for index in range(len(sentence2_w2v)):
        word = sentence2_w2v[index]

        weight = 1-(index/18)
        if (weight < 0.1):
            weight = 0.1

        #weight = 1
        #if word in tf_idf:
        #   weight = tf_idf[word]

        #if (word in idfs and idfs[word] < percentile):
        #     weight = 0

        # we've already confirmed this word is in our model keys, so let's weight it by it's tf-idf weight here
        sentence2_w2v[index] = model.word_vec(word)/np.linalg.norm(model.word_vec(word))*weight

    #print(sentence1)
    #print(sentence2)
    #print(np.mean(np.array(sentence1_w2v), axis=0))
    #print(np.mean(np.array(sentence1_w2v), axis=0))
    #print(vector_cosine_similarity(np.mean(np.array(sentence1_w2v), axis=0), np.mean(np.array(sentence2_w2v), axis=0)))

    mean1 = np.mean(np.array(sentence1_w2v), axis=0)
    mean2 = np.mean(np.array(sentence2_w2v), axis=0)

    if (sum(mean1) == 0 or sum(mean2 == 0)):
        return 0

    return vector_cosine_similarity(mean1, mean2)


def cos_similarity_length_scaling(sentence1, sentence2, model,  apply_stopwords=False, stopwords=None):
    word2vec_keys = model.vocab

    sentence1 = parse_into_wordlist(sentence1)
    sentence2 = parse_into_wordlist(sentence2)

    sentence1_w2v = []
    sentence2_w2v = []
    if (apply_stopwords):
        sentence1_w2v = [word for word in sentence1 if (word in word2vec_keys and word not in stopwords)]
        sentence2_w2v = [word for word in sentence2 if (word in word2vec_keys and word not in stopwords)]

    if (sentence1_w2v == [] or sentence2_w2v == []):
        sentence1_w2v = [word for word in sentence1 if word in word2vec_keys]
        sentence2_w2v = [word for word in sentence2 if word in word2vec_keys]

    l1 = len(sentence1_w2v)
    l2 = len(sentence2_w2v)
    scaling_factor = 1 - (float(abs(l1-l2))/float(max(l1, l2)))
    return model.n_similarity(sentence1_w2v, sentence2_w2v) * scaling_factor


def cos_similarity_exclusion_set(sentence1, sentence2, model, apply_stopwords=False, stopwords=None):
    word2vec_keys = model.vocab

    sentence1 = parse_into_wordlist(sentence1)
    sentence2 = parse_into_wordlist(sentence2)

    sentence1_w2v = []
    sentence2_w2v = []
    if (apply_stopwords):
        sentence1_w2v = [word for word in sentence1 if (word in word2vec_keys and word not in stopwords)]
        sentence2_w2v = [word for word in sentence2 if (word in word2vec_keys and word not in stopwords)]

        excluded1 = set([word for word in sentence1 if word not in word2vec_keys and word not in stopwords])
        excluded2 = set([word for word in sentence2 if word not in word2vec_keys and word not in stopwords])

    if (sentence1_w2v == [] or sentence2_w2v == []):
        sentence1_w2v = [word for word in sentence1 if word in word2vec_keys]
        sentence2_w2v = [word for word in sentence2 if word in word2vec_keys]

        excluded1 = set([word for word in sentence1 if word not in word2vec_keys])
        excluded2 = set([word for word in sentence2 if word not in word2vec_keys])

    overlap = len(excluded1.intersection(excluded2))

    #print(sentence1)
    #print(sentence2)
    #print(sentence1_w2v)
    #print(sentence2_w2v)
    score = model.n_similarity(sentence1_w2v, sentence2_w2v)
    if (overlap):
        #print("Found one!: {}".format(pow(score, 1.0/float(overlap))))
        #print(excluded1.intersection(excluded2))
        return pow(score, 1.0/float(overlap))
    return score