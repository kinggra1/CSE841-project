
# parses a string into a list of lowercase, alphanumeric words
def parse_into_wordlist(sentence):
    result = [filter(str.isalnum, word.lower()) for word in sentence.split()]
    return filter(lambda x: not x == '', result)

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

    print(sentence1)
    print(sentence2)
    print(sentence1_w2v)
    print(sentence2_w2v)
    score = model.n_similarity(sentence1_w2v, sentence2_w2v)
    if (overlap):
        print("Found one!: {}".format(pow(score, 1.0/float(overlap))))
        print(excluded1.intersection(excluded2))
        return pow(score, 1.0/float(overlap))
    return score