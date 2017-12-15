import math

# parses a string into a list of lowercase, alphanumeric words
def parse_into_wordlist(sentence):
    result = [filter(str.isalnum, word.lower()) for word in sentence.split()]
    return filter(lambda x: not x == '', result)


def compute_meta_features(training_data):
    tf_idf = {}
    idfs = {}

    # list of tfs for each word
    tfs = {}

    doc_contain_count = {}
    total_docs = 0

    recorded_sentences = []

    for example in training_data:
        sentence1 = example[3]
        sentence2 = example[4]

        for sentence in [sentence1, sentence2]:
            # only record stats once for each document(sentence)
            if sentence not in recorded_sentences:
                recorded_sentences.append(sentence)

                # usage counts for words in a single sentence
                usage_count = {}

                sentence = parse_into_wordlist(sentence)

                # total words in this sentence
                words_in_doc = len(sentence)
                total_docs += 1

                # add one to the document appearance count for each unique word in this sentence
                for word in set(sentence):
                    if word in doc_contain_count:
                        doc_contain_count[word] += 1
                    else:
                        doc_contain_count[word] = 1

                for word in sentence:
                    # Keep track of how many times this word is used
                    if word in usage_count:
                        usage_count[word] += 1
                    else:
                        usage_count[word] = 1

                # calculate tf for the words in this sentence
                for word in usage_count.keys():
                    if word in tfs:
                        tfs[word].append(float(usage_count[word])/words_in_doc)
                    else:
                        tfs[word] = [float(usage_count[word])/words_in_doc]

    for word in doc_contain_count.keys():
        tf_total = float(sum(tfs[word]))/len(tfs[word])
        idf = math.log(float(total_docs)/doc_contain_count[word])

        idfs[word] = idf
        tf_idf[word] = tf_total*idf

        #print("{} {} : {} {} {}".format(word, doc_contain_count[word], tf_total, idf, tf_total*idf))

    return idfs, tf_idf
