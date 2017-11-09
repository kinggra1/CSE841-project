# from matplotlitb import pyplot as plt

from word2vec import model
from word2vec import stopwords
from paraphrase_corpus import paraphrase_training_data
from paraphrase_corpus import paraphrase_testing_data

from comparison_functions import *

training_results = []
for example in paraphrase_training_data:
    ground_truth = example[0] == '1'  # convert '0' '1' strings to boolean value
    sentence1 = example[3]
    sentence2 = example[4]

###############################################################################################
    # Functions to generate comparison scores between two sentence (0-1 where 0 is not a paraphrase and 1 is)

    # score = cos_similarity(sentence1, sentence2, model)
    # score = cos_similarity(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)
    # score = cos_similarity_length_scaling(sentence1, sentence2, model)
    # score = cos_similarity_exclusion_set(sentence1, sentence2, model)
    score = cos_similarity_exclusion_set(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)

###############################################################################################

    # output same data with ground truth converted to boolean and cosine similarity appended
    training_results.append((ground_truth, example[1], example[2], sentence1, sentence2, score))



# create sets for threshold evaluation
true_positive = [result for result in training_results if result[0]]
true_negative = [result for result in training_results if not result[0]]

training_results.sort(key=lambda result: result[5])
true_positive.sort(key=lambda result: result[5])
true_negative.sort(key=lambda result: result[5])

max_accuracy = 0
best_threshold = 0
pos_index = 0
neg_index = 0
for case in training_results:
    threshold = case[5]

    # update our position int the true_positive and true_negative lists
    while(pos_index < len(true_positive) and true_positive[pos_index][5] <= threshold):
        pos_index += 1

    while(neg_index < len(true_negative) and true_negative[neg_index][5] <= threshold):
        neg_index += 1


    TP = len(true_positive) - pos_index
    FN = pos_index
    TN = neg_index
    FP = len(true_negative) - neg_index
    accuracy = float(TP + TN)/float(len(training_results))

    if (accuracy > max_accuracy):
        max_accuracy = accuracy
        best_threshold = threshold

print("Max accuracy: ", max_accuracy);






# Testing evaluation
correct_count = 0
for test in paraphrase_testing_data:
    ground_truth = test[0] == '1'
    sentence1 = test[3]
    sentence2 = test[4]

###############################################################################################
    # Functions to generate comparison scores between two sentence (higher number is higher likelihood of a match)

    # score = cos_similarity(sentence1, sentence2, model)
    # score = cos_similarity(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)
    # score = cos_similarity_length_scaling(sentence1, sentence2, model)
    # score = cos_similarity_exclusion_set(sentence1, sentence2, model)
    score = cos_similarity_exclusion_set(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)

###############################################################################################

    prediction = score > best_threshold
    if (prediction == ground_truth):
        correct_count += 1

    print("Ground Truth: {} \t Cosine Similarity: {}".format(ground_truth, score))

print("Accuracy: {}".format(float(correct_count)/float(len(paraphrase_testing_data))))