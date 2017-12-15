# from matplotlitb import pyplot as plt

import scikitplot as skplt
import matplotlib.pyplot as plt

from word2vec import model
from word2vec import stopwords
from paraphrase_corpus import paraphrase_training_data
from paraphrase_corpus import paraphrase_testing_data

from comparison_functions import *
from meta_features import *


idfs, tf_idf = compute_meta_features(paraphrase_training_data)
#for key in sorted(idfs.keys()):
#    print(key, idfs[key])



# training_results = []
# for example in paraphrase_training_data:
#     ground_truth = example[0] == '1'  # convert '0' '1' strings to boolean value
#     sentence1 = example[3]
#     sentence2 = example[4]
#
# ###############################################################################################
#     # Functions to generate comparison scores between two sentence (0-1 where 0 is not a paraphrase and 1 is)
#
#     # score = character_bins(sentence1, sentence2)
#     # score = cos_similarity(sentence1, sentence2, model)
#     # score = cos_similarity(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)
#     # score = cos_similarity_length_scaling(sentence1, sentence2, model)
#     # score = cos_similarity_exclusion_set(sentence1, sentence2, model)
#     # score = cos_similarity_exclusion_set(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)
#     # score = cos_similarity_tfidf(sentence1, sentence2, model, idfs=idfs, tf_idf=tf_idf)
#
# ###############################################################################################
#
#     # output same data with ground truth converted to boolean and cosine similarity appended
#     training_results.append((ground_truth, example[1], example[2], sentence1, sentence2, score))
#
#
#
# # create sets for threshold evaluation
# true_positive = [result for result in training_results if result[0]]
# true_negative = [result for result in training_results if not result[0]]
#
# training_results.sort(key=lambda result: result[5])
# true_positive.sort(key=lambda result: result[5])
# true_negative.sort(key=lambda result: result[5])
#
# max_accuracy = 0
# best_threshold = 0
# pos_index = 0
# neg_index = 0
# for case in training_results:
#     threshold = case[5]
#
#     # update our position int the true_positive and true_negative lists
#     while(pos_index < len(true_positive) and true_positive[pos_index][5] <= threshold):
#         pos_index += 1
#
#     while(neg_index < len(true_negative) and true_negative[neg_index][5] <= threshold):
#         neg_index += 1
#
#
#     TP = len(true_positive) - pos_index
#     FN = pos_index
#     TN = neg_index
#     FP = len(true_negative) - neg_index
#     accuracy = float(TP + TN)/float(len(training_results))
#
#     if (accuracy > max_accuracy):
#         max_accuracy = accuracy
#         best_threshold = threshold
#
# print("Max accuracy: ", max_accuracy);






# Testing evaluation
correct_count = 0
positive_total = 0
total = 0
ground_truths = []

char_bin_scores = []
cos_sim_scores = []
cos_sim_stopwords_scores = []
cos_sim_scaling_scores = []
cos_sim_exclusion_scores = []
cos_sim_exclusion_stopwords_scores = []
tf_idf_scores = []

for test in paraphrase_testing_data:
    ground_truth = test[0] == '1'
    sentence1 = test[3]
    sentence2 = test[4]

    if ((len(sentence1.split()) < 15) or len(sentence1.split()) <= 20) or (len(sentence2.split()) <= 15 or len(sentence2.split()) <= 20):
    #if (abs(len(sentence1.split()) - len(sentence2.split())) < 10):
        continue



###############################################################################################
    # Functions to generate comparison scores between two sentence (higher number is higher likelihood of a match)

    char_bin_score = character_bins(sentence1, sentence2) # 0.704347826087
    cos_sim_score = cos_similarity(sentence1, sentence2, model) # 0.721739130435
    cos_sim_stopwords_score = cos_similarity(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)
    cos_sim_scaling_score = cos_similarity_length_scaling(sentence1, sentence2, model)
    cos_sim_exclusion_score = cos_similarity_exclusion_set(sentence1, sentence2, model)
    cos_sim_exclusion_stopwords_score = cos_similarity_exclusion_set(sentence1, sentence2, model, apply_stopwords=True, stopwords=stopwords)
    tf_idf_score = cos_similarity_tfidf(sentence1, sentence2, model, idfs=idfs, tf_idf=tf_idf) # 0.721739130435

###############################################################################################

    # prediction = score > best_threshold
    # if (prediction == ground_truth):
    #     correct_count += 1
    #
    # if ground_truth:
    #     positive_total += 1
    # total += 1
    #

    ground_truths.append(ground_truth)

    char_bin_scores.append(char_bin_score)
    cos_sim_scores.append(cos_sim_score)
    cos_sim_stopwords_scores.append(cos_sim_stopwords_score)
    cos_sim_scaling_scores.append(cos_sim_scaling_score)
    cos_sim_exclusion_scores.append(cos_sim_exclusion_score)
    cos_sim_exclusion_stopwords_scores.append(cos_sim_exclusion_stopwords_score)
    tf_idf_scores.append(tf_idf_score)

    #print("Ground Truth: {} \t Cosine Similarity: {}".format(ground_truth, score))

# print(float(positive_total)/total)
# print("Using Threshold: {}".format(best_threshold))
print("Accuracy: {}".format(float(correct_count)/float(len(paraphrase_testing_data))))


fpr, tpr, thresholds = skplt.metrics.roc_curve(ground_truths, char_bin_scores)
roc_auc = skplt.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='red', lw=2, label='Character Binning (area = %0.2f)' % roc_auc)

fpr, tpr, thresholds = skplt.metrics.roc_curve(ground_truths, cos_sim_scores)
roc_auc = skplt.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Mean Cosine Similarity (area = %0.2f)' % roc_auc)

fpr, tpr, thresholds = skplt.metrics.roc_curve(ground_truths, cos_sim_stopwords_scores)
roc_auc = skplt.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', lw=2, label='Mean w/ Stopwords (area = %0.2f)' % roc_auc)

fpr, tpr, thresholds = skplt.metrics.roc_curve(ground_truths, cos_sim_scaling_scores)
roc_auc = skplt.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='green', lw=2, label='Mean w/ Scaling (area = %0.2f)' % roc_auc)

fpr, tpr, thresholds = skplt.metrics.roc_curve(ground_truths, cos_sim_exclusion_scores)
roc_auc = skplt.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='purple', lw=2, label='Mean w/ Excluded (area = %0.2f)' % roc_auc)

fpr, tpr, thresholds = skplt.metrics.roc_curve(ground_truths, cos_sim_exclusion_stopwords_scores)
roc_auc = skplt.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='black', lw=2, label='Mean w/ Exclusion+Stopwords (area = %0.2f)' % roc_auc)

fpr, tpr, thresholds = skplt.metrics.roc_curve(ground_truths, tf_idf_scores)
roc_auc = skplt.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='cyan', lw=2, label='Mean w/ IDF Importance (area = %0.2f)' % roc_auc)



plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Sentences Over 20 Words Long')
plt.legend(loc="lower right")
plt.show()