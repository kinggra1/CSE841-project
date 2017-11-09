test_file = open('data/MSRPC/msr_paraphrase_test.txt', 'r')
paraphrase_testing_data = [line.split('\t') for line in test_file.readlines()]
paraphrase_testing_data = paraphrase_testing_data[1:]

train_file = open('data/MSRPC/msr_paraphrase_train.txt', 'r')
paraphrase_training_data = [line.split('\t') for line in train_file.readlines()]
paraphrase_training_data = paraphrase_training_data[1:]

