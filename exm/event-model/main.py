
import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import chainer.functions as F
import numpy as np
from chainer import iterators, optimizers
import chainer.computational_graph as c
from chainer import serializers

import constants as const
from pipeline.util import Util
from pipeline.Gold import Gold
import pipeline.preprocess as preproc

from datetime import datetime
import argparse

from model.Model import Model
import pipeline.postprocess as postprocess
import debug.scratch as scratch

from distutils.dir_util import copy_tree

first_start_time = datetime.now()
start_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, required=True, help='yaml file')
parser.add_argument('--train', action='store_true', help='training mode - uses train and dev set; gold is provided')
parser.add_argument('--test', action='store_true', help='testing mode - predicts only; needs a model to load')
args = parser.parse_args()

with open(args.yaml, 'r') as stream:
    params = Util.ordered_load(stream)
params = params['EVENT']

#change output folder to use yaml file
temp1 = args.yaml.split("./")
output_folder = temp1[1].split(".yaml")[0]
params['OUTPUT_FOLDER'] = output_folder

result_file = open(const.OUTPUT_LOG_FILE, 'w')

print('\nMODEL PARAMETERS:')
for k, v in params.items():
    print(k, ':', v)
    result_file.write(k+":"+str(v)+"\n")

print('\n')

USE_FILTER = params['USE_FILTER']
TRAIN_DIR = params['TRAIN_DIR']
VERBOSE = params['VERBOSE']
IS_EMO = params['IS_EMO']
TEST_DIR = params['TEST_DIR']
TRAIN_NER = params['TRAIN_NER']
TRAIN_REL = params['TRAIN_REL']

# GOLD DATA PROCESSING
train_gold_instances, train_invalid_event_count, file_id_events = Gold.load(TRAIN_DIR, VERBOSE, IS_EMO)
print("There are ", train_invalid_event_count, " intersentence events in the train gold identified.")
result_file.write("There are " + str(train_invalid_event_count) + " intersentence events in the train gold identified.")

if args.train:
    test_gold_instances, test_invalid_event_count, _ = Gold.load(TEST_DIR, VERBOSE, IS_EMO)
    print("There are ", test_invalid_event_count, " intersentence events in the test gold identified.")
    result_file.write("There are "+ str(test_invalid_event_count)+ " intersentence events in the test gold identified.")

#CREATE FILTER FROM TRAIN GOLD
#TODO: If the more structures from training data the better it is, then maybe intersentence structures should not be removed.
# All structures must be considered on the document level.
# filter, intersentence_event_count = Gold.extract_filter(train_gold_instances, VERBOSE)
filter = Gold.extract_filter(file_id_events, VERBOSE)
# print("There are ", intersentence_event_count, " intersentence events in the train gold excluded by filter.")
# result_file.write("There are " + str(intersentence_event_count) + " intersentence events in the train gold excluded by filter.")

# TRAINING DATA PROCESSING
train_instances4, train_all_file_ids, train_invalid_rel_count = Model.load(TRAIN_DIR, TRAIN_NER, TRAIN_REL, VERBOSE, IS_EMO)
if 'EXCLUDE_TRAIN' in params:
    train_instances4 = Util.exlude(train_instances4, params['EXCLUDE_TRAIN'])
    train_gold_instances = Util.exlude(train_gold_instances, params['EXCLUDE_TRAIN'])
print("There are ", train_invalid_rel_count, " intersentence relations in the train.")
result_file.write("There are "+ str(train_invalid_rel_count)+ " intersentence relations in the train.")

train_triggers_empty_indices = Model.get_indices_with_empty_triggers(train_instances4)
train_instances3 = Model.remove_instances_with_empty_triggers(train_instances4, train_triggers_empty_indices)
train_new_gold = Gold.remove_instances_with_attributes(train_gold_instances, train_triggers_empty_indices)
train_instances2 = Model.generate_candidate_structures(train_instances3,filter, params['GENERALISATION'],
                                                       params['INCLUDE_MENTION'], USE_FILTER,
                                                       params['PARTIAL_ARGUMENT_MATCHING'], params['VERBOSE'])
train_event_empty_indices = Model.get_indices_with_empty_events(train_instances2, USE_FILTER)
train_instances1 = Model.remove_instances_with_empty_events(train_instances2, train_event_empty_indices)
train_new_gold = Gold.remove_instances_with_attributes(train_new_gold, train_event_empty_indices)
train_instances = Model.generate_numbered_roles_for_some_events(train_instances1, params['EVENTS_WITH_NUMBERED_ROLES'], USE_FILTER)

# CREATE VOCABULARY
word_type2id, role_type2id, trigger_type2id, entity_type2id, singleton = Model.create_vocabulary(train_instances,
                                                    params['UNK_MIN_FREQUENCY'], params['UNK_ASSIGNMENT_PROBABILITY'])


# TEST DATA PROCESSING
test_instances4, test_all_file_ids, test_invalid_rel_count = Model.load(TEST_DIR, params['TEST_NER'], params['TEST_REL'], VERBOSE, IS_EMO)
if 'EXCLUDE_TEST' in params:
    test_instances4 = Util.exlude(test_instances4, params['EXCLUDE_TEST'])
    test_gold_instances = Util.exlude(test_gold_instances, params['EXCLUDE_TEST'])
print("There are ", test_invalid_rel_count, " intersentence relations in the test.")
result_file.write("There are "+ str(test_invalid_rel_count)+ " intersentence relations in the test.")

test_triggers_empty_indices = Model.get_indices_with_empty_triggers(test_instances4)
test_instances3 = Model.remove_instances_with_empty_triggers(test_instances4, test_triggers_empty_indices)
if args.train:
    test_new_gold = Gold.remove_instances_with_attributes(test_gold_instances, test_triggers_empty_indices)
test_instances2 = Model.generate_candidate_structures(test_instances3,filter, params['GENERALISATION'],
                                                    params['INCLUDE_MENTION'], USE_FILTER,
                                                    params['PARTIAL_ARGUMENT_MATCHING'], params['VERBOSE'])
test_event_empty_indices = Model.get_indices_with_empty_events(test_instances2, USE_FILTER)
test_instances1 = Model.remove_instances_with_empty_events(test_instances2, test_event_empty_indices)
if args.train:
    test_new_gold = Gold.remove_instances_with_attributes(test_new_gold, test_event_empty_indices)
test_instances = Model.generate_numbered_roles_for_some_events(test_instances1, params['EVENTS_WITH_NUMBERED_ROLES'], USE_FILTER)

# GENERATE IDS
train_instances_ids, train_unk_count, train_word_count = Model.instances_to_ids(train_instances, word_type2id, role_type2id,trigger_type2id,entity_type2id, USE_FILTER, VERBOSE)
test_instances_ids, test_unk_count, test_word_count = Model.instances_to_ids(test_instances, word_type2id, role_type2id,trigger_type2id,entity_type2id, USE_FILTER, VERBOSE)

print("There are ", train_unk_count, " unk tokens out of ", train_word_count, " words in the train.")
print("There are ", test_unk_count, " unk tokens out of ", test_word_count, " words in the test.")
result_file.write("There are "+ str(train_unk_count)+ " unk tokens in the train.")
result_file.write("There are "+ str(test_unk_count)+ " unk tokens in the test.")

assert len(train_instances_ids) == len(train_new_gold), "Error: len(train_instances_ids) != len(train_new_gold)"
if args.train:
    assert len(test_instances_ids) == len(test_new_gold), "Error: len(test_instances_ids) != len(test_new_gold)"

# EXTRACT THE TARGET LABELS FOR TRAINING
train_target_labels, train_count_tr, train_count_inter = Gold.label_instances(train_instances, train_new_gold, USE_FILTER, VERBOSE)
if args.train:
    test_target_labels, test_count_tr, test_count_inter = Gold.label_instances(test_instances, test_new_gold, USE_FILTER, VERBOSE)
    print("There are ", train_count_tr, " TR event arguments and ", train_count_inter, " intersentence events in the train gold.")
    print("There are ", test_count_tr, " TR event arguments and ", test_count_inter, " intersentence events in the test gold.")
    result_file.write("There are "+ str(train_count_tr)+ " TR event arguments and " + str(train_count_inter)+ " intersentence events in the train.")
    result_file.write("There are "+ str(test_count_tr)+ " TR event arguments and " + str(test_count_inter)+ " intersentence events in the test.")

train_type_dist = Util.extract_event_sample_distribution(train_instances, train_target_labels, USE_FILTER)
if args.train:
    test_type_dist = Util.extract_event_sample_distribution(test_instances, test_target_labels, USE_FILTER)
    Util.print_event_type_dist(train_type_dist, test_type_dist)

#TODO: is there a better way to count these events?
# ones, zeros = Util.count_events_non_events(train_target_labels)
# result_file.write("train  (events/non-events):" + str(ones)+"/"+str(zeros)+"\n")
# print("train  (events/non-events):", Util.count_events_non_events(train_target_labels))
# if args.train:
#     ones, zeros = Util.count_events_non_events(test_target_labels)
#     result_file.write("test  (events/non-events):" + str(ones)+"/"+str(zeros)+"\n")
#     print("test  (events/non-events):", Util.count_events_non_events(test_target_labels))
#     not_the_same = Util.check_instances_with_diff(test_new_gold, test_target_labels, test_instances)

#debug the generated combinations
scratch.printinfo(train_instances)
scratch.printinfo(test_instances)


# MODEL PREPARATION
model = Model.predictor(len(word_type2id), len(trigger_type2id), len(role_type2id), len(entity_type2id), trigger_type2id, entity_type2id,
                        params['DIM_EMBED'], params['DIM_EVENT'], params['DIM_BILSTM'], params['DIM_TRIG_TYPE'],
                        params['DIM_ROLE_TYPE'], params['DIM_ARG_TYPE'], params['DIM_IO'], params['DROPOUT'], params['REPLACE_TYPE'],
                        params['GENERALISATION'], params['THRESHOLD'])
model.load_glove(params['EMBEDDING_PATH'], word_type2id)
optimizer = optimizers.Adam()
optimizer.setup(model)

#UNCOMMENT FOR TESTING
# start = 0
# end = 7
# train_instances = train_instances[start:end]
# train_instances_ids = train_instances_ids[start:end]
# train_target_labels = train_target_labels[start:end]
# test_instances = test_instances[start:end]
# test_instances_ids = test_instances_ids[start:end]
# if args.train:
#     test_target_labels = test_target_labels[start:end]
#     test_new_gold = test_new_gold[start:end]

#shuffle the pair of ids and target
train_instances_ids, train_instances, train_target_labels = Util.shuffle(train_instances_ids, train_instances, train_target_labels)

# PREPARE DATA ITERATORS
train_iter = iterators.SerialIterator(train_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=False, repeat=False)
train_target_iter = iterators.SerialIterator(train_target_labels, batch_size=params['BATCH_SIZE'], shuffle=False, repeat=False)

result_file.write("Epoch\t\tTr_Prec\t\tTr_Rec\t\tTr_F\t\tTs_Prec\t\tTs_Rec\t\tTs_F\n")
print("Epoch\t\tTr_Prec\t\tTr_Rec\t\tTr_F\t\tTs_Prec\t\tTs_Rec\t\tTs_F\n")

train_predictions = []
train_fscores = []
test_fscores = []

train_epoch = 0

prev_fscore = 0.0
prev_recall = 0.0
best_fscore = 0.0

training_patience = 0

batch_ctr = 0

if args.train:
    # TRAINING LOOP
    while train_epoch < params['MAX_EPOCH']:
        batch_start = datetime.now()
        train_batch = train_iter.next()
        target_batch = train_target_iter.next()
        train_prediction, batch_loss, _ = model(train_batch, target_batch)

        # UNCOMMENT TO GENERATE THE COMPUTATIONAL GRAPH
        # g = c.build_computational_graph([batch_loss])
        # with open('myoutput.dot', 'w') as o:
        #     o.write(g.dump())

        train_predictions.extend(train_prediction)
        model.cleargrads()
        batch_loss.backward()
        optimizer.update()

        #batch timers
        batch_ctr += 1
        batch_end = datetime.now()

        if train_iter.is_new_epoch:

            train_epoch += 1

            train_predictions_cnt = Util.count_event_predictions(train_predictions)
            # print("train_predictions_cnt:", train_predictions_cnt)

            Model.generate_prediction_file(train_instances, train_predictions, const.TRAIN_OUTPUT_DIR, train_all_file_ids, params['SPECIAL_ENTITIES'], USE_FILTER)

            # train_fscore, train_recall, train_precision = postprocess.compute_f_score(const.TRAIN_OUTPUT_DIR, TRAIN_DIR, TEST_DIR, IS_TRAIN=True)

            train_fscore, train_recall, train_precision = postprocess.compute_f_score(const.TRAIN_OUTPUT_DIR, TRAIN_DIR, TEST_DIR, params['EVALUATION_SCRIPT'], params['INTERPRETER'], params['DATASET'], IS_TRAIN=True)


            train_fscores.append(train_fscore)


            result_file.write('{:02d}\t\t{:.04f}\t\t{:.04f}\t\t{:.04f}\t\t'.format(train_epoch, train_precision,train_recall,  train_fscore))
            print('{:02d}\t\t{:.04f}\t\t{:.04f}\t\t{:.04f}\t\t'.format(train_epoch, train_precision,train_recall,  train_fscore), end='')

            train_predictions = []
            test_predictions = []

            test_instances_ids, test_instances, test_target_labels = Util.shuffle(test_instances_ids,
                                                                                  test_instances,
                                                                                  test_target_labels)
            test_iter = iterators.SerialIterator(test_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=False,
                                                 repeat=False)
            test_target_iter = iterators.SerialIterator(test_target_labels, batch_size=params['BATCH_SIZE'], shuffle=False, repeat=False)

            test_fscore = 0.0

            batch_ctr = 0

            while True:

                batch_start = datetime.now()

                test_batch = test_iter.next()
                test_prediction, _, _ = model(test_batch)
                test_predictions.extend(test_prediction)

                batch_ctr += 1
                batch_end = datetime.now()

                if test_iter.is_new_epoch:
                    Model.generate_prediction_file(test_instances, test_predictions, const.TEST_OUTPUT_DIR, test_all_file_ids, params['SPECIAL_ENTITIES'], USE_FILTER)
                    # test_fscore, test_recall, test_precision = postprocess.compute_f_score(const.TEST_OUTPUT_DIR, TRAIN_DIR, TEST_DIR, IS_TRAIN=False)
                    test_fscore, test_recall, test_precision = postprocess.compute_f_score(const.TEST_OUTPUT_DIR, TRAIN_DIR, TEST_DIR, params['EVALUATION_SCRIPT'], params['INTERPRETER'], params['DATASET'],IS_TRAIN=False)
                    test_fscores.append(test_fscore)
                    result_file.write('{:.04f}\t\t{:.04f}\t\t{:.04f}'.format(test_precision,test_recall,  test_fscore)+"\n")
                    print('{:.04f}\t\t{:.04f}\t\t{:.04f}'.format(test_precision,test_recall,  test_fscore),end='')
                    end = datetime.now()
                    running_time = end - start_time
                    print("\t", running_time)
                    result_file.close()
                    break

            if test_fscore > best_fscore:
                best_fscore = test_fscore
                training_patience = 0
                if train_epoch >= params['STARTING_EPOCH_FOR_OUTPUT']:
                    # save the model here
                    serializers.save_npz(const.MODEL_OUTPUT_FILE, model)
                    print("Model saved at epoch ", train_epoch, const.MODEL_OUTPUT_FILE)

                    root_folder = params['ROOT_OUTPUT_COPY_FOLDER']
                    output_folder = params['OUTPUT_FOLDER']
                    epoch_output_folder = root_folder + params['EPOCH_OUTPUT_FOLDER']
                    if not os.path.exists(epoch_output_folder):
                        os.makedirs(epoch_output_folder)
                    from_folder = root_folder + "exm/event-model/evaluation/" + output_folder
                    to_folder = epoch_output_folder + output_folder + "_" + str(train_epoch)
                    copy_tree(from_folder, to_folder)
            else:
                training_patience += 1
            if training_patience == params['PATIENCE']:
                break

            # generate new shuffled training set
            train_instances_ids, train_instances, train_target_labels = Util.shuffle(train_instances_ids,
                                                                                     train_instances,
                                                                                     train_target_labels)
            train_iter = iterators.SerialIterator(train_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=False,
                                                  repeat=False)
            train_target_iter = iterators.SerialIterator(train_target_labels, batch_size=params['BATCH_SIZE'],
                                                         shuffle=False,
                                                         repeat=False)
            result_file = open(const.OUTPUT_LOG_FILE, 'a')
            start_time = datetime.now()

elif args.test:
    print("Loading model...", end='')
    serializers.load_npz(params['MODEL'], model)
    print("Finished.")
    test_instances_ids, test_instances = Util.shuffle(test_instances_ids, test_instances)
    test_iter = iterators.SerialIterator(test_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=False,
                                         repeat=False)

    test_fscore = 0.0
    test_predictions = []
    print("Generating test set predictions..", end='')
    total_count = 0
    while True:
        test_batch = test_iter.next()
        test_prediction, _, count = model(test_batch)
        test_predictions.extend(test_prediction)

        total_count += count
        if test_iter.is_new_epoch:
            Model.generate_prediction_file(test_instances, test_predictions, const.TEST_OUTPUT_DIR, test_all_file_ids, params['SPECIAL_ENTITIES'], USE_FILTER)
            if params['SCENARIO'] == 1:
                print("scenario 1")
                # test_fscore, test_recall, test_precision = postprocess.compute_f_score(const.TEST_OUTPUT_DIR, TRAIN_DIR,TEST_DIR, IS_TRAIN=False)
                test_fscore, test_recall, test_precision = postprocess.compute_f_score(const.TEST_OUTPUT_DIR, TRAIN_DIR, TEST_DIR, params['EVALUATION_SCRIPT'], params['INTERPRETER'], params['DATASET'], IS_TRAIN=False)
                print('{:.04f}\t\t{:.04f}\t\t{:.04f}'.format(test_precision, test_recall, test_fscore), end='')
                result_file.write('\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(test_precision, test_recall, test_fscore) + "\n")
            elif params['SCENARIO'] == 0:
                # print("test")
                postprocess.convert_tr_to_t(const.TEST_OUTPUT_DIR)

            break

    test_fscores.append(test_fscore)
    print("Finished.")
    print("Total Classifications:", total_count)

end = datetime.now()
total_time = end - first_start_time
print("\nRunning time:", total_time)
result_file = open(const.OUTPUT_LOG_FILE, 'a')
result_file.write("\nRunning time:" + str(total_time))
result_file.close()
