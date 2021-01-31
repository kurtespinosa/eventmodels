
import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import chainer
import chainer.functions as F
from chainer import iterators, optimizers
import chainer.computational_graph as c
from chainer import serializers

import src.constants as const
from src.util.util_ import Util
from src.util.process_gold_ import GoldDataProcessor as gdp
import src.util.preprocess as preproc

from datetime import datetime
import argparse



from src.util.process_data import DataProcessor as dp
import src.util.postprocess as postprocess

from distutils.dir_util import copy_tree

import logging
logging.basicConfig(filename='log.txt', format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG, filemode='w')
log = logging.getLogger(__name__)


first_start_time = datetime.now()
start_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, required=True, help='yaml file')

args = parser.parse_args()

with open(args.yaml, 'r') as stream:
    params = Util.ordered_load(stream)
params = params['EVENT']

#change output folder to use yaml file
temp1 = args.yaml.split("./")
output_folder = temp1[1].split(".yaml")[0]
output_folder = output_folder.split("yaml/")[1]
params['OUTPUT_FOLDER'] = output_folder

result_file = open(const.OUTPUT_LOG_FILE, 'w')

print('\nMODEL PARAMETERS:')
for k, v in params.items():
    print(k, ':', v)
    result_file.write(k+":"+str(v)+"\n")
print('\n')

#factor common params
IS_EMO = params['IS_EMO'] #is emo contains sentence boundary information
TRAIN_DIR = params['TRAIN_DIR']
VERBOSE = params['VERBOSE']
TEST_DIR = params['TEST_DIR']
TRAIN_NER = params['TRAIN_NER']
TRAIN_REL = params['TRAIN_REL']
TEST_NER = params['TEST_NER']
TEST_REL = params['TEST_REL']
USE_FILTER = params['USE_FILTER']

TASK = params['TASK']


#DEBUG
# data_file = open(const.OUTPUT_DATA_FILE, 'w')


# GOLD DATA PROCESSING
train_gold_instances, train_invalid_event_count, file_id_events, train_num_events = gdp.load(TRAIN_DIR, VERBOSE, IS_EMO)
print("TRAIN set: ", train_num_events, " gold events and of which ", train_invalid_event_count, " are intersentence events.")
result_file.write("TRAIN set: " + str(train_num_events) + " gold events and of which " + str(train_invalid_event_count) + " are intersentence events.\n")
train_gold_instances = gdp.denumber_roles(train_gold_instances)

if TASK in ['TUNE']:
    test_gold_instances, test_invalid_event_count, _, test_num_events = gdp.load(TEST_DIR, VERBOSE, IS_EMO)
    print("TEST set: ", test_num_events, " gold events and of which ", test_invalid_event_count, " are intersentence events.")
    result_file.write("TEST set: " + str(test_num_events) + " gold events and of which " + str(test_invalid_event_count) + " are intersentence events.\n")
    test_gold_instances = gdp.denumber_roles(test_gold_instances)

#CREATE FILTER FROM TRAIN GOLD
#Option 1: Include ALL event structure even if they are intersentence
filter = gdp.extract_filter(file_id_events, VERBOSE)
#Option 2: Exclude intersentence event structures
'''
filter, intersentence_event_count = gdp.extract_filter(train_gold_instances, VERBOSE)
print("There are ", intersentence_event_count, " intersentence events in the train gold excluded by filter.")
result_file.write("There are " + str(intersentence_event_count) + " intersentence events in the train gold excluded by filter.")
'''

# TRAINING DATA PROCESSING
train_instances4, train_all_file_ids, train_invalid_rel_count = dp.load(TRAIN_DIR, TRAIN_NER, TRAIN_REL, VERBOSE, IS_EMO)
if 'EXCLUDE_TRAIN' in params:
    train_instances4 = Util.exlude(train_instances4, params['EXCLUDE_TRAIN'])
    train_gold_instances = Util.exlude(train_gold_instances, params['EXCLUDE_TRAIN'])
print("There are ", train_invalid_rel_count, " intersentence relations in the train.")
result_file.write("There are "+ str(train_invalid_rel_count)+ " intersentence relations in the train.\n")

train_triggers_empty_indices = dp.get_indices_with_empty_triggers(train_instances4)
train_instances3 = dp.remove_instances_with_empty_triggers(train_instances4, train_triggers_empty_indices)
train_new_gold = gdp.remove_instances_with_attributes(train_gold_instances, train_triggers_empty_indices)
train_instances2 = dp.generate_candidate_structures(train_instances3,filter, params['GENERALISATION'],params['INCLUDE_MENTION'], USE_FILTER,params['PARTIAL_ARGUMENT_MATCHING'], dp, VERBOSE)

train_event_empty_indices = dp.get_indices_with_empty_events(train_instances2, USE_FILTER)
train_instances = dp.remove_instances_with_empty_events(train_instances2, train_event_empty_indices)
train_new_gold = gdp.remove_instances_with_attributes(train_new_gold, train_event_empty_indices)

# CREATE VOCABULARY
word_type2id, role_type2id, trigger_type2id, entity_type2id, singleton = dp.create_vocabulary(train_instances,params['UNK_MIN_FREQUENCY'], params['UNK_ASSIGNMENT_PROBABILITY'])

if TASK == "TEST":
    IS_EMO = True #since predictions are based on IS_EMO, this is False for EM and True for TEES
if TASK in ['TUNE', 'TEST'] :
    # TEST DATA PROCESSING
    test_instances4, test_all_file_ids, test_invalid_rel_count = dp.load(TEST_DIR, TEST_NER, TEST_REL, VERBOSE, IS_EMO)
    if 'EXCLUDE_TEST' in params:
        test_instances4 = Util.exlude(test_instances4, params['EXCLUDE_TEST'])
        test_gold_instances = Util.exlude(test_gold_instances, params['EXCLUDE_TEST'])
    print("There are ", test_invalid_rel_count, " intersentence relations in the test.")
    result_file.write("There are "+ str(test_invalid_rel_count)+ " intersentence relations in the test.\n")

    test_triggers_empty_indices = dp.get_indices_with_empty_triggers(test_instances4)
    test_instances3 = dp.remove_instances_with_empty_triggers(test_instances4, test_triggers_empty_indices)
    test_instances2 = dp.generate_candidate_structures(test_instances3,filter, params['GENERALISATION'],params['INCLUDE_MENTION'], USE_FILTER,params['PARTIAL_ARGUMENT_MATCHING'], dp, VERBOSE)
    test_event_empty_indices = dp.get_indices_with_empty_events(test_instances2, USE_FILTER)
    test_instances = dp.remove_instances_with_empty_events(test_instances2, test_event_empty_indices)

if TASK == 'TUNE':
    test_new_gold = gdp.remove_instances_with_attributes(test_gold_instances, test_triggers_empty_indices)
    test_new_gold = gdp.remove_instances_with_attributes(test_new_gold, test_event_empty_indices)

# GENERATE IDS
train_instances_ids, train_unk_count = dp.instances_to_ids(train_instances, word_type2id, role_type2id,trigger_type2id,entity_type2id, USE_FILTER, VERBOSE)
print("There are ", train_unk_count, " unk tokens in the train.")
result_file.write("There are "+ str(train_unk_count)+ " unk tokens in the train.\n")
if TASK in ['TUNE', 'TEST'] :
    test_instances_ids, test_unk_count = dp.instances_to_ids(test_instances, word_type2id, role_type2id,trigger_type2id,entity_type2id, USE_FILTER, VERBOSE)
    print("There are ", test_unk_count, " unk tokens in the test.")
    result_file.write("There are "+ str(test_unk_count)+ " unk tokens in the test.\n")


assert len(train_instances_ids) == len(train_new_gold), "Error: len(train_instances_ids) != len(train_new_gold)"
if TASK in ['TUNE'] :
    assert len(test_instances_ids) == len(test_new_gold), "Error: len(test_instances_ids) != len(test_new_gold)"

train_instances_ids = gdp.generate_gold_actions(train_instances_ids, train_new_gold)
if TASK in ['TUNE'] :
    test_instances_ids = gdp.generate_gold_actions(test_instances_ids, test_new_gold)
if TASK in ['TEST']:
    test_instances_ids = gdp.generate_gold_actions(test_instances_ids)

train_instances_ids = Util.merge(train_instances, train_instances_ids)
if TASK in ['TUNE', 'TEST'] :
    test_instances_ids = Util.merge(test_instances, test_instances_ids)


# compute nestedness in train and test
# train_nestedness = Util.compute_nestedness(train_instances, train_new_gold, USE_FILTER)
# if TASK in ['TUNE'] :
#     test_nestedness = Util.compute_nestedness(test_instances, test_new_gold,  USE_FILTER)


#NESTEDNESS STATS
'''
if TASK == "TUNE":
    print("Train nestedness statistics:")
    print("Level : Events")
    for level, count in train_nestedness.items():
        print(str(level), ":", str(count))
    print("Test nestedness statistics:")
    print("Level : Events")
    for level, count in test_nestedness.items():
        print(str(level), ":", str(count))

    result_file.write("Train nestedness statistics:")
    result_file.write("Level : Events")
    for level, count in train_nestedness.items():
        result_file.write(str(level) + ":" + str(count))
    result_file.write("Test nestedness statistics:")
    result_file.write("Level : Events")
    for level, count in test_nestedness.items():
        result_file.write(str(level) +  ":" + str(count))
'''



# train_num_events_ids = Util.count_gold_events_in_ids(train_instances_ids, USE_FILTER)
# if TASK in ['TUNE'] :
#     test_num_events_ids = Util.count_gold_events_in_ids(test_instances_ids, USE_FILTER)

# if TASK in ['TUNE'] :
#     train_max_num_gold_actions_d, train_max_num_gold_actions = Util.max_num_gold_actions(train_instances_ids)
#     test_max_num_gold_actions_d, test_max_num_gold_actions = Util.max_num_gold_actions(test_instances_ids)
#     least_beam_size = max(train_max_num_gold_actions, test_max_num_gold_actions)
#     print("Least beam size:", least_beam_size)

# print("TRAIN set: ", train_num_events_ids)
# if TASK in ['TUNE'] :
#     print("TEST set: ", test_num_events_ids)


# #UNCOMMENT FOR TESTING
# start_id = 0
# end_id = 100
# train_instances = train_instances[start_id:end_id]
# train_instances_ids = train_instances_ids[start_id:end_id]
# test_instances = test_instances[start_id:end_id]
# test_instances_ids = test_instances_ids[start_id:end_id]
# if TASK in ['TUNE', 'TEST'] :
#     test_new_gold = test_new_gold[start_id:end_id]


#DEBUG
# for i in train_instances_ids:
#     for j in i:
#         data_file.write(str(j)+"\n")
# for i in test_instances_ids:
#     for j in i:
#         data_file.write(str(j)+"\n")
# data_file.close()



# # MODEL PREPARATION
model = dp.modeler(len(word_type2id), len(trigger_type2id), len(role_type2id), len(entity_type2id),params)
model.load_glove(params['EMBEDDING_PATH'], word_type2id)

optimizer = None
if params['OPTIMISER'] == 'Adam':
    optimizer = optimizers.Adam(amsgrad=True, weight_decay_rate=params['WEIGHT_DECAY_RATE'])
elif params['OPTIMISER'] == 'SGD':
    optimizer = optimizers.SGD(lr=float(params['LR']))
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))

if TASK in ["TUNE","TRAIN"]:
    print("TRAINING OR TUNING PARAMS USING TRAIN AND DEV SETS..")
    log.info("TRAINING OR TUNING PARAMS USING TRAIN AND DEV SETS..")

    # PREPARE DATA ITERATORS
    train_iter = iterators.SerialIterator(train_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=True,
                                          repeat=False)

    result_file.write("Epoch\tLoss\tTr_P\tTr_R\tTr_F\tTs_P\tTs_R\tTs_F\n")
    print("\nEpoch\tLoss\tTr_P\tTr_R\tTr_F\tTs_P\tTs_R\tTs_F\n")

    train_predictions = []
    train_fscores = []
    test_fscores = []

    train_epoch = 0

    prev_fscore = 0.0
    prev_recall = 0.0
    prev_train_fscore = 0.0

    training_patience = 0

    epoch_loss = 0.0
    batch_count = 0

    best_test_score = 0.0

    new_train_instance_ids = []
    while train_epoch < params['MAX_EPOCH']:

        train_batch = train_iter.next()
        train_prediction, batch_loss, _ = model(train_batch, TRAIN=True)

        # UNCOMMENT TO GENERATE THE COMPUTATIONAL GRAPH
        '''
        g = c.build_computational_graph([batch_loss])
        with open('myoutput.dot', 'w') as o:
            o.write(g.dump())
        '''

        epoch_loss += batch_loss
        batch_count += 1

        train_predictions.extend(train_prediction)
        new_train_instance_ids.extend(train_batch)
        model.cleargrads()
        batch_loss.backward()
        optimizer.update()

        if train_iter.is_new_epoch:
            train_epoch += 1

            # compute the train scores only when tuning
            if TASK == "TUNE":
                ave_loss = epoch_loss / batch_count

                batch_count = 0
                epoch_loss = 0

                dp.generate_prediction_file_for_sbm(new_train_instance_ids, train_predictions, filter, const.TRAIN_OUTPUT_DIR, train_all_file_ids, params['SPECIAL_ENTITIES'], params['POST_FILTER'] )
                train_fscore, train_recall, train_precision = postprocess.compute_f_score(const.TRAIN_OUTPUT_DIR, TRAIN_DIR, TEST_DIR, params['EVALUATION_SCRIPT'], params['INTERPRETER'], params['DATASET'], IS_TRAIN=True)
                train_fscores.append(train_fscore)

                # if args.test:
                #     result_file.write('{:02d}\t{:.04f}\t{:.04f}\t{:.04f}\t'.format(train_epoch, train_precision,train_recall,  train_fscore)+"\n")
                #     print('{:02d}\t{:.04f}\t{:.04f}\t{:.04f}\t'.format(train_epoch, train_precision, train_recall,  train_fscore), end='')
                # else:
                result_file.write('{:02d}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(train_epoch,ave_loss.data.tolist(),train_precision,train_recall,train_fscore))
                print('{:02d}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(train_epoch,ave_loss.data.tolist(),train_precision,train_recall,train_fscore),end='')


                train_predictions = []
                test_predictions = []
                epoch_early_update = 0
                epoch_num_events = 0
                test_num_events = 0
                epoch_num_gold_out_of_beam = 0
                epoch_num_nbest_false_pos = 0
                epoch_num_nbest_true_pos_addfix_below_thres = 0
                epoch_num_nbest_true_pos_addfix_above_thres = 0

                test_iter = iterators.SerialIterator(test_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=True,repeat=False)
                new_test_instance_ids = []

                test_fscore = 0.0
                test_recall = 0.0
                test_ave_loss = 0.0
                test_batch_count = 0
                test_epoch_loss = 0.0



                while True:
                    test_batch = test_iter.next()
                    test_prediction, batch_loss, _ = model(
                        test_batch)
                    test_predictions.extend(test_prediction)
                    new_test_instance_ids.extend(test_batch)

                    test_epoch_loss += batch_loss
                    test_batch_count += 1

                    if test_iter.is_new_epoch:

                        test_ave_loss = test_epoch_loss / test_batch_count
                        dp.generate_prediction_file_for_sbm(new_test_instance_ids, test_predictions, filter,const.TEST_OUTPUT_DIR,test_all_file_ids,params['SPECIAL_ENTITIES'],params['POST_FILTER'])
                        test_fscore, test_recall, test_precision = postprocess.compute_f_score(const.TEST_OUTPUT_DIR, TRAIN_DIR, TEST_DIR, params['EVALUATION_SCRIPT'], params['INTERPRETER'], params['DATASET'],IS_TRAIN=False)
                        test_fscores.append(test_fscore)


                        end = datetime.now()
                        running_time = end - start_time

                        result_file.write(
                            '\t{:10.4f}\t{:10.4f}\t{:10.4f}\t'.format(test_precision,test_recall, test_fscore))
                        result_file.write(str(running_time)+"\n")
                        print('\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(test_precision, test_recall, test_fscore), end='')
                        print("\t", running_time)
                        result_file.close()
                        break
                # if test_fscore <= prev_fscore:
                #     training_patience += 1
                # else:
                #     training_patience = 0
                # prev_fscore = test_fscore

                if test_fscore > best_test_score:
                    best_test_score = test_fscore
                    training_patience = 0

                    # copy contents of directory to a new one if it has found a new best dev score
                    # if test_fscore > best_test_score:
                    best_test_score = test_fscore
                    if train_epoch >= params['STARTING_EPOCH_FOR_OUTPUT']:
                        # save the model here
                        serializers.save_npz(const.MODEL_OUTPUT_FILE, model)
                        print("\nModel saved: ", const.MODEL_OUTPUT_FILE)

                        root_folder = params['ROOT_OUTPUT_COPY_FOLDER']
                        output_folder = params['OUTPUT_FOLDER']
                        epoch_output_folder = root_folder + params['EPOCH_OUTPUT_FOLDER']
                        if not os.path.exists(epoch_output_folder):
                            os.makedirs(epoch_output_folder)
                        from_folder = root_folder + params['EPOCH_OUTPUT_FOLDER'] + output_folder
                        to_folder = epoch_output_folder + output_folder + "_" + str(train_epoch)
                        copy_tree(from_folder, to_folder)
                else:
                    training_patience += 1
                if training_patience == params['PATIENCE']:
                    break



                result_file = open(const.OUTPUT_LOG_FILE, 'a')
                start_time = datetime.now()
            train_iter = iterators.SerialIterator(train_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=True,
                                                  repeat=False)
            new_train_instance_ids = []

            if TASK == "TRAIN":
                print("Epoch:", train_epoch)
                if train_epoch >= params['STARTING_EPOCH_FOR_OUTPUT']:
                    # save the model here
                    serializers.save_npz(const.MODEL_OUTPUT_FILE+"_"+str(train_epoch), model)
                    print("\nModel saved: ", const.MODEL_OUTPUT_FILE+"_"+str(train_epoch))

                    root_folder = params['ROOT_OUTPUT_COPY_FOLDER']
                    output_folder = output_folder
                    epoch_output_folder = root_folder + params['EPOCH_OUTPUT_FOLDER']
                    if not os.path.exists(epoch_output_folder):
                        os.makedirs(epoch_output_folder)
                    from_folder = root_folder + "sbm/model/evaluation/" + output_folder
                    to_folder = epoch_output_folder + output_folder + "_" + str(train_epoch)
                    copy_tree(from_folder, to_folder)
elif TASK == "TEST":
    print("Loading model...", end='')
    serializers.load_npz(params['MODEL'], model)
    print("Finished.")

    test_iter = iterators.SerialIterator(test_instances_ids, batch_size=params['BATCH_SIZE'], shuffle=True,repeat=False)

    test_fscore = 0.0
    test_predictions = []
    new_test_instance_ids = []
    print("GENERATING PREDICTIONS ON TEST SET..", end='')

    totalcount =  0

    while True:
        test_batch = test_iter.next()
        test_prediction, _, count = model(test_batch)
        test_predictions.extend(test_prediction)
        new_test_instance_ids.extend(test_batch)

        totalcount += count

        if test_iter.is_new_epoch:
            dp.generate_prediction_file_for_sbm(new_test_instance_ids, test_predictions, filter,const.TEST_OUTPUT_DIR, test_all_file_ids,params['SPECIAL_ENTITIES'], params['POST_FILTER'])
            if params['SCENARIO'] == 1:
                print("scenario 1")
                test_fscore, test_recall, test_precision = postprocess.compute_f_score(const.TEST_OUTPUT_DIR, TRAIN_DIR,
                                                                                       TEST_DIR, params['EVALUATION_SCRIPT'], params['INTERPRETER'], params['DATASET'],IS_TRAIN=False)
                print('\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(test_precision, test_recall, test_fscore), end='')
                result_file.write('\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(test_precision, test_recall,test_fscore) + "\n")
            elif params['SCENARIO'] == 0:
                postprocess.convert_tr_to_t(const.TEST_OUTPUT_DIR)
            break
    print("Finished.")
    print("Total Classifications:", totalcount)
end = datetime.now()
total_time = end - first_start_time
print("\nRunning time:", total_time)
result_file = open(const.OUTPUT_LOG_FILE, 'a')
result_file.write("\nRunning time:" + str(total_time))
result_file.close()