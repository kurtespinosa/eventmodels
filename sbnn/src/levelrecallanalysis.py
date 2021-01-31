'''
Compares the dev gold and dev pred files
sbm_gold_ra.yaml

From the a2 files in pred, do the following:
1. copy the a1 files from dev to the pred folder
2. copy the .split.txt and .txt to the pred folder
3. generate split.ann using the process_tees_outputs.. script
4. generate split.rel.ann, split.ner.ann using the same script
5. modify the dataset in the yaml file and run this script
'''

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

import constants as const
from pipeline.util import Util
from pipeline.Gold import Gold
import pipeline.preprocess as preproc

from datetime import datetime
import argparse

from model.Model import Model
import pipeline.postprocess as postprocess

from distutils.dir_util import copy_tree

import collections

first_start_time = datetime.now()
start_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', type=str, required=True, help='yaml file')
# parser.add_argument('--train', action='store_true', help='training only to get final model')
# parser.add_argument('--test', action='store_true', help='testing mode - predicts only; needs a model to load')

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

#factor common params
IS_EMO = params['IS_EMO']
TRAIN_DIR = params['TRAIN_DIR']
VERBOSE = params['VERBOSE']
TEST_DIR = params['TEST_DIR']
TRAIN_NER = params['TRAIN_NER']
TRAIN_REL = params['TRAIN_REL']
TEST_NER = params['TEST_NER']
TEST_REL = params['TEST_REL']
USE_FILTER = params['USE_FILTER']

TASK = params['TASK']

# GOLD DATA PROCESSING
train_gold_instances, train_invalid_event_count, file_id_events, train_num_events = Gold.load(TRAIN_DIR, VERBOSE, IS_EMO)
print("TRAIN set: ", train_num_events, " gold events and of which ", train_invalid_event_count, " are intersentence events.")
result_file.write("TRAIN set: " + str(train_num_events) + " gold events and of which " + str(train_invalid_event_count) + " are intersentence events.\n")
train_gold_instances = Gold.denumber_roles(train_gold_instances)

test_gold_instances, test_invalid_event_count, _, test_num_events = Gold.load(TEST_DIR, VERBOSE, IS_EMO)
print("TEST set: ", test_num_events, " gold events and of which ", test_invalid_event_count, " are intersentence events.")
result_file.write("TEST set: " + str(test_num_events) + " gold events and of which " + str(test_invalid_event_count) + " are intersentence events.\n")
test_gold_instances = Gold.denumber_roles(test_gold_instances)


# TRAINING DATA PROCESSING
train_instances4, train_all_file_ids, train_invalid_rel_count = Model.load(TRAIN_DIR, TRAIN_NER, TRAIN_REL, VERBOSE, IS_EMO)
print("There are ", train_invalid_rel_count, " intersentence relations in the train.")
result_file.write("There are "+ str(train_invalid_rel_count)+ " intersentence relations in the train.\n")

train_triggers_empty_indices = Model.get_indices_with_empty_triggers(train_instances4)
train_instances3 = Model.remove_instances_with_empty_triggers(train_instances4, train_triggers_empty_indices)
train_new_gold = Gold.remove_instances_with_attributes(train_gold_instances, train_triggers_empty_indices)
train_instances2 = Model.generate_candidate_structures(train_instances3,filter, params['GENERALISATION'],params['INCLUDE_MENTION'], USE_FILTER,params['PARTIAL_ARGUMENT_MATCHING'])

train_event_empty_indices = Model.get_indices_with_empty_events(train_instances2, USE_FILTER)
train_instances = Model.remove_instances_with_empty_events(train_instances2, train_event_empty_indices)
train_new_gold = Gold.remove_instances_with_attributes(train_new_gold, train_event_empty_indices)

'''
From the given .a2 predictions, generate the split.ann and then the split.ner.ann and split.rel.ann
Then replace .a2 with the original dev .a2 file as well as the split.ann and retain only the ner.ann and rel.ann.
Test instances events should contain the predicted events sorted in levels.
'''

# TEST DATA PROCESSING
test_instances4, test_all_file_ids, test_invalid_rel_count = Model.load(TEST_DIR, TEST_NER, TEST_REL, VERBOSE, IS_EMO)
print("There are ", test_invalid_rel_count, " intersentence relations in the test.")
result_file.write("There are "+ str(test_invalid_rel_count)+ " intersentence relations in the test.\n")

test_triggers_empty_indices = Model.get_indices_with_empty_triggers(test_instances4)
test_instances3 = Model.remove_instances_with_empty_triggers(test_instances4, test_triggers_empty_indices)
test_instances2 = Model.generate_candidate_structures(test_instances3,filter, params['GENERALISATION'],params['INCLUDE_MENTION'], USE_FILTER,params['PARTIAL_ARGUMENT_MATCHING'])
test_event_empty_indices = Model.get_indices_with_empty_events(test_instances2, USE_FILTER)
test_instances = Model.remove_instances_with_empty_events(test_instances2, test_event_empty_indices)

test_new_gold = Gold.remove_instances_with_attributes(test_gold_instances, test_triggers_empty_indices)
test_new_gold = Gold.remove_instances_with_attributes(test_new_gold, test_event_empty_indices)



# assert len(train_instances) == len(test_instances), "Error: different lengths"


# compute nestedness in train and test
train_nestedness = Util.compute_nestedness(train_instances, train_new_gold, USE_FILTER)
test_nestedness = Util.compute_nestedness(test_instances, test_new_gold,  USE_FILTER)

# assert len(train_new_gold) == len(train_instances) == len(test_instances) == len(test_new_gold), "Error: unequal length"


gold_per_level = dict()
predicted_per_level = dict()

for i in range(len(train_new_gold)):
    gold_events = train_new_gold[i][3]
    gold_levels = train_instances[i][5]
    pred_events = test_new_gold[i][3]
    pred_levels = test_instances[i][5]


    # gold: create a map betweeen a trigger and its associated events
    gold_trig_event_map = Util.create_trig_event_map(gold_events)
    pred_trig_event_map = Util.create_trig_event_map(pred_events)

    # loop thru triggers in the gold levels and compare to the levels in pred if events are predicted

    for g in range(len(gold_levels)):
        for trig, _ in gold_levels[g].items():
            g_events = gold_trig_event_map[trig]
            p_events = pred_trig_event_map[trig]

            max_events = len(p_events)
            pred_idx = [k for k in range(len(p_events))]  # stores all the indices

            #check if gold events are in pred events
            for e in g_events:
                c_gold = Util.replace_event_arg_with_trigger(e, gold_events)
                c_gold = collections.Counter(c_gold)
                if g in gold_per_level:
                    gold_per_level[g] += 1
                else:
                    gold_per_level[g] = 1
                predicted = False

                if pred_idx:
                    ctr = 0
                    while ctr < len(pred_idx):
                        idx = pred_idx[ctr]
                        c_pred = Util.replace_event_arg_with_trigger(p_events[idx], pred_events)
                        c_pred = collections.Counter(c_pred)
                        if c_gold == c_pred:
                            predicted = True
                            if g == 4:
                                print("analysis")
                            if g in predicted_per_level:
                                predicted_per_level[g] += 1
                            else:
                                predicted_per_level[g] = 1
                            pred_idx.remove(idx)
                            break
                        else:
                            ctr += 1
                    # if not predicted:
                    #     print("test")
            # print("test")

print("Gold events per level:", gold_per_level)
print("Predicted events per level:", predicted_per_level)

print("End")
