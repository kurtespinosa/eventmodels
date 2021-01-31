import os

import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from collections import OrderedDict

def produce_rel_and_ne(DIR):
    files = os.listdir(DIR)
    for file in files:
        tokens = file.split(".")
        if len(tokens) == 3:
            file_id = tokens[0]
            if tokens[2] == 'ann':
                file_obj = open(DIR+file, 'r')
                file_lines = file_obj.readlines()
                rel_file = open(DIR+file_id+".split.rel.ann", 'w')
                ner_file = open(DIR + file_id + ".split.ner.ann", 'w')
                for line in file_lines:
                    line_tokens = line.split()
                    if line_tokens[0].startswith("T"): #entity or trigger
                        ner_file.write(line)
                    elif line_tokens[0].startswith("R"):
                        rel_file.write(line)
                rel_file.close()
                ner_file.close()

def reformat_rel(DIR):
    files = os.listdir(DIR)
    for file in files:
        tokens = file.split(".")
        if len(tokens) >= 3:
            file_id = tokens[0]
            # print(tokens)
            if tokens[2] == 'rel':
                file_obj = open(DIR + file, 'r')
                file_lines = file_obj.readlines()
                rel_file = open(DIR + file_id + ".split.rel.ann", 'w')
                for line in file_lines:
                    line_tokens = line.split()

                    #If Fenia changes the output, comment the two lines below.
                    arg1 = line_tokens[2].split(":")[1]
                    arg2 = line_tokens[3].split(":")[1]

                    # arg1 = line_tokens[2]
                    # arg2 = line_tokens[3]

                    role = line_tokens[1]
                    ending1 = role[-1]
                    if ending1.isdigit():#remove the numbering
                        role = role[0:-1]

                    new_line = line_tokens[0]+"\t"+role+" "+arg1+" "+arg2
                    rel_file.write(new_line+"\n")
                rel_file.close()

def recreate_a1_a2(DIR):
    files = os.listdir(DIR)
    for file in files:
        if file.endswith("split.ann"):
            file_id = file.split(".split.ann")[0]
            file_obj = open(DIR + file, 'r')
            file_lines = file_obj.readlines()
            a1_file = open(DIR + file_id + ".a1", 'w')
            a2_file = open(DIR + file_id + ".a2_temp", 'w')
            for line in file_lines:
                if line.startswith("TR"):
                    a2_file.write(line)
                elif line.startswith("T"):
                    if line.split()[1] in ['Protein_domain_or_region', 'DNA_domain_or_region']:
                        a2_file.write(line)
                    else:
                        a1_file.write(line)
                elif line.startswith("E"):
                    a2_file.write(line)
            a1_file.close()
            a2_file.close()


def change_tr_to_t(DIR):
    files = os.listdir(DIR)
    for file in files:
        if file.endswith(".a2_temp"):
            file_id = file.split(".a2_temp")[0]
            file_write = open(DIR+file_id+".a2", 'w')
            lines = open(DIR+file, 'r').readlines()
            for line in lines:
                new_line = ''
                if line.startswith("E"):
                    tokens = line.split()
                    type_and_trig_id = tokens[1]
                    event_type, trigger_id = type_and_trig_id.split(":")
                    new_trigger_id = "T"+trigger_id[2:]
                    the_rest = ''
                    if len(tokens) > 2:
                        for token in tokens[2:]:
                            new_token = token.split(":")
                            new_role = new_token[0]
                            try:
                                new_arg = new_token[1]
                            except:
                                print("here")
                            if new_arg.startswith("TR"):
                                new_arg = "T" + new_token[1][2:]
                            combined = new_role + ":" + new_arg
                            the_rest += combined + " "
                    new_line = tokens[0]+"\t"+event_type+":"+new_trigger_id+" "+the_rest.rstrip()
                else:
                    tokens = line.split()
                    trig_id = tokens[0]
                    if trig_id.startswith("TR"):
                        trig_id = "T" + trig_id[2:]
                    new_line = trig_id+"\t"+tokens[1]+" "+tokens[2]+" "+tokens[3]+"\t"+' '.join(tokens[4:])
                file_write.write(new_line+"\n")
            file_write.close()


def _take_ids_in_gold_ner(DIR, filetype):
    files = os.listdir(DIR)
    for file in files:
        if file.endswith(filetype):
            file_id = file.split(".")[0]
            gold_type_lines = open(DIR+file_id+".split.ann", 'r').readlines()
            pred_type_lines = open(DIR + file, 'r').readlines()
            gold_dict = OrderedDict()
            for line in gold_type_lines:
                tokens = line.split()
                gold_dict[tokens[0]] = tokens[1:]
            pred_dict = OrderedDict()
            for line in pred_type_lines:
                tokens = line.split()
                pred_dict[tokens[0]] = tokens[1:]

            final_dict = OrderedDict()
            for k,v in pred_dict.items():
                if k in gold_dict:
                    final_dict[k] = v
            new_file = open(DIR+file_id+filetype, 'w')
            for k,v in final_dict.items():
                word = ' '.join(v[3:])
                line = k+"\t"+v[0]+" "+v[1]+" "+v[2]+"\t"+word
                new_file.write(line+"\n")
            new_file.close()


def _take_ids_in_gold_rel(DIR, filetype):
    files = os.listdir(DIR)
    for file in files:
        if file.endswith(filetype):
            file_id = file.split(".")[0]
            gold_type_lines = open(DIR + file_id + ".split.ann", 'r').readlines()
            pred_type_lines = open(DIR + file, 'r').readlines()
            gold_dict = OrderedDict()
            for line in gold_type_lines:
                tokens = line.split()
                gold_dict[tokens[0]] = tokens[1:]
            pred_dict = OrderedDict()
            for line in pred_type_lines:
                tokens = line.split()
                pred_dict[tokens[0]] = tokens[1:]

            final_dict = OrderedDict()
            for k, v in pred_dict.items():
                if k in gold_dict:
                    final_dict[k] = v
            new_file = open(DIR + file_id + filetype, 'w')
            for k, v in final_dict.items():
                line = k + "\t" + v[0] + " " + v[1] + " " + v[2]
                new_file.write(line + "\n")
            new_file.close()


# produce NER.ANN and REL.ANN from SPLIT.ann
def prepare_train_data(TRAIN_DIR, TEST_DIR):
    PRODUCE_NER_REL_AND_FORMAT = True
    dev = True
    if PRODUCE_NER_REL_AND_FORMAT:
        produce_rel_and_ne(TRAIN_DIR)
        reformat_rel(TRAIN_DIR)
        if dev:
            produce_rel_and_ne(TEST_DIR)
            reformat_rel(TEST_DIR)

def prepare_test_date(TRAIN_DIR, TEST_DIR):
    FORMAT_REL = True
    dev = True
    if FORMAT_REL:
        reformat_rel(TRAIN_DIR)
        if dev:
            reformat_rel(TEST_DIR)

def create_scenario1_data(TRAIN_DIR, TEST_DIR):
    BREAK_SPLIT_A1_AND_A2 = True
    if BREAK_SPLIT_A1_AND_A2:
        recreate_a1_a2(TRAIN_DIR)
        recreate_a1_a2(TEST_DIR)
        change_tr_to_t(TRAIN_DIR)
        change_tr_to_t(TEST_DIR)

def take_ids_in_gold(TRAIN_DIR, TEST_DIR):
    TAKE_IDS_IN_GOLD = False
    dev = True
    if TAKE_IDS_IN_GOLD:
        _take_ids_in_gold_ner(TRAIN_DIR, ".split.ner.ann")
        _take_ids_in_gold_rel(TRAIN_DIR, ".split.rel.ann")
        if dev:
            _take_ids_in_gold_ner(TEST_DIR, ".split.ner.ann")
            _take_ids_in_gold_rel(TEST_DIR, ".split.rel.ann")


# TRAIN_DIR = "/Users/kurt/Desktop/teesdataset/CG13-single/train/"
# TEST_DIR = "/Users/kurt/Downloads/cg_dev_preds_RE-predTR/cg_dev_preds_l8_triggers/"
# prepare_train_data(TRAIN_DIR, TEST_DIR)

# # create_scenario1_data(TRAIN_DIR, TEST_DIR)
# reformat_rel(TEST_DIR)