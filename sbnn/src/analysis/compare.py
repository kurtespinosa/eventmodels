'''
Compare two a2 files
a)
'''

import collections
import os


def compare_files(gold, pred):
    # gold = '/Users/kurt/Desktop/exm/dataset/input/tempdev32221112/PMID-9840923.a2'
    # pred = '/Users/kurt/Desktop/exm/model/evaluation/exm_gold_32221112/pred/PMID-9840923.a2'


    gold_lines = open(gold, 'r').readlines()
    pred_lines = open(pred, 'r').readlines()

    # def get_trigger(event):
    #
    def get_sub_events(args):
        subs = []
        for s in args:
            arg = s.split(":")[1]
            if arg.startswith("E"):
                subs.append(arg)
        return subs

    def get_nonsub_events(args):
        nonsub = []
        for s in args:
            arg = s.split(":")[1]
            if arg.startswith("T"):
                nonsub.append(arg)
        return nonsub

    def get_event_and_trigger_dict(lines):
        event_dict = dict()
        trigger_dict = collections.defaultdict(list)
        for line in lines:
            if line.startswith("E"):
                tokens = line.split()
                event_id = tokens[0]
                trigger_id = tokens[1].split(":")[1]
                structure = tokens[2:]
                event_dict[event_id] = tokens[1:]
                trigger_dict[trigger_id].append(tokens[:])
        return event_dict, trigger_dict

    def compare(gold, pred):
        gold_ctr = collections.Counter(gold)
        pred_ctr = collections.Counter(pred)
        if gold_ctr == pred_ctr:
            return True
        return False

    def recurse_compare(gold, pred):
        # trigger_id = gold[0].split(":")[1]
        gold_args = gold[1:]
        pred_args = pred[1:]
        same = False
        subevents_gold = get_sub_events(gold_args)
        subevents_pred = get_sub_events(pred_args)
        if len(subevents_gold) == 0 and len(subevents_pred) == 0:
            same = compare(gold, pred)
        else:
            for s in subevents_gold:
                gold_event = gold_event_dict[s]
                for p in subevents_pred:
                    pred_event = pred_event_dict[p]
                    same = recurse_compare(gold_event, pred_event)
                    if same:
                        break
                if not same:
                    same = False
                    break
            if same:
                nonsub_gold = get_nonsub_events(gold_args)
                nonsub_pred = get_nonsub_events(pred_args)
                same = compare(nonsub_gold, nonsub_pred)

        return same

    gold_event_dict, gold_trigger_dict = get_event_and_trigger_dict(gold_lines)
    pred_event_dict, pred_trigger_dict = get_event_and_trigger_dict(pred_lines)

    found_ctr = 0
    notfound_ctr = 0

    for event_id, structure in gold_event_dict.items():
        trigger_id = structure[0].split(":")[1]
        pred_args = pred_trigger_dict[trigger_id]
        found = False
        # if len(pred_args) == 0:
        #     # print("Not found")
        #     fou
        # else:
        #
        gold_args = structure
        for p in pred_args:

            p_args = p[1:]
            gold_sub = get_sub_events(gold_args)
            if len(gold_sub) == 0:
                same = compare(gold_args, p_args)
                if same:
                    print("Found")
                    found_ctr += 1
                    found = True
                    break
        if not found:
            subevents = get_sub_events(gold_args)
            found = False
            if len(subevents) == 0:
                for p in pred_args:
                    try:
                        if len(get_sub_events(p[2:])) == 0:
                            same = compare(gold_args, p[1:])
                            if same:
                                found = True
                    except:
                        print("analysis")
                if found:
                    print("Found")
                    found_ctr += 1
            else:
                for p in pred_args:
                    same = recurse_compare(structure, p[1:])
                    if same:
                        found = True
                        print("Found")
                        found_ctr += 1
                        break
            if not found:
                print("Not found:", event_id, " ", structure)
                notfound_ctr += 1

    total = found_ctr + notfound_ctr
    print("Total:", total)
    print("Found:", found_ctr/total*100)
    return total, found_ctr, notfound_ctr

gold_dir = "/Users/kurt/Desktop/events/dataset/tees/CG13/dev/"
pred_dir = "/Users/kurt/Desktop/events/epoch-output-folder/sbnn_cg13_20190804180242/40/pred/"
# gold_dir = "/Users/kurt/Desktop/sbm/dataset/input/train2/"
# pred_dir = "/Users/kurt/Desktop/sbm/model/evaluation/EPOCH_OUTPUT_FOLDER/sbm_gold_03_62/pred/"
# gold_dir = '/Users/kurt/Desktop/sbm/dataset/emo/train_dev/dev/'
# pred_dir = '/Users/kurt/Desktop/sbm/model/evaluation/sbm_emo_03/pred/'
# gold_dir = '/Users/kurt/Desktop/sbm/dataset/emo/train_dev/tempdev/'
# pred_dir = '/Users/kurt/Desktop/sbm/model/evaluation/sbm_emo_xx/pred/'

gdir = os.listdir(gold_dir)
pdir = os.listdir(pred_dir)

total_ctr = 0
found_ctr = 0
notfound_ctr = 0
for i in range(len(gdir)):
    file = gdir[i]
    # if file == 'PMID-22172720.a2':
    #     print("here")
    if file.endswith(".a2"):
        # gold = gold_dir+file+"_temp" # this file uses TR for triggers
        gold = gold_dir + file + ""  # this file uses TR for triggers
        pred = pred_dir+file
        print("Gold:", gold)
        print("Pred:", pred)
        total, found, notfound = compare_files(gold, pred)
        total_ctr += total
        found_ctr += found
        notfound_ctr += notfound


print("Total:", total_ctr)
print("Found:", found_ctr)
print("Found:", found_ctr/total_ctr*100)