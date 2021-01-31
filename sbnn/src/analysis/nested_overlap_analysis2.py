'''
Computes precision and recall of nested and overlapping events
precision = predicted overlapping events over all gold events
recall = gold overlapping events over all predicted events

- detect the nested/overlapping events
- then write to a different folder
- then evaluate


**** new ****
# precision=correctly predicted overlapping events over predicted overlapping events, and
# recall=covered gold overlapping events over gold overlapping events
#
# precision = x/y
#
# x : predicted events that are gold overlapping events
# y : predicted overlapping events
#
# recall = x/y
# x : predicted events that are gold overlapping events
# y : gold overlapping events

'''

import os
import collections

#detect nested events
'''
Nested events are those events with argument that are events
'''
def extract_fscore(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        fscore = ''
        for line in lines:
            if line.split()[0] == '===[SUB-TOTAL]===' or line.split()[0] == '====[TOTAL]====' or line.split()[0] == '==[ALL-TOTAL]==':
                tokens = line.split()
                recall = tokens[-3]
                precision = tokens[-2]
                fscore = tokens[-1]
                break
    return float(fscore.strip()), float(recall.strip()), float(precision.strip())

def get_event_args(event):
    events = []
    for a in event:
        arg = a.split(":")[1]
        if arg.startswith("E"):
            events.append(arg)
    return events
def detect_nested_events(file):
    contents = dict()
    events = dict()
    arg_events = set()
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            tokens = line.split()
            id = tokens[0]
            if line.startswith("E"):
                args = get_event_args(tokens[2:])
                events[id] = line
                if len(args) > 0:
                    contents[id] = line
                    for a in args:
                        arg_events.add(a)
            elif line.startswith("T"):
                contents[id] = line


    #num nested
    num_nested = 0
    nested = dict()
    for id, line in contents.items():
        if id.startswith("E"):
            num_nested += 1
            nested[id] = line

    # adding the sub-events that are not nested events for event matching purposes
    for arg in arg_events:
        if arg not in contents:
            contents[arg] = events[arg]

    total_events = len(events)
    return contents, nested, num_nested, total_events

def detect_nonnested_events(file):
    contents = dict()
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            tokens = line.split()
            id = tokens[0]
            if line.startswith("E"):
                args = get_event_args(tokens[2:])
                if len(args) == 0:
                    contents[id] = line
            elif line.startswith("T"):
                contents[id] = line
    return contents

def detect_overlapping_events(file):
    # Build a dict: [argument] = {events containing the argument}
    arg_event = collections.defaultdict(set)
    events = dict()
    contents = dict()
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            tokens = line.split()
            id = tokens[0]
            if line.startswith("E"):
                tokens = line.split()
                for token in tokens[2:]:
                    arg = token.split(":")[1]
                    arg_event[arg].add(id)
                contents[id] = line
            elif line.startswith("T"):
                contents[id] = line

    # num of events that are overlapping
    overlap = set() #an event id is counted only once even if it participates in more than one overlap
    for arg, common in arg_event.items():
        if len(common) > 1: #there should be at least two events sharing an argument for there to be an overlapping
            for c in common:
                overlap.add(c)
    num_overlap = len(overlap)

    #get the overlapping event definitions
    overlapping = dict()
    for i in overlap:
        overlapping[i] = contents[i]


    ### from here expand the events to include their sub-events, for matching purposes
    #unique overlaps
    unique = set()
    for arg, common in arg_event.items():
        if len(common) > 1:
            unique.add(arg) # add for matching purposes
            for c in common:
                unique.add(c)

    # add events that are arguments to unique but not common
    events_toadd = set()
    for i in unique:
        if i.startswith("E") and i in contents: #not an entity
            event = contents[i]
            tokens = event.split()
            if len(tokens) > 2:
                args = tokens[2:]
                for a in args:
                    arg = a.split(":")[1]
                    if arg.startswith("E"):
                        events_toadd.add(arg)

    #check recursively the events that need to be added by adding events_toadd
    for e in events_toadd:
        unique.add(e)
    while True:
        more_events = set()
        for e in events_toadd:
            if e in contents:
                event = contents[e]
                tokens = event.split()
                if len(tokens) > 2:
                    args = tokens[2:]
                    for a in args:
                        arg = a.split(":")[1]
                        if arg.startswith("E"):
                            more_events.add(arg)

        for k in more_events:
            unique.add(k)
        events_toadd = more_events
        if len(events_toadd) == 0:
            break

    #get all the unique events + triggers in contents
    finalcontents = dict()
    for id, line in contents.items():
        if id.startswith("T"):
            finalcontents[id] = line
        elif id.startswith("E"):
            if id in unique:
                finalcontents[id] = line

    #total events
    total_events = 0
    for id, _ in contents.items():
        if id.startswith("E"):
            total_events += 1
    return finalcontents, overlapping, num_overlap, total_events

def detect_nonoverlapping_events(file):
    arg_event = collections.defaultdict(set)
    events = dict()
    contents = dict()
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            tokens = line.split()
            id = tokens[0]
            if line.startswith("E"):
                tokens = line.split()
                for token in tokens[2:]:
                    arg = token.split(":")[1]
                    arg_event[arg].add(id)
                contents[id] = line
            elif line.startswith("T"):
                contents[id] = line
    # unique overlaps
    unique = set()
    for arg, common in arg_event.items():
        if len(common) > 1:
            unique.add(arg)
            for c in common:
                unique.add(c)

    #add the events that are not in unique
    final = dict()
    args_to_add = []
    for id, line in contents.items():
        if id.startswith("T"):
            final[id] = line
        elif id.startswith("E"):
            if id not in unique:
                tokens = line.split()
                if len(tokens) > 2:
                    args = tokens[2:]
                    valid = True
                    events = []
                    for a in args:
                        arg = a.split(":")[1]
                        if arg.startswith("E"):
                            events.append(arg)
                        if arg in unique:
                            valid = False
                            break
                    if valid: #exclude events with sub-events that are overlapping
                        final[id] = line
                        args_to_add.extend(events) #add the arguments since these sub-events are nonoverlap


    for a in args_to_add:
        final[a] = contents[a]

    # remove any events that has sub-event not in there
    while True:
        to_remove = []
        for id, line in final.items():
            if line.startswith("E"):
                tokens = line.split()
                if len(tokens) > 2:
                    args = tokens[2:]
                    for a in args:
                        arg = a.split(":")[1]
                        if arg.startswith("E"):
                            if arg not in final:
                                to_remove.append(id)
        if len(to_remove) == 0:
            break
        else:
            for r in to_remove:
                del final[r]

    return final

def extract_remaining(file, contents_nested, contents_overlap):
    #combine them
    nested_overlap = set()
    for n in contents_nested:
        nested_overlap.add(n)
    for n in contents_overlap:
        nested_overlap.add(n)

    #get all the contents, including triggers
    contents = dict()
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            tokens = line.split()
            id = tokens[0]
            contents[id] = line
    #subtract the combined nested_overlap from the contents, remaining contains triggers+flat events
    contents_set = set()
    for k,v in contents.items():
        contents_set.add(k)
    remaining = contents_set - nested_overlap

    #separate flat from triggers to get count of flat events
    flat = dict()
    # triggers = dict()
    # others = dict()
    for k in remaining:
    #     if k.startswith("T"):
    #         triggers[k] = v
    #     else:
        if k.startswith("E"):
            flat[k] = contents[k]

    # contents = {**flat, **triggers, **others}
    # assert len(contents) == len(remaining), "Error: should be equal"
    # if len(contents) != len(remaining):
    #     print()
    num_flat = len(flat)

    return contents, flat, num_flat




def evaluate(gold, tempdir, tempfile):
    try:
        INTERPRETER = "python3 "
        EVALUATION_SCRIPT = "/Users/kurt/Desktop/events/sbm/model/evaluation/evaluation-CG-modified.py"

        command = INTERPRETER + EVALUATION_SCRIPT + " -r " + gold + " -d " + tempdir + " > " + tempfile

        os.system(command)
    except Exception as e:
        print(e)
def write_to_file(dir, file, contents):
    with open(dir+file, 'w') as fwrite:
        for id, contents in contents.items():
            fwrite.write(contents)

def goldstatistics():
    gold = "/Users/kurt/Desktop/events/dataset/tees/GE13/dev/"
    files = os.listdir(gold)
    total_events = 0
    total_nested = 0
    total_overlap = 0
    total_flat = 0
    for file in files:
        if file.endswith(".a2"):
            contents, nested, num_nested, event_cnt1 = detect_nested_events(gold + file)
            finalcontents, overlapping, num_overlap, event_cnt2 = detect_overlapping_events(gold+file)
            contents, flat, num_flat = extract_remaining(gold + file, nested, overlapping)
            assert event_cnt1 == event_cnt2, "Error: not equal event count"
            total_nested += num_nested
            total_overlap += num_overlap
            total_flat += num_flat
            total_events += event_cnt2
    print(total_events, total_nested, total_overlap, total_flat)


def get_result(ref, pred, file):
    evaluate(ref, pred, file)
    fscore, recall, precision = extract_fscore(file)
    return precision, recall, fscore


##----------------
goldstatistics()

# gold = "/Users/kurt/Desktop/events/dataset/tees/CG13/dev/"
# models = ["sbm", 'tees', 'exm']
#
# for model in models:
#     pred_prec = None
#     pred_rec = None
#     if model == 'sbm':
#         pred_prec = "/Users/kurt/Desktop/events/sbm/model/evaluation/sbm_tees_gold_cg13_dev_08_27_23_01/pred/"
#         pred_rec = "/Users/kurt/Downloads/tempgoldsbm/"
#     if model == 'tees':
#         pred_prec = "/Users/kurt/Desktop/events/dataset/tees/devrelationpreds/CG13-devel/dev/"
#         pred_rec = "/Users/kurt/Downloads/tempgoldtees/"
#     if model == 'exm':
#         pred_prec = "/Users/kurt/Desktop/events/EPOCH_OUTPUT_FOLDER/exm_tees_gold_cg13_05_02_8/pred/"
#         pred_rec = "/Users/kurt/Downloads/tempgoldexm/"
#
#
#     root = "/Users/kurt/Desktop/events/sbm/model/analysis/results/"
#     nested = 'nested'
#     overlap = 'overlap'
#     remaining = 'remaining'
#     temp_model =  root+"temp/"+model+"/"
#     tempdir_nested = temp_model +nested+"/"
#     tempdir_overlap = temp_model +overlap+"/"
#     tempdir_remaining = temp_model +remaining+"/"
#     pred = "_pred.out"
#
#
#     contents = None
#
#     dirs = [tempdir_nested, tempdir_overlap, tempdir_remaining]
#     for d in dirs:
#         if not os.path.exists(d):
#             os.makedirs(d)
#         pred_dir = os.listdir(d)
#         for f in pred_dir:
#             os.remove(os.path.join(d, f))
#
#     print(model)
#     files = os.listdir(pred_prec)
#     for file in files:
#         if file.endswith(".a2"):
#             temp_file = pred_prec+file
#             contents_nested, _, _, _ = detect_nested_events(temp_file)
#             contents_overlap, _, _, _ = detect_overlapping_events(temp_file)
#             contents_flat, _, _ = extract_remaining(temp_file, contents_nested, contents_overlap)
#
#             write_to_file(tempdir_nested, file, contents_nested)
#             write_to_file(tempdir_overlap, file, contents_overlap)
#             write_to_file(tempdir_remaining, file, contents_flat)
#
#     pred_file = temp_model+"nested"+pred
#     precision_nested, _, _ = get_result(gold, tempdir_nested, pred_file)
#     # print("\t\tnested:",precision_nested)
#     pred_file = temp_model+"overlap"+pred
#     precision_overlap, _, _ = get_result(gold, tempdir_overlap, pred_file)
#     # print("\t\toverlap:", precision_overlap)
#     pred_file = temp_model+"remaining"+pred
#     precision_remaining, _, _ = get_result(gold, tempdir_remaining, pred_file)
#     # print("\t\tremaining:", precision_remaining)
#
#     dirs = [tempdir_nested, tempdir_overlap, tempdir_remaining]
#     for d in dirs:
#         if not os.path.exists(d):
#             os.makedirs(d)
#         pred_dir = os.listdir(d)
#         for f in pred_dir:
#             os.remove(os.path.join(d, f))
#
#     # print("\trecall")
#     files = os.listdir(gold)
#     for file in files:
#         if file.endswith(".a2"):
#             temp_file = gold + file
#             contents_nested, _, _, _ = detect_nested_events(temp_file)
#             contents_overlap, _, _, _ = detect_overlapping_events(temp_file)
#             contents_flat, _, _ = extract_remaining(temp_file, contents_nested, contents_overlap)
#
#             write_to_file(tempdir_nested, file, contents_nested)
#             write_to_file(tempdir_overlap, file, contents_overlap)
#             write_to_file(tempdir_remaining, file, contents_flat)
#     pred_file = temp_model + "nested" + pred
#     _, recall_nested, _ = get_result(pred_rec, tempdir_nested, pred_file)
#
#     pred_file = temp_model + "overlap" + pred
#     _, recall_overlap, _ = get_result(pred_rec, tempdir_overlap, pred_file)
#
#     pred_file = temp_model + "remaining" + pred
#     _, recall_remaining, _ = get_result(pred_rec, tempdir_remaining, pred_file)
#
#
#     fscore_nested = (2 * precision_nested * recall_nested) / (precision_nested + recall_nested)
#     fscore_overlap = (2 * precision_overlap * recall_overlap) / (precision_overlap + recall_overlap)
#     fscore_remaining = (2 * precision_remaining * recall_remaining) / (precision_remaining + recall_remaining)
#     print("\t\tnested:", fscore_nested)
#     print("\t\toverlap:", fscore_overlap)
#     print("\t\tremaining:", fscore_remaining)


#
# ########## Revised Computation ##############
#
# def extract_and_write_components(data, model, root):
#     '''
#     An event can be both nested and overlapping so they can be counted twice.
#     Given:
#         E1: A1 E2
#         E2: A2
#         E3: A1 A2
#     Nested: E1
#     Overlap: E1, E3
#     Flat: E2
#
#     E1 is counted twice.
#     '''
#     #create the directories
#     nested = 'nested'
#     overlap = 'overlap'
#     remaining = 'remaining'
#     temp_model =  root+"temp/"+model+"/"
#     tempdir_nested = temp_model +nested+"/"
#     tempdir_overlap = temp_model +overlap+"/"
#     tempdir_remaining = temp_model +remaining+"/"
#
#     #remove previous files or create the directories
#     dirs = [tempdir_nested, tempdir_overlap, tempdir_remaining]
#     for d in dirs:
#         if not os.path.exists(d):
#             os.makedirs(d)
#         pred_dir = os.listdir(d)
#         for f in pred_dir:
#             os.remove(os.path.join(d, f))
#
#     # loop through the gold/pred directories and extract and write the respective nested/overlap/flat events
#     files = os.listdir(data)
#     for file in files:
#         if file.endswith(".a2"):
#             temp_file = data+file
#             _, contents_nested, _, _ = detect_nested_events(temp_file)
#             _, contents_overlap, _, _ = detect_overlapping_events(temp_file)
#             _, contents_flat, _ = extract_remaining(temp_file, contents_nested, contents_overlap)
#
#             write_to_file(tempdir_nested, file, contents_nested)
#             write_to_file(tempdir_overlap, file, contents_overlap)
#             write_to_file(tempdir_remaining, file, contents_flat)
#     return tempdir_nested, tempdir_overlap, tempdir_remaining, temp_model
#
# # toggle model to compute f1 score respectively
# model = 'sbm' #tees, exm
#
# # set gold directory (same) and different prediction directories
# gold = "/Users/kurt/Desktop/events/dataset/tees/CG13/dev/"
# pred = None
# if model == 'sbm':
#     pred = "/Users/kurt/Desktop/events/sbm/model/evaluation/sbm_tees_gold_cg13_dev_08_27_23_01/pred/"
# elif model == 'tees':
#     pred = "/Users/kurt/Desktop/events/dataset/tees/devrelationpreds/CG13-devel/dev/"
# elif model == 'exm':
#     pred = "/Users/kurt/Desktop/events/EPOCH_OUTPUT_FOLDER/exm_tees_gold_cg13_05_02_8/pred/"
#
# #extract the nested, overlap, flat respectively
# root = "/Users/kurt/Desktop/events/sbm/model/analysis/results/" # where to extract the respective nested, overlap, flat events
# dir_gold_nested, dir_gold_overlap, dir_gold_flat, temp_model = extract_and_write_components(gold, model, root)
# dir_pred_nested, dir_pred_overlap, dir_pred_flat, _ = extract_and_write_components(pred, model, root)
#
# #evaluate precision, recall, fscore
# nested_p, nested_r, nested_f    = get_result(dir_gold_nested, dir_pred_nested, temp_model+"nested_pred.out")
# overlap_p, overlap_r, overlap_f = get_result(dir_gold_overlap, dir_pred_overlap, temp_model+"overlap_pred.out")
# flat_p, flat_r, flat_f          = get_result(dir_gold_flat, dir_pred_flat, temp_model+"flat_pred.out")


