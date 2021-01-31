'''
Given events check count how many nested events and overlapping events per event type
'''

import collections
import os



def count_nested(file):
    nested_count_per_trigger = dict()
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            if line.startswith("E"):
                print(line)
                tokens = line.split()
                trigger = tokens[1].split(":")[0]
                for token in tokens[1:]:
                    arg = token.split(":")[1]
                    if arg.startswith("E"):
                        print(arg)
                        if trigger in nested_count_per_trigger:

                            nested_count_per_trigger[trigger] += 1
                        else:
                            nested_count_per_trigger[trigger] = 1
                        break
    total = 0
    for type, count in nested_count_per_trigger.items():
        total = total + count

    return nested_count_per_trigger, total
def count_events(file):
    count = 0
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            if line.startswith("E"):
                count += 1
    return count

def count_overlap(file):

    arg_event = collections.defaultdict(list)
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            if line.startswith("E"):
                print(line)
                tokens = line.split()
                event = tokens[0]
                for token in tokens[2:]:
                    arg = token.split(":")[1]
                    arg_event[arg].append(event)
    total = 0
    overlap = dict()
    for arg, lst in arg_event.items():
        if len(lst) > 1:
            overlap[arg] = lst
            total = total + len(lst)

    return overlap, total

def extract_remaining(file, contents_nested, contents_overlap):
    nested_overlap = set()
    for n in contents_nested:
        nested_overlap.add(n)
    for n in contents_overlap:
        nested_overlap.add(n)

    contents = dict()
    with open(file, 'r') as fileread:
        lines = fileread.readlines()
        for line in lines:
            tokens = line.split()
            id = tokens[0]
            contents[id] = line

    contents_set = set()
    for k,v in contents.items():
        contents_set.add(k)
    remaining = contents_set - nested_overlap

    final = dict()
    for k,v in contents.items():
        if k.startswith("T") or (k.startswith("E") and k in remaining):
            final[k] = v

    return final

##GOLD
# inputdir = "/Users/kurt/Desktop/events/dataset/tees/CG13/temp1/"
##EXM
# inputdir = "/Users/kurt/Desktop/EPOCH_OUTPUT_FOLDER/exm_tees_gold_cg13_05_02_8/pred/"
##TEES
# inputdir = "/Users/kurt/Desktop/dataset/tees/devrelationpreds/CG13-devel/unmerging/"
##SBM
inputdir = "/Users/kurt/Desktop/events/dataset/tees/CG13-single/gold/dev/"

# inputdir = "/Users/kurt/Desktop/events/dataset/tees/CG13/dev/"

#SBM
# inputdir = "/Users/kurt/Desktop/events/sbm/model/evaluation/sbm_tees_gold_cg13_dev_08_27_23_01/pred/"

#TEES
# inputdir = "/Users/kurt/Desktop/events/dataset/tees/devrelationpreds/CG13-devel/dev/"

#Exm
# inputdir = "/Users/kurt/Desktop/events/EPOCH_OUTPUT_FOLDER/exm_tees_gold_cg13_05_02_8/pred/"

files = os.listdir(inputdir)
overlap_total = 0
nested_total = 0
total_events = 0
nested_types = set()
for file in files:
    print(file)
    if file.endswith(".a2"):
        overlap, overlap_count = count_overlap(inputdir+file)
        nested, nested_count = count_nested(inputdir+file)
        events_count = count_events(inputdir+file)
        overlap_total += overlap_count
        nested_total += nested_count
        total_events += events_count

        for type, count in nested.items():
            nested_types.add(type)
        print(overlap)
        print(nested)

print("overlap_total:", overlap_total)
print("nested_total:", nested_total)
print(nested_types)
print("total events:", total_events)


