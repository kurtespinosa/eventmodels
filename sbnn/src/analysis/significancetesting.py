'''
Test if there is a significance between the outputs of two systems a and b.
Null hypothesis: There is no significant difference between system a and system b.
Use approximate randomisation test as described here:
https://cs.stanford.edu/people/wmorgan/sigtest.pdf
If R values approaches p=0.5, then the null hypothesis is true.
'''


import os
import random
import collections

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


def get_sen_boundaries():
    txtdir = "/Users/kurt/Desktop/dataset/tees/CG13/dev/"
    files = os.listdir(txtdir)
    pmid_sen_boundaries = collections.defaultdict(list)
    for file in files:
        if file.endswith(".split.txt"):
            # print(file)
            pmid = file.split(".split.txt")[0]
            lines = open(txtdir+file, 'r').readlines()
            for i in range(len(lines)):
                if i % 2 == 0:
                    pmid_sen_boundaries[pmid].append(lines[i])
    # print(pmid_sen_boundaries)
    return pmid_sen_boundaries

def compute_diff(a_temp, b_temp):
    temp = "/Users/kurt/Downloads/"
    ref = '/Users/kurt/Desktop/dataset/tees/CG13/dev/'
    INTERPRETER = "python3 "
    EVALUATION_SCRIPT = "/Users/kurt/Desktop/sbm/model/evaluation/evaluation-CG-modified.py"
    a_pred = temp + "apred.out"
    b_pred = temp + "bpred.out"

    #evaluate each set of predictions
    command = INTERPRETER + EVALUATION_SCRIPT + " -r " + ref + " -d " + a_temp +  " > " + a_pred
    os.system(command)
    a_fscore, recall, precision = extract_fscore(a_pred)
    # print(a_fscore, recall, precision)

    command = INTERPRETER + EVALUATION_SCRIPT + " -r " + ref + " -d " + b_temp +  " > " + b_pred
    os.system(command)
    b_fscore, recall, precision = extract_fscore(b_pred)
    # print(b_fscore, recall, precision)

    diff = abs(a_fscore - b_fscore)
    return diff

def get_sentence_events(lines, pmid, pmid_sen_boundaries):
    def offset_in_sen(offset_b, sen_b):
        tlow, thigh = offset_b.split(":")
        slow, shigh = sen_b.split(":")

        inside = False
        if int(tlow) >= int(slow) and int(thigh) <= int(shigh):
            inside = True
        return inside

    def same_contents(lst):
        same = True
        p = lst[0]
        if len(lst) > 1:
            for l in lst[1:]:
                if l != p:
                    same = False
                    break
                p = l
        return same

    trig_offsets = dict()
    for line in lines:
        tokens = line.split()
        if line.startswith("T") and len(tokens) > 1:
            trigid = tokens[0]
            offsets = tokens[2]+":"+tokens[3]
            trig_offsets[trigid] = offsets
    sen_boundaries = pmid_sen_boundaries[pmid]

    #triggers per sentence number
    trig_in_sen = collections.defaultdict(list)
    for trig, offset in trig_offsets.items():
        for s in range(len(sen_boundaries)):
            if offset_in_sen(offset, sen_boundaries[s]):
                trig_in_sen[s].append(trig)

    sen_events = dict()
    event_offsets = collections.defaultdict(list)
    # just checking the offsets of the trigger
    for line in lines:
        tokens = line.split()
        if line.startswith("E"):
            eventid = tokens[0]
            trigid = tokens[1].split(":")[1]
            offsets = trig_offsets[trigid]
            event_offsets[eventid].append(offsets)
    # check offsets of the argument events if there are
    for line in lines:
        tokens = line.split()
        if line.startswith("E"):
            eventid = tokens[0]
            if len(tokens) > 2:
                args = tokens[2:]
                for arg in args:
                    argid = arg.split(":")[1]
                    if argid in event_offsets:
                        offsets = event_offsets[argid]#maybe a different offset
                        event_offsets[eventid].extend(offsets)
    #check if in the event_offsets some are different for same event id
    events_in_sen = collections.defaultdict(list)
    for eventid, lst in event_offsets.items():
        sens = []
        for offset in lst:
            for s in range(len(sen_boundaries)):
                if offset_in_sen(offset, sen_boundaries[s]):
                    sens.append(s)
        assert len(sens) > 0, "Error: len(sens) == 0"
        if same_contents(sens):
            events_in_sen[sens[0]].append(eventid)

    lines_sen = collections.defaultdict(list)
    for sen, trig in trig_in_sen.items():
        lines_sen[sen].extend(trig)
    for sen, events in events_in_sen.items():
        lines_sen[sen].extend(events)


    #extract lines with ids
    contents = dict()
    for line in lines:
        tokens = line.split()
        if line.startswith("T"):
            contents[tokens[0]] = tokens[1]+" "+tokens[2]+" "+tokens[3]+"\t"+' '.join(tokens[4:])
        elif line.startswith("E"):
            s = ''
            for t in tokens[1:]:
                s += t+" "
            news = s.rstrip()
            contents[tokens[0]] = news

    final_lines = collections.defaultdict(list)
    for sen, ids in lines_sen.items():
        newids = []
        for id in ids:
            if id in contents:
                newids.append(id+"\t"+contents[id])
        final_lines[sen] = newids

    return  final_lines


def shuffle_sentences():
    a_dir = "/Users/kurt/Desktop/sbm/model/evaluation/sbm_tees_gold_cg13_dev_08_27_23_01/pred/"
    b_dir = "/Users/kurt/Desktop/dataset/tees/devrelationpreds/CG13-devel/unmerging/"

    pmid_sen_boundaries = get_sen_boundaries()

    a_out = os.listdir(a_dir)

    a_temp = a_dir + "a_temp/"
    b_temp = b_dir + "b_temp/"
    if not os.path.exists(a_temp):
        os.makedirs(a_temp)
    if not os.path.exists(b_temp):
        os.makedirs(b_temp)

    #remove files
    pred_dir = os.listdir(a_temp)
    for f in pred_dir:
        os.remove(os.path.join(a_temp, f))
    pred_dir = os.listdir(b_temp)
    for f in pred_dir:
        os.remove(os.path.join(b_temp, f))


    # swap PMID predictions at p=0.5
    cnt_above = 0
    cnt_below = 0
    for file in a_out:
        if file.endswith(".a2"):
            pmid = file.split(".a2")[0]
            # print(file)

            a_file = a_dir + file
            b_file = b_dir + file
            with open(a_file, 'r') as fread:
                a_lines = fread.readlines()
                a_sentencelevel_events = get_sentence_events(a_lines, pmid, pmid_sen_boundaries)
            with open(b_file, 'r') as fread:
                b_lines = fread.readlines()
                b_sentencelevel_events = get_sentence_events(b_lines, pmid, pmid_sen_boundaries)

            alen = len(a_sentencelevel_events)
            blen = len(b_sentencelevel_events)
            looplen = max(alen,blen)
            for i in range(looplen):
                num = random.random()

                a_i = a_sentencelevel_events[i]
                b_i = b_sentencelevel_events[i]

                if num > 0.5:
                    cnt_above += 1
                    # a file to a_temp
                    with open(a_temp + file, 'a') as fwrite:
                        for line in a_i:
                            fwrite.writelines(line+"\n")
                    with open(b_temp + file, 'a') as fwrite:
                        for line in b_i:
                            fwrite.writelines(line+"\n")
                else:
                    cnt_below += 1
                    # a file exchange with b
                    with open(a_temp + file, 'a') as fwrite:
                        for line in b_i:
                            fwrite.writelines(line+"\n")
                    with open(b_temp + file, 'a') as fwrite:
                        for line in a_i:
                            fwrite.writelines(line+"\n")

    return a_temp, b_temp



def shuffle_files():
    a_dir = "/Users/kurt/Desktop/sbm/model/evaluation/sbm_tees_gold_cg13_dev_08_27_23_01/pred/"
    b_dir = "/Users/kurt/Desktop/dataset/tees/devrelationpreds/CG13-devel/unmerging/"

    a_out = os.listdir(a_dir)

    a_temp = a_dir+"a_temp/"
    b_temp = b_dir+"b_temp/"
    if not os.path.exists(a_temp):
        os.makedirs(a_temp)
    if not os.path.exists(b_temp):
        os.makedirs(b_temp)

    #swap PMID predictions at p=0.5
    cnt_above = 0
    cnt_below = 0
    for file in a_out:
        if file.endswith(".a2"):
            # print(file)
            num = random.random()
            a_file = a_dir+file
            b_file = b_dir+file
            with open(a_file, 'r') as fread:
                a_lines = fread.readlines()
            with open(b_file, 'r') as fread:
                b_lines = fread.readlines()
            if num > 0.5:
                cnt_above += 1
                #a file to a_temp
                with open(a_temp + file, 'w') as fwrite:
                    fwrite.writelines(a_lines)
                with open(b_temp + file, 'w') as fwrite:
                    fwrite.writelines(b_lines)
            else:
                cnt_below += 1
                #a file exchange with b
                with open(a_temp + file, 'w') as fwrite:
                    fwrite.writelines(b_lines)
                with open(b_temp + file, 'w') as fwrite:
                    fwrite.writelines(a_lines)

    return a_temp, b_temp





R = 10000
r_times = 0
orig_diff = 2.25
for r in range(R):
    a_temp, b_temp = shuffle_sentences()
    diff = compute_diff(a_temp, b_temp)
    # print("\t"+str(diff))
    if diff >= orig_diff:
        r_times += 1
    ratio = (r_times+1) / (R+1)
    if r % 10 == 0:
        print(str(r)+":"+str(ratio))






