import os
import collections

# def _extract_events(a1_lines, a2_lines):
def _extract_events(a2_lines):

    events = dict()
    for line in a2_lines:
        if line.startswith("E"):
            event_type = line.split()[1].split(":")[0]
            # event_multiset = _convert_to_multiset(a1_dict, a2_dict, line)
            # events[event_type].append(event_multiset)
            if event_type in events:
                cnt = events[event_type]
                events[event_type] = cnt + 1
            else:
                events[event_type] = 1
            # events[event_type] = line
    return events



# a1_pred_dir = "/Users/kurt/GoogleDrive/02N/04PROFDEV/PHD/phd/02process/experiments/deepevent/event-structure-detection/pycharm/ESC/input/emo/dev/"

EM_dir = "/Users/kurt/GoogleDrive/02N/04PROFDEV/PHD/phd/02process/experiments/deepevent/event-structure-detection/dataset/CG/dev/"

pred_dir = "/Users/kurt/GoogleDrive/02N/04PROFDEV/PHD/phd/02process/experiments/deepevent/event-structure-detection/codeversions/event-util-system/model/evaluation/scenario0_T.5/pred/"

gold_dir = "/Users/kurt/GoogleDrive/02N/04PROFDEV/PHD/phd/02process/experiments/deepevent/event-structure-detection/codeversions/event-util-system/dataset/input/dev/"


less = dict()
more = dict()

pred = os.listdir(pred_dir)
for f in pred:
    print(f)
    # em_file = open(EM_dir + f.split(".")[0] + ".split.a2t1", 'r')
    pred_file = open( pred_dir + f, 'r')
    gold_file = open(gold_dir + f, 'r')


    # em_lines = em_file.readlines()
    pred_lines = pred_file.readlines()
    gold_lines = gold_file.readlines()

    # em_events = _extract_events(em_lines)
    pred_events = _extract_events(pred_lines)
    gold_events = _extract_events(gold_lines)

    type = 'Positive_regulation'

    # if type in gold_events and type in em_events and type in pred_events:
    if type in gold_events and type in pred_events:
        less[f] = (gold_events, pred_events)

print("finished")