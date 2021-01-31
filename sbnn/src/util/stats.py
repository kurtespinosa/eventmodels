#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 08/05/2019 17:21
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""
from collections import Counter, OrderedDict
import logging
log = logging.getLogger(__name__)

from src.util import constants as const
from src.util.util import DefaultOrderedDict
from src.util import util

import matplotlib
matplotlib.use('agg') # prevents showing the plot
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def sort_per_file_and_sen_id(data, is_gold):
    '''
    Sorts data per file and per sentence id.
    :param data: sentence-based event information
    :param is_gold: condition to determine the constant index to use to refer to the indices
    :return: a dictionary of files of dictionary sentences and their event info
    '''
    d = DefaultOrderedDict(DefaultOrderedDict)
    for i in data:
        if is_gold:
            file_id = i[const.GOLD_SENTENCE_INFO_IDX][const.GOLD_SENTENCE_FILE_ID_IDX]
            sen_id = i[const.GOLD_SENTENCE_INFO_IDX][const.GOLD_SENTENCE_POS_IDX]
        else:
            file_id = i[const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_FILE_ID_IDX]
            sen_id = i[const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_POS_IDX]
        d[file_id][sen_id] = i
    return d

def retrieve_levels_events(train_srtd, test_gold_srtd, test_srtd, file, id):
    '''
    Check existence of file and sentence in gold and pred
    :param train_srtd: contains the gold levels
    :param test_gold_srtd: contains predicted events
    :param test_srtd: contains predicted levels
    :param file:
    :param id:
    :return:
    '''

    if file in train_srtd:
        if id in train_srtd[file]:
            gold_levels = train_srtd[file][id][const.PRED_CAND_STRUCTURES_IDX]
        else:
            gold_levels = None
    else:
        gold_levels = None
    if file in test_gold_srtd:
        if id in test_gold_srtd[file]:
            pred_events = test_gold_srtd[file][id][const.GOLD_EVENTS_IDX]
        else:
            pred_events = None
    else:
        pred_events = None
    if file in test_srtd:
        if id in test_srtd[file]:
            pred_levels = test_srtd[file][id][const.PRED_CAND_STRUCTURES_IDX]
        else:
            pred_levels = None
    else:
        pred_levels = None

    return gold_levels, pred_events, pred_levels


def level_recall_analysis(train_gold, train, test_gold, test):
    '''
    Return a count of gold per level and the predictions as well.
    :param train_gold: contains the gold events
    :param train: contains the gold events sorted in levels
    :param test_gold: contains the predicted events
    :param test: contains the predicted events sorted in levels
    :return: count of gold and predicted per type and per level
    '''


    # sort per file id and sentence id, important step as using instance index is not correct
    train_gold_srtd = sort_per_file_and_sen_id(train_gold, is_gold=True)
    train_srtd = sort_per_file_and_sen_id(train, is_gold=False)
    test_gold_srtd = sort_per_file_and_sen_id(test_gold, is_gold=True)
    test_srtd = sort_per_file_and_sen_id(test, is_gold=False)

    gold_per_level = OrderedDict()
    predicted_per_level = OrderedDict()

    # loop through using the gold since it is the reference
    for file, sen_ids in train_gold_srtd.items():
        for id, info in sen_ids.items():
            gold_events = info[const.GOLD_EVENTS_IDX]
            # retrieve corresponding events and levels
            gold_levels, pred_events, pred_levels = retrieve_levels_events(train_srtd, test_gold_srtd, test_srtd, file, id)

            # check if all have values, otherwise continue to next sentence id
            if gold_levels is not None and pred_events is not None and pred_levels is not None:
                # gold: create a map betweeen a trigger and its associated events
                gold_trig_event_map = util.create_trig_event_map(gold_events)
                pred_trig_event_map = util.create_trig_event_map(pred_events)

                # loop thru triggers in the gold levels and compare to the levels in pred if events are predicted
                for g in range(len(gold_levels)):
                    for trig, _ in gold_levels[g].items():
                        g_events = gold_trig_event_map[trig]
                        p_events = pred_trig_event_map[trig]

                        # stores all the indices of the pred events
                        pred_idx = [k for k in range(len(p_events))]

                        # check if each gold event is in pred events
                        for e in g_events:
                            c_gold = util.replace_event_arg_with_trigger(e, gold_events)
                            c_gold = Counter(c_gold)
                            # add level counter
                            if g in gold_per_level:
                                gold_per_level[g] += 1
                            else:
                                gold_per_level[g] = 1

                            if pred_idx:
                                ctr = 0 # ctr over num of pred events
                                while ctr < len(pred_idx):
                                    idx = pred_idx[ctr]
                                    c_pred = util.replace_event_arg_with_trigger(p_events[idx], pred_events)
                                    c_pred = Counter(c_pred)

                                    # if gold event and pred event are the same, then add level counter
                                    if c_gold == c_pred:
                                        if g in predicted_per_level:
                                            predicted_per_level[g] += 1
                                        else:
                                            predicted_per_level[g] = 1
                                        # remove pred event once matched
                                        pred_idx.remove(idx)
                                        break
                                    else:
                                        ctr += 1
    return gold_per_level, predicted_per_level

def generate_heatmaps(train_ids, dev_ids, use_filter, train_gold, dev_gold, output_dir):
    '''
    Creates a heatmap on the number of occurrences of an event type of a particular argument count (or length)
    This is useful to see the distribution of lengths of event structures per event type.
    :param train_ids:
    :param dev_ids:
    :param use_filter:
    :param train_gold:
    :param dev_gold:
    :param output_dir:
    :return:
    '''
    # predictions
    max_train_pred_ids, train_pred_type_ctr = get_max_rel_and_dict_in_pred(train_ids, use_filter)
    max_dev_pred_ids, dev_pred_type_ctr = get_max_rel_and_dict_in_pred(dev_ids, use_filter)

    train_pred_ar, train_pred_types = convert_dict_to_2d_array_and_extract_ids(train_pred_type_ctr, max_train_pred_ids)
    dev_pred_ar, dev_pred_types = convert_dict_to_2d_array_and_extract_ids(dev_pred_type_ctr, max_dev_pred_ids)
    draw_heat_map(train_pred_ar, train_pred_types, max_train_pred_ids, output_dir,
                        'train_pred.pdf', 'Event type', 'Argument count',
                        'Number of occurrences of an event type of an argument count')
    draw_heat_map(dev_pred_ar, dev_pred_types, max_dev_pred_ids, output_dir,
                        'dev_pred.pdf', 'Event type', 'Argument count',
                        'Number of occurrences of an event type of an argument count')
    # gold
    max_train_gold, train_gold_type_ctr = get_max_rel_and_dict_in_gold(train_gold)
    max_dev_gold, dev_gold_type_ctr = get_max_rel_and_dict_in_gold(dev_gold)

    train_gold_ar, train_gold_types = convert_dict_to_2d_array_and_extract_ids(train_gold_type_ctr, max_train_gold)
    dev_gold_ar, dev_gold_types = convert_dict_to_2d_array_and_extract_ids(dev_gold_type_ctr, max_dev_gold)
    draw_heat_map(train_gold_ar, train_gold_types, max_train_gold, output_dir,
                        'train_gold.pdf', 'Event type', 'Argument count',
                        'Number of occurrences of an event type of an argument count')
    draw_heat_map(dev_gold_ar, dev_gold_types, max_dev_gold, output_dir,
                        'dev_gold.pdf', 'Event type', 'Argument count',
                        'Number of occurrences of an event type of an argument count')

def get_max_rel_and_dict_in_gold(instances):
    '''
    Return the max number of relations of any event type and the dictionary of counts.

    :param instances: the instances with the event structures
    :return: the max num of relations and the dictionary of counts
    '''
    max = 0

    type_ctr = DefaultOrderedDict(DefaultOrderedDict)

    for i in instances:
        events = i[const.GOLD_EVENTS_IDX]
        for id, st in events.items():
            # store length of type with ctr
            type = st[0].split(":")[0]
            cnt = len(st)-1
            if cnt in type_ctr[type]:
                temp = type_ctr[type][cnt]
                type_ctr[type][cnt] = temp + 1
            else:
                type_ctr[type][cnt] = 1

            # determine max
            if cnt > max:
                max = cnt

    max += 1 # to count NONE as one argument
    return max, type_ctr



def get_max_rel_and_dict_in_pred(instances, USE_FILTER):
    '''
    Return the maximum number relations of any event structure and the dictionary of counts

    The relations include NONE as one argument to represent no argument events.
    :param instances: contains the events from all instances
    :param USE_FILTER: condition to determine the index of the events
    :return: the maximum num of relations and dictionary of counts
    '''
    max = 0

    type_ctr = DefaultOrderedDict(DefaultOrderedDict)

    ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX

    for i in instances:
        events = i[ind]
        for level in events:
            for id, info in level.items():
                for l in info:
                    cnt = len(l[0])-1

                    # store length of type with ctr
                    type = i[const.IDS_TRIGGERS_MERGED_IDX][id][0]
                    if cnt in type_ctr[type]:
                        temp = type_ctr[type][cnt]
                        type_ctr[type][cnt] = temp + 1
                    else:
                        type_ctr[type][cnt] = 1

                    # determine max
                    if cnt > max:
                        max = cnt
    max += 1  # to count NONE as one argument
    return max, type_ctr

def convert_dict_to_2d_array_and_extract_ids(d, max):
    '''
    Convert a dictionary of dictionary into a list of list.

    If the element in dictionary b is not present, its value is set to zero.
    :param d: the dictionary a containing dictionary b of counts
    :param max: the maximum number in dictionary b
    :return: the 2d array and dictionary ids
    '''
    lines = []
    types = []
    for type, ctrs in sorted(d.items()):
        line = []
        types.append(type)
        for i in range(max):
            if i in ctrs:
                line.append(ctrs[i])
            else:
                line.append(0)
        lines.append(line)
    return lines, types


def draw_heat_map(ar, ys, xlen, output_dir, name, ylabel, xlabel, title):
    '''
    Draw a heatmap and save it.

    :param ar: the 2d array
    :param ys: the y labels
    :param xlen: the x labels (integer)
    :param output_dir: output directory
    :param name: filename
    :param ylabel: y axis label
    :param xlabel: x axis label
    :param title: title
    :return: None
    '''

    # construct panda confusion matrix
    cm = pd.DataFrame(ar, index = [i for i in ys], columns=[i for i in range(xlen)])

    # plot
    plt.ioff()
    plt.figure(figsize=(15, 10))
    sn.set(font_scale=1)  # for label size
    sn.heatmap(cm, annot=True, fmt="d")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.savefig(output_dir+name, bbox_inches='tight')


def get_max_nbest(train_ids, USE_FILTER):
    '''
    Returns the maximum number of nbest size needed so that all events are predicted
    when threshold is zero (done for sanity checking). This is computed by
    counting the N, which is the maximum number of arguments in a particular trigger.
    nbest = (2^n)/2 + (2^n)/4.
    :param train_ids:
    :return:
    '''
    ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
    maxlen = 0
    for i in train_ids:
        events = i[ind]
        for level in events:
            for trig, structures in level.items():
                for s in structures[0][2]:
                    l = len(s)
                    if l > maxlen:
                        maxlen = l
    n_best = ((2 ** maxlen) / 2) + ((2 ** maxlen) / 4)
    return n_best

def arg_cnt_event_ctr(train_ids, USE_FILTER):
    '''
    Returns the number of events with a particular number of arguments sorted ascending by key (number of args).
    This is useful to see the distribution of the length of events in terms of number
    of arguments regardless of the event type. For detailed version, see generate_heatmaps function above.
    :param train_ids:
    :return:
    '''
    d = OrderedDict()
    ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
    for i in train_ids:
        events = i[ind]
        for level in events:
            for trig, structures in level.items():
                for s in structures[0][2]:
                    l = len(s)
                    if l in d:
                        d[l] += 1
                    else:
                        d[l] = 1
    #sort
    d = sorted(d.items(), key=lambda key_value: key_value[0])
    return d

# def types_and_nested_level_count(train_ids):

