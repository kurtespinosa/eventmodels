#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 17/05/2019 18:20
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""

from collections import OrderedDict

import src.util.constants as const
from src.util.util import DefaultOrderedDict
import chainer

def get_max_timesteps(level_sens):
    max_ts = 0
    for s in level_sens:
        for trig, trigstructures in s.items():
            ts = len(trigstructures[0][0]) # num edges
            if ts > max_ts:
                max_ts = ts
    return max_ts

def get_bylevel(batch, max_level):
    levels = [i[const.PRED_CAND_STRUCTURES_IDX] for i in batch]
    bylevel = []
    for i in range(max_level):
        level = []
        for s in levels:
            if i < len(s):
                level.append(s[i])
            else:
                level.append([])
        bylevel.append(level)
    return bylevel

def get_buffers(level):
    buffers = []
    for sen in level:
        trigbuffer = OrderedDict()
        for trig, trigstructures in sen.items():
            # true_edges = trigstructures[0][0]
            edges = trigstructures[0][1]
            buffer = []
            for e in range(len(edges)):
                # buffer.append((edges[e], const.ACTION_NONE, true_edges[e]))
                if edges[e] is 'NONE':
                    edge = (const.ID_NONE_ROLE_TYPE, const.ID_NONE_ENTITY_TYPE) # insert the role as role id
                else:
                    edge = edges[e]
                buffer.append((edge, const.ACTION_NONE))
            trigbuffer[trig] = buffer
        buffers.append(trigbuffer)
    return buffers

def buffer_arg_generator(batch):
    # get max levels
    max_level = max([len(i[const.PRED_CAND_STRUCTURES_IDX]) for i in batch])

    # arrange batch by levels
    bylevel = get_bylevel(batch, max_level)

    # get time step per level
    for l in bylevel:
        max_ts = get_max_timesteps(l)
        buffers = get_buffers(l)
        for t in range(max_ts):
            buffer_ts = timestep_buffer_generator(buffers, t)
            arg_ts = timestep_arg_generator(buffers, t)
            yield buffer_ts, arg_ts

            # scores = score_newstructures(newstructures)
            # predictions = convert_scores_to_preds(scores, threshold)
            # if chainer.config.train:
            #     loss = compute_loss(scores, gold)
            # nbest = select_nbest(scores, nbest)
            # pqs = nbest
            # structures = extract_nbeststructures(nbest)


def apply_actions(structures, buffer_ts, arg_ts):
    new_structures = []

    if len(structures) == 0:
        for s in range(len(arg_ts)):
            trig_sen = DefaultOrderedDict(list)
            for trig, arg in arg_ts[s].items():
                buffer = buffer_ts[s][trig]
                arg_id, _ = arg
                for a in const.ACTION_LIST:
                    structure = []
                    structure.append((arg_id, a))
                    structure.extend(buffer)
                    trig_sen[trig].append(structure)
            new_structures.append(trig_sen)
    else:
        for s in structures:
            trig_sen = DefaultOrderedDict()
            for trig, arg in arg_ts[s].items():
                buffer = buffer_ts[s][trig]
                arg_id, _ = arg
                for a in const.ACTION_LIST:
                    structure = []
                    structure.append((arg_id, a))
                    structure.extend(buffer)
                trig_sen[trig].append(structure)
            new_structures.append(trig_sen)
    return new_structures


def timestep_buffer_generator(buffers, t):
    sen_ts = []
    for sen in buffers:
        ts_trig = OrderedDict()
        for trig, buffer in sen.items():
            if t < len(buffer):
                arg = buffer[t+1:]
            else:
                arg = []
            ts_trig[trig] = arg
        sen_ts.append(ts_trig)
    return sen_ts

def timestep_arg_generator(buffers, t):
    sen_ts = []
    for sen in buffers:
        ts_trig = OrderedDict()
        for trig, buffer in sen.items():
            if t < len(buffer):
                arg = buffer[t]
            else:
                arg = []
            ts_trig[trig] = arg
        sen_ts.append(ts_trig)
    return sen_ts







                # #  extract gold label sequence for all events in this timestep
                # if chainer.config.train:
                #     num_events = len(trigstructures[0][2])
                #     gold_actions = collections.OrderedDict()
                #     for l in range(num_events):
                #         label = trigstructures[0][2][l][i][1]
                #         label_action = label.index(1)
                #         if l in gold_actions:
                #             prev_gold_action = gold_actions[l]
                #             new_gold_action = ''.join(prev_gold_action) + str(label_action)
                #             gold_actions[l] = new_gold_action
                #         else:
                #             gold_actions[l] = str(label_action)
                #
                # #  add actions for each structure
                # for s in structure:
                #     for a in range(len(const.ACTION_LIST)):
                #
                #         #  extract previous action from s
                #         new_action = str(prev_action) + str(a)
                #
                #         # compute target label
                #         target_label = 0  # if gold sequence of actions and current sequence of actions are the same, then is this the target path for this event
                #         if chainer.config.train:
                #             for g, gold_action in gold_actions.items():
                #                 str_action = ''.join(gold_action)
                #                 if str_action == new_action:
                #                     target_label = 1
                #                     break
                #         new_structure = s + [(entry, a), target_label]