#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 07/06/2019 11:35
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""

import os

os.environ["CHAINER_SEED"] = "0"
import random

random.seed(0)
import numpy as np

np.random.seed(0)


# standard
import time
import logging
import sys
import cProfile as profile
import gc
import itertools
import collections
import heapq
import copy
import math
from datetime import datetime

log = logging.getLogger(__name__)

# 3rd party
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

# modules
import src.util.constants as const
from src.util.util import DefaultOrderedDict
from src.util.batchprocessor import *


# class Loader(chainer.Chain):
#     def __init__(self):
#         pass
#
#     def load_glove(self, path, vocab):
#         print("Loading embeddings...", end='')
#         with open(path, "r") as fi:
#             for line in fi:
#                 line_list = line.strip().split(" ")
#                 word = line_list[0]
#                 if word in vocab:
#                     vec = self.xp.array(line_list[1::], dtype=np.float32)
#                     self.embed_wordtype.W.data[vocab[word]] = vec
#         print("Finished.")


class SBM2(chainer.Chain):
    def __init__(self, n_word_types, n_trig_types, n_role_types, n_entity_types, params, config=None):
        super(SBM2, self).__init__()
        with self.init_scope():
            self.DIM_EMBED = params['dim_embed']
            self.DIM_BILSTM = params['dim_bilstm']
            self.DIM_TRIG_TYPE = params['dim_arg_type']
            self.DIM_ROLE_TYPE = params['dim_role_type']
            self.DIM_ARG_TYPE = params['dim_arg_type']
            self.DROPOUT = params['dropout']
            self.THRESHOLD = params['threshold']
            self.N_BEST = params['n_best']
            self.MARGIN = params['margin']
            self.action_dim = params['dim_action']

            self.trigword_dim = self.DIM_BILSTM
            self.trig_dim = self.DIM_TRIG_TYPE + self.trigword_dim

            self.argword_dim = self.DIM_BILSTM
            self.arg_dim = self.DIM_ARG_TYPE + (self.DIM_BILSTM * 2)

            self.len_relation = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2) + self.DIM_ROLE_TYPE + self.DIM_ARG_TYPE + (self.DIM_BILSTM * 2)
            self.hidden_dim = int(self.arg_dim / 2)

            self.len_type_and_arg = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2)

            self.bilstm = L.NStepBiLSTM(1, self.DIM_EMBED, self.DIM_BILSTM, 0)
            self.embed_wordtype = L.EmbedID(n_word_types, self.DIM_EMBED, ignore_label=-1)
            self.embed_trigtype = L.EmbedID(n_trig_types, self.DIM_TRIG_TYPE, ignore_label=-1)
            self.embed_roletype = L.EmbedID(n_role_types, self.DIM_ROLE_TYPE, ignore_label=-1)
            self.embed_enttype = L.EmbedID(n_entity_types, self.DIM_ARG_TYPE, ignore_label=-1)
            self.embed_action = L.EmbedID(self.action_dim, self.action_dim, ignore_label=-1)

            self.linear_structure = L.Linear(None, self.len_relation + self.action_dim)
            self.linear_buffer = L.Linear(None, self.len_relation + self.action_dim)

            self.state_representation = L.Linear(None, self.arg_dim)
            self.linear1 = L.Linear(None, self.hidden_dim)
            self.linear2 = L.Linear(None, self.hidden_dim)
            self.linear = L.Linear(None, 1)

            if config is not None:
                self.update_params(config, n_word_types, n_trig_types, n_role_types, n_entity_types)

    def load_pret_embed(self, path, vocab):
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=self.xp.float32)
                    self.embed_wordtype.W.data[vocab[word]] = vec
        log.info("Loaded embeddings: %s", path)

    def update_params(self, config, n_word_types, n_trig_types, n_role_types, n_entity_types):
        # self.DIM_BILSTM = config['DIM_BILSTM']
        # self.DIM_ARG_TYPE = config['DIM_ARG_TYPE']
        # self.DIM_ROLE_TYPE = config['DIM_ROLE_TYPE']
        # self.action_dim = config['DIM_ACTION']
        # self.N_BEST = config['N_BEST']
        # self.MARGIN = config['MARGIN']
        self.DROPOUT = config['DROPOUT']

        # self.trigword_dim = self.DIM_BILSTM
        # self.DIM_TRIG_TYPE = self.DIM_ARG_TYPE
        # self.trig_dim = self.DIM_TRIG_TYPE + self.trigword_dim

        # self.argword_dim = self.DIM_BILSTM
        # self.arg_dim = self.DIM_ARG_TYPE + (self.DIM_BILSTM * 2)

        # self.len_relation = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2) + self.DIM_ROLE_TYPE + self.DIM_ARG_TYPE + (self.DIM_BILSTM * 2)
        # self.hidden_dim = int(self.arg_dim / 2)

        # self.len_type_and_arg = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2)

        # self.bilstm = L.NStepBiLSTM(1, self.DIM_EMBED, self.DIM_BILSTM, 0)
        # self.embed_wordtype = L.EmbedID(n_word_types, self.DIM_EMBED, ignore_label=-1)
        # self.embed_trigtype = L.EmbedID(n_trig_types, self.DIM_TRIG_TYPE, ignore_label=-1)
        # self.embed_roletype = L.EmbedID(n_role_types, self.DIM_ROLE_TYPE, ignore_label=-1)
        # self.embed_enttype = L.EmbedID(n_entity_types, self.DIM_ARG_TYPE, ignore_label=-1)
        # self.embed_action = L.EmbedID(self.action_dim, self.action_dim, ignore_label=-1)

        # self.linear_structure = L.Linear(None, self.len_relation + self.action_dim)
        # self.linear_buffer = L.Linear(None, self.len_relation + self.action_dim)

        # self.state_representation = L.Linear(None, self.arg_dim)
        # self.linear1 = L.Linear(None, self.hidden_dim)
        # self.linear2 = L.Linear(None, self.hidden_dim)
        # self.linear = L.Linear(None, 1)

    def _bilstm_layer(self, batch):
        # TODO: Implement batch mechanism
        xs = []
        for i in batch:
            xs.append(self.xp.array(i[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]).astype("i"))

        # TODO: concat all xs into one big list and store lengths
        embed_xs = [F.dropout(self.embed_wordtype(item), ratio=self.DROPOUT) for item in xs]

        # TODO: separate embed_xs back into list-of-list
        hy, cy, bilstm_xs = self.bilstm(None, None, embed_xs)

        # TODO: into batch again
        return bilstm_xs

    def _construct_structure_embeddings(self, instance, bilstm_i, trigger, structure, entities_ids, triggers_ids, structures_above_threshold=None):
        def _represent_mentions(mention_ids, bilstm_i):
            try:
                id = mention_ids[0]
                bi = bilstm_i[id]
            except:
                bi = self.xp.zeros((self.DIM_BILSTM * 2), dtype=self.xp.float32)
            mention_array = self.xp.array([bi.data]).astype("f")
            for i in range(len(mention_ids) - 1):
                id = mention_ids[i + 1]
                bi = bilstm_i[id]
                temp = self.xp.array([bi.data]).astype("f")
                mention_array = self.xp.concatenate((mention_array, temp))
            final_mention_representation = F.average(mention_array, axis=0)
            return final_mention_representation

        def _get_word_ids(xsi, mention):
            word_ind = []
            sentence_ids = xsi[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]
            for i in mention:
                if i in sentence_ids:
                    ind = sentence_ids.index(i)
                    word_ind.append(ind)
            return word_ind

        def _represent_type_and_argument(arg_ids, type_index, bilstm_i, type_label, structures_above_threshold=None):

            embedding_list = []
            if structures_above_threshold is not None:
                embedding_list = structures_above_threshold[type_label]
            else:
                defn = arg_ids[type_label]
                type_id = defn[const.IDS_ARG_TYPE]
                type_embedding = None
                if type_index == const.IDS_TRIGGERS_IDX:
                    type_embedding = self.embed_trigtype(self.xp.array([type_id]).astype("i"))
                elif type_index == const.IDS_ENTITIES_IDX:
                    type_embedding = self.embed_enttype(self.xp.array([type_id]).astype("i"))

                mention = defn[const.IDS_ARG_MENTION]
                mention_ids = _get_word_ids(instance, mention)
                mention_embedding = _represent_mentions(mention_ids, bilstm_i)

                flattened_type_embedding = F.flatten(type_embedding)
                flattened_mention_embedding = F.flatten(mention_embedding)
                type_and_argument_embedding = F.hstack([flattened_type_embedding, flattened_mention_embedding])

                reshaped_type_and_argument_embedding = F.reshape(type_and_argument_embedding, (1, self.len_type_and_arg))
                embedding_list.append(reshaped_type_and_argument_embedding)
            return embedding_list

        def _construct_relation_embedding(trig_embedding, pair, entities_ids, triggers_ids, bilstm_i, structures_above_threshold=None):
            relation = pair[0]
            action = pair[1]
            if action == const.ACTION_NONE:
                action_embedding = Variable(self.xp.zeros((self.action_dim), dtype=self.xp.float32))
            else:
                action_embedding = self.embed_action(self.xp.array([action]).astype("i"))

            if relation[1] == const.NONE_ROLE_TYPE:
                role = 0
                type_id = 0
                role_type_embedding = None
                try:
                    role_type_embedding = self.embed_roletype(self.xp.array([role]).astype("i"))
                except:
                    print("debug")
                type_embedding = self.embed_enttype(self.xp.array([type_id]).astype("i"))

                mention = [0]  # TODO: how to represent this better?
                mention_ids = _get_word_ids(instance, mention)
                mention_embedding = _represent_mentions(mention_ids, bilstm_i)

                flattened_type_embedding = F.flatten(type_embedding)
                flattened_mention_embedding = F.flatten(mention_embedding)
                type_and_argument_embedding = F.hstack([flattened_type_embedding, flattened_mention_embedding])

                arg_embedding = F.reshape(type_and_argument_embedding, (1, self.len_type_and_arg))
                relation_embedding = []
                a = F.flatten(trig_embedding)
                b = F.flatten(role_type_embedding)
                c = F.flatten(arg_embedding)
                d = F.flatten(action_embedding)
                z = F.hstack([a, b, c, d])
                emb = F.reshape(z, (1, self.len_relation + self.action_dim))
                relation_embedding.append(emb)
            else:
                role = relation[0]
                arg = relation[1]
                role_type_embedding = self.embed_roletype(self.xp.array([role]).astype("i"))
                is_trigger = arg in triggers_ids

                if is_trigger:
                    arg_embedding = _represent_type_and_argument(triggers_ids, const.IDS_TRIGGERS_IDX, bilstm_i, arg,
                                                                 structures_above_threshold)
                else:
                    arg_embedding = _represent_type_and_argument(entities_ids, const.IDS_ENTITIES_IDX, bilstm_i, arg)
                relation_embedding = []
                if len(arg_embedding) != 0:
                    for i in range(len(arg_embedding)):
                        a = F.flatten(trig_embedding)
                        b = F.flatten(role_type_embedding)
                        c = F.flatten(arg_embedding[i])
                        d = F.flatten(action_embedding)
                        z = F.hstack([a, b, c, d])
                        emb = F.reshape(z, (1, self.len_relation + self.action_dim))
                        relation_embedding.append(emb)
            return relation_embedding

        trig_embedding = _represent_type_and_argument(triggers_ids, const.IDS_TRIGGERS_IDX, bilstm_i, trigger)[0]
        structure_embedding = []
        for pair in structure:
            embedding = _construct_relation_embedding(trig_embedding, pair, entities_ids, triggers_ids, bilstm_i, structures_above_threshold)
            structure_embedding.append(embedding)
        return structure_embedding

    def perform_action(self, S, B):
        S_ = self.xp.zeros((self.len_relation + self.action_dim), dtype=self.xp.float32)
        final_b = self.xp.zeros((self.len_relation + self.action_dim), dtype=self.xp.float32)

        # get reduced B representation
        if len(B) > 0:
            stack = F.vstack(B[0])
            for i in range(len(B) - 1):
                stack = F.vstack((stack, F.vstack(B[i + 1])))
            linear_b = F.relu(self.linear_buffer(stack))
            sum_b = F.sum(linear_b, 0)
            final_b = F.reshape(sum_b, (1, self.len_relation + self.action_dim))

        # replicate B len(S) times
        B_ = F.tile(final_b, (len(S), 1))

        # compose len(s) S
        if len(S) > 0:
            S_ = F.reshape(F.sum(F.relu(self.linear_structure(F.vstack(S[0]))), 0), (1, self.len_relation + self.action_dim))
            for i in range(len(S) - 1):
                temp = F.reshape(F.sum(F.relu(self.linear_structure(F.vstack(S[i + 1]))), 0), (1, self.len_relation + self.action_dim))
                S_ = F.vstack((S_, temp))

        new_state = F.concat((S_, B_))
        state_rep = self.state_representation(new_state)
        relu = F.relu(state_rep)
        h1 = self.linear1(relu)
        relu2 = F.relu(h1)
        h2 = self.linear2(relu2)
        relu3 = F.relu(h2)
        raw_score = self.linear(relu3)
        sigmoid_score = F.sigmoid(raw_score)

        return raw_score, sigmoid_score, state_rep

    def check_if_all_subevents_present(self, cur_buffer, cur_structure, trigger_subevent_embeddings):
        present = True
        for rel in cur_buffer:
            entry = rel[0]
            if type(entry) is tuple:  # not None
                arg_id = entry[1]
                if arg_id.startswith("TR"):
                    if arg_id not in trigger_subevent_embeddings:
                        present = False
                        break

        for rel in cur_structure:
            entry = rel[0]
            if type(entry) is tuple:  # not None
                arg_id = entry[1]
                if arg_id.startswith("TR"):
                    if arg_id not in trigger_subevent_embeddings:
                        present = False
                        break
        return present

    def _predict(self, instance, bilstm_i, max_pq):
        def _entry_in_buffer(entry, cur_buffer):
            yes = False
            for c in cur_buffer:
                edge = c[0]
                if entry == edge:
                    yes = True
                    break
            return yes

        def _remove_entry(cur_buffer, entry):
            new_buffer = []
            for c in cur_buffer:
                edge = c[0]
                if entry != edge:
                    new_buffer.append(c)
            return new_buffer

        def _extract_sub_event_predictions_combinations(structure, predictions, combi):
            '''
            There's no need to generate combinations for the buffer as only the structure gets to be added in the final structures
            :param structure:
            :param predictions:
            :param combi:
            :return:
            '''
            combination = []

            last_relation = structure[-1]
            lst = []

            # add the previous structure
            if combi is None:  # first step in search
                lst.append((last_relation[0], 0))

            else:
                # print("test")
                for c in combi:
                    lst = []
                    lst.append(c)
                    # combination.append(lst)
                    combination.append(lst)

                lst = []
                # add the combination of the last relation
                arg = last_relation[0][1]
                action = last_relation[1]
                if action != const.ACTION_IGNORE:
                    if type(last_relation[0]) is tuple:
                        if arg.startswith("TR") and arg in predictions:
                            preds = predictions[arg]
                            for p in range(len(preds)):
                                lst.append((arg, p))  # pth index
                        else:
                            lst.append((arg, 0))  # the representation at index 0
                    else:  # might not be executed at all
                        lst.append((last_relation[0], 0))
                else:
                    lst.append((arg, 0))

            combination.append(lst)

            final_list_of_combination = []
            for comb in itertools.product(*combination):
                l = list(comb)
                final_list_of_combination.append(l)

            return final_list_of_combination

        def extract_combination(sub_event_prediction_combination, structure_emb):
            len_s = len(structure_emb)
            temp_structure_emb = []

            for i in range(len_s):
                try:
                    rel = sub_event_prediction_combination[i]
                except:
                    print("debug e_c")
                temp_structure_emb.append(structure_emb[i][rel[1]])  # 1 is the index

            # ind = len_s
            # for j in range(len_b):
            #     rel = sub_event_prediction_combination[ind+j]
            #     try:
            #         temp_buffer_emb.append(buffer_emb[j][rel[1]])
            #     except:
            #         print("debug")

            return temp_structure_emb

        triggers = []
        action_list = [const.ACTION_IGNORE, const.ACTION_ADD, const.ACTION_ADDFIX]
        triggers_structures_mapping = dict()

        # gather triggers in bottom-up sequence
        groups = instance[const.PRED_CAND_STRUCTURES_IDX]
        entities_ids = instance[const.PRED_ENTITIES_IDX]
        triggers_ids = instance[const.PRED_TRIGGERS_IDX]
        for g in range(len(groups)):
            for trig, trigstructures in groups[g].items():
                triggers.insert(0, (trig, g))
                triggers_structures_mapping[trig] = trigstructures

        # sentence level variables for sentence-level learning
        ctr = 0
        states_dict = {}
        trigger_subevent_embeddings = collections.defaultdict(list)
        final_event_structures = collections.defaultdict(list)
        instance_loss = 0.0

        # max_num_gold_actions = 0

        early_stop = False

        count = 0

        while triggers:
            trigger, level = triggers.pop()
            for s in range(len(triggers_structures_mapping[trigger])):
                # track n-best states
                pq = []

                edges = triggers_structures_mapping[trigger][s][1]
                true_edges = triggers_structures_mapping[trigger][s][0]

                # initial state
                structure = []
                buffer = []
                for e in range(len(edges)):
                    buffer.insert(0, (edges[e], const.ACTION_NONE, true_edges[e]))
                list_buffer = copy.deepcopy(buffer)
                new_buffer = copy.deepcopy(buffer)
                new_structure = copy.deepcopy(structure)
                states_dict[ctr] = (new_buffer[:], new_structure[:], None, '', -1, 0, 0, None)
                heapq.heappush(pq, (0.0, ctr))
                ctr += 1

                if chainer.config.train:
                    num_events = len(triggers_structures_mapping[trigger][s][2])
                    gold_actions = dict()

                for i in range(len(list_buffer)):
                    entry, _, true_entry = list_buffer.pop()

                    # check if entry is a sub-event
                    arg_id = None
                    if type(entry) is tuple:  # not None
                        arg_id = entry[1]
                        if arg_id.startswith("TR"):
                            if arg_id not in trigger_subevent_embeddings:
                                early_stop = True
                                break  # can't perform the search since the argument event has not been detected

                    # extract gold labels
                    if chainer.config.train:
                        for l in range(num_events):
                            label = triggers_structures_mapping[trigger][s][2][l][i][1]
                            label_action = label.index(1)
                            if l in gold_actions:
                                prev_gold_action = gold_actions[l]
                                new_gold_action = ''.join(prev_gold_action) + str(label_action)
                                gold_actions[l] = new_gold_action
                            else:
                                gold_actions[l] = str(label_action)

                                # check the max number of unique gold actions at this timestep
                                # all_unique = set()
                                # for k, v in gold_actions.items():
                                #     all_unique.add(v)
                                #
                                # max_num = len(all_unique)
                                # if max_num > max_num_gold_actions:
                                #     max_num_gold_actions = max_num

                                # if train_epoch in [1, 50, 100, 150, 200]:
                                #     print("GOLD actions:")
                                #     for k,v in gold_actions.items():
                                #         print(v)

                    new_pq = []

                    for pqi in range(len(pq)):
                        _, state_ctr = heapq.heappop(pq)
                        cur_buffer, cur_structure, _, prev_action, temp_target_label, _, _, combi = states_dict[state_ctr]

                        if not _entry_in_buffer(entry, cur_buffer):  # check if edge in buffer if not skip this state
                            continue

                        cur_buffer = _remove_entry(cur_buffer, entry)
                        new_buffer = copy.deepcopy(cur_buffer)

                        # if TRAIN:
                        #     print("get the event reps of the gold sub-events")
                        #
                        # else:
                        # TODO check relations only that are found in the gold druin
                        if not self.check_if_all_subevents_present(cur_buffer, cur_structure, trigger_subevent_embeddings):
                            early_stop = True
                            break

                        # TODO: process the relations once
                        buffer_emb = self._construct_structure_embeddings(instance, bilstm_i, trigger, new_buffer,
                                                                          entities_ids, triggers_ids,
                                                                          trigger_subevent_embeddings)

                        # process the structure_emb before action is added
                        # construct the relation embeddings
                        # TODO process relations once
                        new_structure = copy.deepcopy(cur_structure)
                        common_structure_emb = self._construct_structure_embeddings(instance, bilstm_i, trigger, new_structure,
                                                                                    entities_ids, triggers_ids,
                                                                                    trigger_subevent_embeddings)

                        for a in range(len(action_list)):
                            structure_emb = []
                            new_structure = []
                            if entry == const.NONE_ROLE_TYPE and a == const.ACTION_ADD:
                                continue  # skip this adding NONE does not make sense, only IGNORE or ADDFIX

                            new_structure.append((entry, action_list[a], true_entry))

                            # construct the relation embeddings
                            # TODO process all actions once
                            temp_emb = self._construct_structure_embeddings(instance, bilstm_i, trigger, new_structure, entities_ids, triggers_ids, trigger_subevent_embeddings)

                            # construct new action
                            new_action = str(prev_action) + str(a)

                            # get gold action
                            target_label = 0
                            if chainer.config.train:
                                for g, gold_action in gold_actions.items():
                                    str_action = ''.join(gold_action)
                                    if str_action == new_action:
                                        target_label = 1
                                        break

                            # construct the structure and the structure_emb
                            assert len(temp_emb) == 1, "ERROR: Cannot have more than one entry."
                            # if len(temp_emb) > 1:
                            #     print("debug")
                            structure_emb.extend(common_structure_emb)
                            structure_emb.append(temp_emb[0])

                            # create the structure
                            new_structure = copy.deepcopy(cur_structure)
                            new_structure.append((entry, action_list[a], true_entry))

                            # score the action and push resulting states to PQ
                            sub_event_predictions_combinations = _extract_sub_event_predictions_combinations(
                                new_structure, trigger_subevent_embeddings, combi)

                            temp_structure_embs = []
                            for sub_event_prediction_combination in sub_event_predictions_combinations:
                                temp_structure_emb = extract_combination(sub_event_prediction_combination, structure_emb)
                                temp_structure_embs.append(temp_structure_emb)

                            raw_scores, sigmoid_scores, state_reps = self.perform_action(temp_structure_embs, buffer_emb)
                            count += 1

                            for k in range(len(sub_event_predictions_combinations)):
                                # push into PQ resulting state with the score
                                if chainer.config.train:
                                    states_dict[ctr] = (new_buffer[:], new_structure[:], state_reps[k], new_action, target_label, raw_scores[k], sigmoid_scores[k], sub_event_predictions_combinations[k])
                                else:
                                    states_dict[ctr] = (new_buffer[:], new_structure[:], state_reps[k], new_action, -1, raw_scores[k], sigmoid_scores[k], sub_event_predictions_combinations[k])

                                new_score = 1.0 - sigmoid_scores[k].data[0]  # because min pq is used
                                heapq.heappush(new_pq, (new_score, ctr))
                                ctr += 1

                    # add margin to all non-gold actions
                    if chainer.config.train:
                        new_pq_with_margin = []
                        for k in range(len(new_pq)):
                            temp_score, temp_state_ctr = new_pq[k]
                            temp_buffer, temp_structure, temp_state_rep, temp_new_action, target_label, raw_score, sigmoid_score, combi = states_dict[temp_state_ctr]
                            if target_label == 0:  # non-gold
                                before = sigmoid_score
                                sigmoid_score = sigmoid_score + self.MARGIN
                                temp_score = temp_score - self.MARGIN
                                states_dict[temp_state_ctr] = (temp_buffer, temp_structure, temp_state_rep, temp_new_action, target_label, raw_score, sigmoid_score, combi)
                                new_pq_with_margin.append((temp_score, temp_state_ctr))
                            else:
                                new_pq_with_margin.append((temp_score, temp_state_ctr))
                        new_pq = []
                        for n in range(len(new_pq_with_margin)):
                            temp_score, temp_state_ctr = new_pq_with_margin[n]
                            heapq.heappush(new_pq, (temp_score, temp_state_ctr))

                    # create new pq copy
                    new_pq_copy = []
                    for k in range(len(new_pq)):
                        new_pq_copy.append(new_pq[k])
                    # set the PQ size to N
                    nbest_pq = []
                    count = 0
                    if len(new_pq) > 0:
                        for _ in range(self.N_BEST):
                            temp_score, temp_state_ctr = heapq.heappop(new_pq)
                            heapq.heappush(nbest_pq, (temp_score, temp_state_ctr))
                            count += 1
                            if count == self.N_BEST:  # number of items in heap is greater than or equal to N
                                break
                            if not new_pq:  # pq is already empty, number of items in heap is less than N
                                break

                    # DEBUG: print n-best
                    # for n in range(len(nbest_pq)):
                    #     _, temp_state_ctr = nbest_pq[n]
                    #     _, temp_structure, temp_state_rep, temp_action, _, raw_score, sigmoid_score, _ = \
                    #     states_dict[temp_state_ctr]

                    # fix the event if it is greater than threshold
                    to_be_removed = []
                    for n in range(len(nbest_pq)):
                        _, temp_state_ctr = nbest_pq[n]
                        _, temp_structure, temp_state_rep, temp_action, _, raw_score, sigmoid_score, combi = states_dict[temp_state_ctr]

                        if temp_action.endswith(str(const.ACTION_ADDFIX)):  # found an event, event before the last edge
                            to_be_removed.append(temp_state_ctr)  # always remove addfix from beam
                            if sigmoid_score.data[0] >= self.THRESHOLD:
                                trigger_subevent_embeddings[trigger].append(temp_state_rep)
                                final_event_structures[trigger].append([temp_structure, raw_score])

                    if chainer.config.train:  # early update
                        # check 1: if gold is out of beam
                        to_be_updated = dict()
                        for n in range(len(new_pq_copy)):
                            _, temp_state_ctr = new_pq_copy[n]
                            _, _, _, temp_action, temp_target_label, raw_score, _, _ = states_dict[temp_state_ctr]
                            gold_predicted = False
                            if temp_target_label == 1:  # it is gold
                                for q in range(len(nbest_pq)):
                                    _, temp_state_ctr_2 = nbest_pq[q]
                                    if temp_state_ctr == temp_state_ctr_2:  # gold was predicted
                                        gold_predicted = True
                                        break
                                if not gold_predicted:
                                    early_stop = True
                                    to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)

                        # check 2:
                        for n in range(len(nbest_pq)):
                            _, temp_state_ctr = nbest_pq[n]
                            _, _, _, temp_action, temp_target_label, raw_score, sigmoid_score, _ = states_dict[temp_state_ctr]
                            # predicted action is not gold,
                            if temp_target_label == 0:  # not gold
                                to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)
                            # predicted, gold but less than threshold
                            if temp_action.endswith(str(const.ACTION_ADDFIX)) and temp_target_label == 1:
                                if sigmoid_score.data[0] < self.THRESHOLD:
                                    to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)
                                else:
                                    to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)

                        # DEBUG: print what gets updated
                        # for key, value in to_be_updated.items():
                        #     _, _, _, temp_action, temp_target_label, orig_score, sigmoid_score, _ = states_dict[key]
                        #     value_score = value[0]
                        #     value_label = value[1]
                        #     value_action = value[2]
                        #     assert value_action == temp_action, "Error: action not equal"

                        # compute the loss and update
                        for key, value in to_be_updated.items():
                            _, _, _, _, temp_target_label, orig_score, _, _ = states_dict[key]
                            value_score = value[0]
                            value_label = value[1]

                            # assert value_score.data[0][0] == orig_score.data[0][0], "Error: values not equal"
                            assert value_label == temp_target_label, "Error: values not equal"
                            new_target = self.xp.array([[temp_target_label]], 'i')
                            loss = F.sigmoid_cross_entropy(F.reshape(orig_score, (1, 1)), new_target)  # use un-normalised probability
                            instance_loss += loss

                    # remove fixed events from n-best
                    new_n_best_pq = []
                    for n in range(len(nbest_pq)):
                        temp_score, temp_state_ctr = nbest_pq[n]
                        if temp_state_ctr not in to_be_removed:
                            new_n_best_pq.append((temp_score, temp_state_ctr))
                    temp_pq = []
                    for n in range(len(new_n_best_pq)):
                        temp_score, temp_state_ctr = new_n_best_pq[n]
                        heapq.heappush(temp_pq, (temp_score, temp_state_ctr))

                    pq = temp_pq[:]

                    if early_stop:
                        break

                if early_stop:
                    break

            if early_stop:
                break

        predictions = final_event_structures
        return predictions, instance_loss, count

    def __call__(self, batch):
        batch_predictions = []

        bilstm_batch = self._bilstm_layer(batch)

        batch_loss = 0

        max_pq = 0
        batch_cnt = 0

        for i in range(len(batch)):
            instance = batch[i]
            if chainer.config.train:
                predictions, instance_loss, _ = self._predict(instance, bilstm_batch[i], max_pq)
                batch_loss += instance_loss
            else:
                predictions, _, count = self._predict(instance, bilstm_batch[i], max_pq)
                batch_cnt += count

            batch_predictions.append(predictions)

        return batch_predictions, batch_loss, batch_cnt