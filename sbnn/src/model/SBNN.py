#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 28/04/2019 00:15

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
import itertools
import collections
import heapq
import copy
import math
from datetime import datetime
import logging
import sys
import cProfile as profile
import gc

log = logging.getLogger(__name__)

# 3rd party
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer.backends import cuda
from chainer.cuda import to_cpu

# modules
import src.util.constants as const
from src.util.util import DefaultOrderedDict
from src.util.batchprocessor import *

# import pdb

# from analysis import profile

class SBNN(chainer.Chain):
    def __init__(self, n_word_types, n_arg_type2id, n_role_types, params):
        super(SBNN, self).__init__()
        with self.init_scope():
            self.dim_embed = params['dim_embed']
            self.dim_bilstm = params['dim_bilstm']
            self.dim_role_type = params['dim_role_type']
            self.dim_arg_type = params['dim_arg_type']
            self.dropout = params['dropout']
            self.threshold = params['threshold']
            self.n_best = params['n_best']
            self.margin = params['margin']
            self.action_dim = params['dim_action']
            self.max_pos = params['max_pos']
            self.dim_pos = params['dim_pos']
            self.input_rep = params['input_rep']
            self.dim_level1_types = params['dim_level1_types']
            self.dim_level2_types = params['dim_level2_types']
            self.args = params

            # add abs position dimensions
            if 'pos_abs' in self.input_rep:  # word + pos_abs
                self.arg_type_and_word_dim = self.dim_arg_type + (self.dim_bilstm * 2) + self.dim_pos
            else:  # just word
                self.arg_type_and_word_dim = self.dim_arg_type + (self.dim_bilstm * 2)

            # add generalisation levels
            if self.args['level1_type_gen']:
                self.arg_type_and_word_dim = self.arg_type_and_word_dim + self.dim_level1_types
            if self.args['level2_type_gen']:
                self.arg_type_and_word_dim = self.arg_type_and_word_dim + self.dim_level2_types

            # add the role type dim
            self.relation_dim = self.arg_type_and_word_dim + self.dim_role_type + self.arg_type_and_word_dim

            # add relative position dim, not part of arg_type_and_word_dim since this describes the trigger and arg relative position
            if self.args['add_rel_pos']:
                self.relation_dim = self.relation_dim + self.dim_pos

            # add action dim
            self.relation_dim = self.relation_dim + self.action_dim

            self.hidden_dim = int(self.arg_type_and_word_dim / 2)

            # embed_init = chainer.initializers.Normal()
            self.embed_positiontype = L.EmbedID(self.max_pos, self.dim_pos, ignore_label=-1)
            self.embed_wordtype = L.EmbedID(n_word_types, self.dim_embed, ignore_label=-1)
            # self.embed_trigtype = L.EmbedID(n_trig_types, self.dim_arg_type,  ignore_label=-1)
            self.embed_roletype = L.EmbedID(n_role_types, self.dim_role_type, ignore_label=-1)
            self.embed_argtype = L.EmbedID(n_arg_type2id, self.dim_arg_type, ignore_label=-1)  # join the entity and trigger embedding matrix
            self.embed_action = L.EmbedID(len(const.ACTION_LIST) + 1, self.action_dim, ignore_label=-1)

            self.embed_rel_positiontype = L.EmbedID(self.max_pos * 2, self.dim_pos, ignore_label=-1)  # *2 for positive and negative positions
            self.embed_level1_typegen = L.EmbedID(self.dim_level1_types, self.dim_level1_types, ignore_label=-1)
            self.embed_level2_typegen = L.EmbedID(self.dim_level2_types, self.dim_level2_types, ignore_label=-1)

            self.bilstm = L.NStepBiLSTM(1, self.dim_embed, self.dim_bilstm, 0)

            self.linear_structure = L.Linear(None, self.relation_dim)
            self.linear_buffer = L.Linear(None, self.relation_dim)

            self.state_representation = L.Linear(None, self.arg_type_and_word_dim)
            self.linear1 = L.Linear(None, self.hidden_dim)
            self.linear2 = L.Linear(None, self.hidden_dim)
            self.linear = L.Linear(None, 2)

            # for relation attention
            self.rel_attn_lin = L.Linear(None, self.relation_dim)
            self.rel_context_vec = L.Linear(1, self.relation_dim, nobias=True)

            # structure attention
            self.WQ = L.Linear(self.relation_dim, self.relation_dim, nobias=True)
            self.WK = L.Linear(self.relation_dim, self.relation_dim, nobias=True)
            self.WV = L.Linear(self.relation_dim, self.relation_dim, nobias=True)

            # debugging variable
            self.files = set()
            self.instances = 0

    def load_pret_embed(self, path, vocab):
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=self.xp.float32)
                    self.embed_wordtype.W.data[vocab[word]] = vec
        log.info("Loaded embeddings: %s", path)

    def sequence_embed(self, xs, embed):
        """
        Embeds the sequences (batch).
        Args:
            xs: list of arrays of ids
            embed: embedding link
        Returns: the embedded sequence
        """
        xs_len = self.xp.array([len(x) for x in xs], dtype=self.xp.int32)
        xs_section = self.xp.cumsum(xs_len[:-1])
        xs_em = embed(F.concat(xs, axis=0))
        # if chainer.config.train:
        #     xs_em = F.dropout(xs_em, ratio=self.dropout)
        xs_embed = F.split_axis(xs_em, to_cpu(xs_section), axis=0)
        return xs_embed

    def extract_embed(self, xs, token_rep):
        """
        Extract embedding of sequences from token_rep.
        Args:
            xs: list of arrays of ids
            embed: embedding link
        Returns: the embedded sequence
        """
        xs_len = self.xp.array([len(x) for x in xs], dtype=self.xp.int32)
        xs_section = self.xp.cumsum(xs_len[:-1])
        new_xs = [self.xp.array(x) for x in xs]
        xs_temp = F.concat(new_xs, axis=0)
        xs_em = F.embed_id(xs_temp, token_rep)
        xs_embed = F.split_axis(xs_em, to_cpu(xs_section), axis=0)
        return xs_embed

    def get_enclosing_word_ind(self, sentence, targetword):
        '''
        We get the enclosing word index since it using the bilstm/word reps will capture the context of this word. Hence, this index where
        the word is part of will be represented better by its context rather than it's direct word representation only.
        :param sentence:
        :param targetword:
        :return:
        '''
        indx = []
        # if word is a subset of a word in sentence
        for s in sentence:
            for t in targetword.split():
                if t in s:
                    i = sentence.index(s)
                    indx.append(i)
                    break
        # if word is mapped to many words in the sentence
        if len(indx) == 0:
            for s in sentence:
                if targetword.startswith(s):
                    i = sentence.index(s)
                    l = len(s)
                    targetword = targetword[l:]
                    indx.append(i)
                    if len(targetword) == 0:
                        break
        return indx

    def _get_mention_inds(self, instance, mention, arg_id):
        '''Returns the indices of the mention within the sentence.'''
        word_inds = []
        sentence_ids = instance[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]
        for i in mention:
            if i in sentence_ids:
                ind = sentence_ids.index(i)
                word_inds.append(ind)
        if len(word_inds) == 0:
            if arg_id in instance[const.IDS_ENTITIES_MERGED_IDX]:
                defn = instance[const.IDS_ENTITIES_MERGED_IDX][arg_id]
            else:
                defn = instance[const.IDS_TRIGGERS_MERGED_IDX][arg_id]
            triggerword = ' '.join(defn[const.IDS_ARG_MENTION:])
            sentencewords = instance[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_WORD_INDICES]
            word_inds = self.get_enclosing_word_ind(sentencewords, triggerword)
        return word_inds


    def represent_input(self, word_ids, pos_ids):
        bilstm_batch = None
        pos_abs_embed = None
        if 'word' in self.input_rep:
            xs_embed = self.sequence_embed(word_ids, self.embed_wordtype)
            h, c, bilstm_batch = self.bilstm(None, None, xs_embed)
        if 'pos_abs' in self.input_rep:
            pos_abs_embed = self.sequence_embed(pos_ids, self.embed_positiontype)
        return bilstm_batch, pos_abs_embed

    def decide_input_rep(self, bilstm_batch, pos_embed, i):
        rep = None
        if 'word' in self.input_rep and 'pos_abs' in self.input_rep:
            rep = F.concat((bilstm_batch[i], pos_embed[i]), axis=1)
        elif 'word' in self.input_rep:
            rep = bilstm_batch[i]
        return rep

    def get_arg_rel_pos(self, trig_id, arg_id, instance):
        '''
        Computes the relative position of the argument wrt to the trig index.

        :param trig_idx: id of the trigger
        :param arg_idx: id of the argument
        :return: relative position of the argument
        '''
        # get the word ids of the arg/trig
        if arg_id in instance[const.IDS_ENTITIES_IDX]:
            arg_word_ids = instance[const.IDS_ENTITIES_IDX][arg_id][const.IDS_ARG_MENTION]
        else:
            arg_word_ids = instance[const.IDS_TRIGGERS_IDX][arg_id][const.IDS_ARG_MENTION]
        trig_word_ids = instance[const.IDS_TRIGGERS_IDX][trig_id][const.IDS_ARG_MENTION]

        # get the indices of the word ids
        arg_word_indices = self._get_mention_inds(instance, arg_word_ids, arg_id)
        trig_word_indices = self._get_mention_inds(instance, trig_word_ids, trig_id)

        # compute the rel pos of arg index vs trigger index, use first index for comparison
        rel_pos = arg_word_indices[0] - trig_word_indices[0]

        return rel_pos

    def relation_attention(self, relations):
        u = F.relu(self.rel_attn_lin(relations))
        us = F.matmul(u, self.rel_context_vec.W)
        weights = F.softmax(us, axis=0)
        structure_embed = weights * relations
        return structure_embed

    def represent_structure(self, structure, embed, rel_agg_attn):
        temp1 = F.reshape(structure, (structure.shape[0] * structure.shape[1], -1))
        emb_stack = embed(F.relu(temp1))
        rel_lin = F.relu(emb_stack)
        # aggregate
        if rel_agg_attn:  # do attention
            structure_embed = F.reshape(F.sum(self.relation_attention(rel_lin), 0), (1, self.relation_dim))
        else:  # sum by default
            temp2 = F.reshape(rel_lin, (structure.shape[0], structure.shape[1], -1))
            structure_embed = F.sum(temp2, axis=1)
        return structure_embed

    def predict_score(self, state):
        state_rep = self.state_representation(F.relu(state))
        h1 = F.relu(self.linear1(state_rep))
        h2 = F.relu(self.linear2(h1))  # TODO do i need this extra layer?
        raw_score = self.linear(h2)
        softmax_score = F.softmax(raw_score)
        return raw_score, softmax_score, state_rep

    def structure_attn(self, B, S):
        def convert_to_list(structure):
            rel_lst = []
            for s in structure:
                rel_lst.append(s[0])

            # stack them
            rel_stack = F.vstack(rel_lst)
            return rel_stack

        if len(B) > 0 and len(S) > 0:
            B_stack = convert_to_list(B)
            S_stack = convert_to_list(S)
            BS = F.vstack((B_stack, S_stack))
        elif len(S) > 0:
            S_stack = convert_to_list(S)
            BS = F.vstack((S_stack))
        elif len(B) > 0:
            B_stack = convert_to_list(B)
            BS = F.vstack((B_stack))

        # calculate key, query and value matrices
        Q = F.matmul(BS, F.transpose(self.WQ.W))
        K = F.matmul(BS, F.transpose(self.WK.W))
        V = F.matmul(BS, F.transpose(self.WV.W))
        # calculate the attention
        softmax = F.softmax(F.matmul(Q, F.transpose(K)) / F.sqrt(self.xp.array([K.shape[1]]).astype('float32')))
        finalscore = F.matmul(softmax, V)

        return finalscore

    def score_nbest(self, S, B):
        if self.args['struct_attn']:
            temp_sb = self.structure_attn(B, S[0])
            for i in range(len(S) - 1):
                temp = self.structure_attn(B, S[i + 1])
                temp_sb = F.vstack((temp_sb, temp))

        else:  # relation aggregation or sum
            temp_b = self.represent_structure(B, self.linear_buffer, self.args['rel_agg_attn'])
            temp_s = self.represent_structure(S, self.linear_structure, self.args['rel_agg_attn'])
            temp_sb = F.concat((temp_s, temp_b), axis=1)
        raw_score, softmax_score, state_rep = self.predict_score(temp_sb)
        return raw_score, softmax_score, state_rep

    def arg2ids(self, arg, triggers_ids, entities_ids, instance):
        is_trigger = arg in triggers_ids
        if is_trigger:
            defn = triggers_ids[arg]
            type_id = defn[const.IDS_ARG_TYPE]
        else:
            defn = entities_ids[arg]
            type_id = defn[const.IDS_ARG_TYPE]

        # arg level type id
        level1id = const.ID_NONE_LEVEL_1_TYPE
        if self.args['level1_type_gen']:
            level1id = instance[const.IDS_TYPE_GENERALISATION][arg][1]

        level2id = const.ID_NONE_LEVEL_2_TYPE
        if self.args['level2_type_gen']:
            level2id = instance[const.IDS_TYPE_GENERALISATION][arg][2]

        # arg mention id
        mention = defn[const.IDS_ARG_MENTION]
        mention_inds = self._get_mention_inds(instance, mention, arg)

        return (type_id, mention_inds, level1id, level2id)

    def convert2ids(self, instance, relations, entities_ids, triggers_ids, trig_ids, trigger):
        rel_ids = []

        for rel in relations:
            # role id
            role = rel[0]

            # arg type id
            arg = rel[1]
            arg_ids = self.arg2ids(arg, triggers_ids, entities_ids, instance)

            # rel pos id
            rel_pos = 0  # trigger and arg position is the same so rel pos is 0
            if self.args['add_rel_pos']:
                if rel[1] is const.ID_NONE_ARG_TYPE:
                    arg = trigger
                else:
                    arg = rel[1]
                rel_pos = self.get_arg_rel_pos(trigger, arg, instance)

            assert len(arg_ids) == 4, log.debug("Error: arg length more than 4")
            assert len(trig_ids) == 4, log.debug("Error: trig length more than 4")
            rel_id = [trig_ids[0], trig_ids[1], trig_ids[2], trig_ids[3], role, arg_ids[0], arg_ids[1], arg_ids[2], arg_ids[3], rel_pos]
            rel_ids.append(rel_id)
        return rel_ids

    def emb_arg(self, type, mention, l1, l2, token_rep):
        # embed types
        type_emb = self.embed_argtype(self.xp.array([type]).astype("i"))

        # extract mention ids TODO average the multi-word reps
        mention_emb = self.extract_embed(mention, token_rep)
        mention_emb = [F.average(x, axis=0) for x in mention_emb]
        mention_emb = F.vstack(mention_emb)
        mention_emb = F.reshape(mention_emb, (1, mention_emb.shape[0], mention_emb.shape[1]))

        # embed level 1 ids
        final = F.concat((type_emb, mention_emb), axis=2)
        if self.args['level1_type_gen']:
            l1_emb = self.embed_level1_typegen(self.xp.array([l1]).astype("i"))
            final = F.concat((final, l1_emb), axis=2)

        # embed level 2 ids
        if self.args['level2_type_gen']:
            l2_emb = self.embed_level2_typegen(self.xp.array([l2]).astype("i"))
            final = F.concat((final, l2_emb), axis=2)

        return final

    def embed_relations(self, zip_ids, token_rep):

        # embed trigger
        trigger = self.emb_arg(zip_ids[0], zip_ids[1], zip_ids[2], zip_ids[3], token_rep)

        # embed role
        role_emb = self.embed_roletype(self.xp.array([zip_ids[4]]).astype("i"))

        # embed arg TODO or replace this with the event embedding if it's been predicted down the level
        # check if arg has been predicted previously, if so create as many events
        arg = self.emb_arg(zip_ids[5], zip_ids[6], zip_ids[7], zip_ids[8], token_rep)

        # embed rel pos ids
        final = F.concat((trigger, role_emb, arg), axis=2)
        if self.args['add_rel_pos']:
            rel_pos_emb = self.embed_rel_positiontype(self.xp.array([zip_ids[9]]).astype("i"))
            final = F.concat((final, rel_pos_emb), axis=2)
        return final

    def extract_arg_embeddings(self, edges, relation_reps, trigger_subevent_embeddings=None):
        '''
        If relations is empty, then create zeros for the structure.
        Else,
            1) form the combinations extracting the index of the argument event embedding
            2) create the combinations
            3) create the structure embeddings while replacing the event argument with event embeddings
        :param trigger_subevent_embeddings: contains the dictionary of trigger and its list of event representations
        :param edges: contains the edges before this timestep
        :param relation_reps: contains the relation representations before this timestep
        :return:
        '''
        # if arg_id not in trigger_subevent_embeddings:
        #     F.reshape(relation_reps[timestep], (1, 1, -1))
        #
        # if len(edges) == 0:
        #     stacked_rel_embs = F.reshape(Variable(self.xp.zeros((self.relation_dim - self.action_dim), dtype=self.xp.float32)), (1, 1, -1))
        # else:
        # extract the combinations
        lst = []
        for i, relation in enumerate(edges):
            arg_id = relation[1][1]
            sublst = []
            emb = []
            if trigger_subevent_embeddings is not None:
                emb = trigger_subevent_embeddings[arg_id]

            if len(emb) != 0:
                for j in range(len(emb)):
                    sublst.append((arg_id, j))
            else:
                sublst.append((arg_id, -1))  # relation_reps contains all the relations under the trigger, -1 means it does not have a sub-event
            lst.append(sublst)

        # create the combinations
        final_list_of_combination = []
        for comb in itertools.product(*lst):
            l = list(comb)
            final_list_of_combination.append(l)

        # form the new structure embeddings while replacing the argument in each relation with the event embedding specified in the index
        new_structure_embs = []
        for structure in final_list_of_combination:
            new_rel_embs = []
            for i, relation in enumerate(structure):
                arg_id = relation[0]
                index = relation[1]
                rep = relation_reps[i]
                if index != -1:  # replace the arg rep with the event rep
                    event = trigger_subevent_embeddings[arg_id][index]
                    trigger_and_role = relation_reps[i][0:self.arg_type_and_word_dim + self.dim_role_type]
                    rest = relation_reps[i][self.arg_type_and_word_dim + self.dim_role_type + self.arg_type_and_word_dim:]
                    rep = F.concat((trigger_and_role, event, rest), axis=0)
                new_rel_embs.append(rep)
            stacked_new_rel_embs = F.stack(new_rel_embs)
            new_structure_embs.append(stacked_new_rel_embs)
        stacked_rel_embs = F.stack(new_structure_embs)
        return stacked_rel_embs

    def add_action_embeddings(self, prev_action, structure_emb, actions_emb, towhich):
        '''
        add action to all structures and in each structure the action added to each relation is based on the sequence of the previous action sequence
        :param prev_action: list of sequence of actions
        :param common_structure_emb: the list of structures which is a list of relations
        :param actions_emb: embeddings for all actions
        :return:
        '''
        action_seq_emb = F.embed_id(self.xp.array(prev_action), actions_emb)  # go thru each action and add 1 to move index 1 up
        # new_action = F.expand_dims(action_seq_emb, axis=0)  # for common, no repeats as rels as actions equals rels
        # if new_action.shape[1] > common_structure_emb.shape[1]:  # for the arg, repeat the structure to num actions then reshape new_action
        structure_wd_action_emb = None
        if towhich == 'A':
            assert structure_emb.shape[1] == 1, 'Error: there should only be 1 relation in A'
            new_action = F.expand_dims(action_seq_emb, axis=1)
            num_structures = structure_emb.shape[0]
            num_actions = new_action.shape[0]
            structure_emb = F.repeat(structure_emb, num_actions, axis=0) # repeat structures to each action
            new_action = F.repeat(new_action, num_structures, axis=0)
            assert new_action.shape[0] == structure_emb.shape[0], log.debug("new_action.shape[0] != structure_emb.shape[0]")
            structure_wd_action_emb = F.concat((new_action, structure_emb), axis=2)
            # new_action = F.reshape(new_action, (new_action.shape[0] * new_action.shape[1], 1, -1))
            # new_action = F.reshape(new_action, (new_action.shape[1], 1, -1))
        # elif new_action.shape[1] < common_structure_emb.shape[1]:  # for buffer, repeat the actions
        elif towhich == 'B':
            new_action = F.expand_dims(action_seq_emb, axis=0)
            structure_emb = F.expand_dims(structure_emb, axis=0)
            new_action = F.repeat(new_action, structure_emb.shape[1], axis=1)
            assert new_action.shape[1] == structure_emb.shape[1], log.debug("new_action.shape[1] != common_structure_emb.shape[1]")
            structure_wd_action_emb = F.concat((new_action, structure_emb), axis=2)
        elif towhich == 'S':
            new_action = F.expand_dims(action_seq_emb, axis=0)
            if new_action.shape[0] < structure_emb.shape[0]:
                new_action = F.repeat(new_action, structure_emb.shape[0], axis=0)
        # assert new_action.shape[1] == structure_emb.shape[1], log.debug("new_action.shape[1] != common_structure_emb.shape[1]")
            structure_wd_action_emb = F.concat((new_action, structure_emb), axis=2)
        return structure_wd_action_emb

    def extract_target_labels(self, prev_actions, gold_actions, num_sets):
        targets = []
        new_actions = []
        # create actions

        for _ in range(num_sets):
        #     actions.extend(const.ACTION_LIST)
            for prev_action in prev_actions:
                if prev_action == [-1]:  # remove the first state and set it to empty
                    prev_action = []
                base = prev_action[:]

                actions = []
                actions.extend(const.ACTION_LIST)

                for action in actions:
                    new_action = base + [action]
                    target_label = 0
                    if chainer.config.train:
                        for g, gold_action in gold_actions.items():
                            if gold_action == new_action:
                                target_label = 1
                                break
                    targets.append(target_label)
                    new_actions.append(new_action)
        return targets, new_actions

    def create_newstructures(self, relation, actual_relation, cur_structures, num_sets):
        new_structures = []
        # create actions
        # actions = []
        # for _ in range(num_sets):
        #     actions.extend(const.ACTION_LIST)
        for _ in range(num_sets):
            for cur_structure in cur_structures:
                actions = []
                actions.extend(const.ACTION_LIST)
                for action in actions:
                    action_entry = (relation, action, actual_relation)
                    temp_s = list(cur_structure.copy())
                    temp_s.append(action_entry)
                    new_structures.append(temp_s)
        return new_structures
    def extract_and_embed_relations(self, trigger, triggers_ids, entities_ids, instance, edges, token_rep, true_edges):
        # trigger ids
        trig_ids = self.arg2ids(trigger, triggers_ids, entities_ids, instance)
        # extract relation ids from edges
        relation_ids = self.convert2ids(instance, edges, entities_ids, triggers_ids, trig_ids, trigger)
        # extract relation embs
        zip_ids = list(zip(*[i for i in relation_ids]))
        relation_reps = F.reshape(self.embed_relations(zip_ids, token_rep), (len(relation_ids), -1))
        # combine ids and edges
        relations = list(zip(relation_ids, edges, true_edges))
        return relation_reps, relations

    def select_nbest(self, pq):
        sorted_list = sorted(pq, key=lambda state: state[0], reverse=True)  # sort in descending order
        nbest_states = sorted_list[:self.n_best] if len(sorted_list) >= self.n_best else sorted_list[:]
        return nbest_states

    def extract_gold_actions(self, num_events, timestep, gold_actions, trig_structure):
        for l in range(num_events):
            label = trig_structure[2][l][timestep][1]
            label_action = label.index(1)
            if l in gold_actions:
                temp = gold_actions[l]
                temp.append(label_action)
                gold_actions[l] = temp
            else:
                gold_actions[l] = [label_action]
        return gold_actions
    def fix_events(self, nbest_states, trigger_subevent_embeddings, final_event_structures, states_dict, trigger):
        to_be_removed = []
        for n in range(len(nbest_states)):
            _, temp_state_ctr = nbest_states[n]
            temp_structure, temp_state_rep, temp_action, _, raw_score, softmax_score = states_dict[temp_state_ctr]

            # found an event, event before the last edge
            if temp_action[-1] == const.ACTION_ADDFIX:
                # always remove addfix from beam
                to_be_removed.append(temp_state_ctr)

                if softmax_score.data[1] >= self.threshold:
                    trigger_subevent_embeddings[trigger].append(temp_state_rep)
                    final_event_structures[trigger].append([temp_structure, raw_score])
            if temp_action == [1]:  # remove adding NONE in the first timestep
                to_be_removed.append(temp_state_ctr)
        return to_be_removed

    def remove_fixed_events(self, nbest_states, to_be_removed):
        for item in to_be_removed:
            for i, (_, temp_state_ctr) in enumerate(nbest_states):
                if item == temp_state_ctr:
                    del nbest_states[i]
                    break

    def compute_loss(self, pq, states_dict, nbest_states, early_update):
        # check 1: if gold is out of beam: gold but not nbest, early update
        ## idea: score is closer to 0 but expected is 1 so move score closer to 1
        to_be_updated = collections.OrderedDict()
        for n in range(len(pq)):  # loop through all states
            _, temp_state_ctr = pq[n]
            _, _, temp_action, temp_target_label, raw_score, _ = states_dict[temp_state_ctr]
            gold_predicted = False
            if temp_target_label == 1:  # if it is gold
                for q in range(len(nbest_states)):  # see if it is predicted
                    _, temp_state_ctr_2 = nbest_states[q]
                    if temp_state_ctr == temp_state_ctr_2:  # gold was predicted
                        gold_predicted = True
                        break
                if not gold_predicted:  # add to list to be updated if not predicted
                    to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)
                    early_update = True

        # check 2: nbest but not gold, nbest but less than threshold
        ## idea: score is close to 1 but expected is 0 so move score closer to 0
        for n in range(len(nbest_states)):
            _, temp_state_ctr = nbest_states[n]
            _, _, temp_action, temp_target_label, raw_score, softmax_score = states_dict[temp_state_ctr]

            # nbest action but not gold
            if temp_target_label == 0:  # not gold
                to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)

            # nbest and gold but less than threshold
            ## idea: score is less than threshold but expected to be close to 1 so move score closer to 1
            if temp_action[-1] == const.ACTION_ADDFIX and temp_target_label == 1:
                if softmax_score.data[1] < self.threshold:
                    to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)

        # compute the loss and update
        orig_scores = []
        targets = []
        for key, value in to_be_updated.items():
            _, _, _, temp_target_label, orig_score, _ = states_dict[key]
            orig_scores.append(orig_score)
            targets.append(temp_target_label)

        loss = F.softmax_cross_entropy(F.stack(orig_scores), self.xp.array(targets))
        return loss, early_update

    def profile_predict(self, instance, token_rep):
        profile.runctx('self._predict(instance, token_rep)', globals(), locals())

    def extract_nbest_emb(self, actions_emb, relation_reps, nbest_states, states_dict):

        actions = []
        relations = []
        prev_actions = []
        cur_structures = []
        for pqi, (_, state_ctr) in enumerate(nbest_states):
            cur_structure, _, prev_action, _, _, _ = states_dict[state_ctr]
            actions.append([i + 1 for i in prev_action])
            relations.append([i for i in range(len(prev_action))])
            prev_actions.append(prev_action)
            cur_structures.append(cur_structure)

        # action embedding
        act_emb = F.embed_id(self.xp.array(actions), actions_emb)

        # relation embedding
        rel_emb = F.embed_id(self.xp.array(relations), relation_reps)

        return F.concat((rel_emb, act_emb), axis=2), prev_actions, cur_structures

    def _predict(self, instance, token_rep):

        # gather triggers in bottom-up sequence
        levels = instance[const.PRED_CAND_STRUCTURES_IDX]
        entities_ids = instance[const.PRED_ENTITIES_IDX]
        triggers_ids = instance[const.PRED_TRIGGERS_IDX]

        # add id of NONE TODO not sure if this should be here
        entities_ids[const.NONE_ARG_TYPE] = [const.ID_NONE_ARG_TYPE, '', '', [const.ID_NONE_WORD_TYPE]]

        # sentence level variables for sentence-level learning
        trigger_subevent_embeddings = DefaultOrderedDict(list)
        final_event_structures = DefaultOrderedDict(list)
        instance_loss = 0.0

        early_update = False

        # debugging variable
        file_id = instance[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_FILE_ID_IDX]
        if file_id not in self.files:
            self.files.add(file_id)
        self.instances += 1
        log.debug('Processing: %s \t %s \t %s', file_id, len(self.files), self.instances)

        num_classification = 0

        # extract action embs from ALL actions including NONE, since same for the batch it can be retrieve once
        actions = [0] + [a + 1 for a in const.ACTION_LIST]
        actions_emb = self.embed_action(self.xp.array(actions).astype("i"))

        # loop through triggers in sequence thru levels
        # TODO process all triggers in the same level at once
        for level in levels:

            for trigger, trigstructures in level.items():

                # if trigger == 'TR47':
                #     print()

                # -- extract info -- #
                trig_structure = trigstructures[0]  # there's only one
                true_edges = trig_structure[0]
                edges = trig_structure[1]
                log.debug("trigger: %s, edges: %s", trigger, edges)

                # -- extract ids and reps -- #
                # TODO can be done once for all triggers in this level
                relation_reps, relations = self.extract_and_embed_relations(trigger, triggers_ids, entities_ids, instance, edges, token_rep, true_edges)

                # -- initial state -- #
                pq = []
                states_dict = DefaultOrderedDict()
                structure = []
                ctr = 0
                # must have a value previous action for the add_action_embeddings to work
                states_dict[ctr] = (structure, None, [const.ACTION_NONE], -1, 0, 0)
                pq.append((0.0, ctr))
                ctr += 1

                # init gold actions
                gold_actions = None
                if chainer.config.train:
                    num_events = len(trig_structure[2])
                    gold_actions = collections.OrderedDict()  # accumulates the gold actions thru timesteps

                # select n-best
                nbest_states = self.select_nbest(pq)


                # loop through timesteps
                for timestep, (_, relation, actual_relation) in enumerate(relations):

                    # reset pq: every time step has a new pq where to select the nbest from,
                    # TODO the idea is that having one pq for all time steps may slow down the search but can also be tried
                    pq = new_structures = state_reps = new_actions = target_labels = raw_scores = softmax_scores = temp = arg_structure_emb = structures_emb = new_buffer_wd_actions_emb = []

                    # check if at the current time step the entry has sub-events which were not predicted and stop the search (break)
                    # TODO can be changed to continue to next timestep as next argument may have been predicted down the level but too many things to consider below if so
                    arg_id = relation[1]
                    log.debug("\targ id: %s", arg_id)
                    if arg_id.startswith("TR"):
                        if arg_id not in trigger_subevent_embeddings:
                            log.debug("arg id not predicted in previous level: %s, %s, %s", arg_id, trigger, file_id)
                            break

                    # extract gold labels
                    if chainer.config.train:
                        gold_actions = self.extract_gold_actions(num_events, timestep, gold_actions, trig_structure)

                    # -- extract buffer emb and add action-- #
                    # NOTE no need to create combinations based on the predicted argument events since this just serves as context and no final structure will be created from this
                    # buffer_emb = self.extract_arg_embeddings(relations[timestep + 1:], relation_reps[timestep + 1:]) # [structures x relations x emb] TODO: can be done before the timestep loop??
                    # buffer_emb = self.extract_arg_embeddings(relations[:], relation_reps[:])



                    # just index the buffer representations based on the time step
                    log.debug("\t\tcreating buffer emb..")
                    buffer_emb = relation_reps[timestep+1:]
                    if buffer_emb.shape[0] == 0:
                        buffer_emb = F.reshape(Variable(self.xp.zeros((self.relation_dim - self.action_dim), dtype=self.xp.float32)), (1, -1))
                    buffer_wd_action_emb = self.add_action_embeddings([const.ACTION_NONE + 1], buffer_emb, actions_emb, 'B')
                    # buffer_actions = F.expand_dims(F.repeat(actions_emb[const.ACTION_NONE + 1], buffer_emb.shape[0], axis=0), axis=0)
                    # buffer_wd_action_emb = F.concat((buffer_emb, buffer_actions), axis=1)

                    # --- extract the common structure emb -- #
                    # can produce more structures but num relations in each will be the same up to before this time step
                    # common_structure_emb = self.extract_arg_embeddings(relations[:timestep], relation_reps[:timestep], trigger_subevent_embeddings)
                    # common_structure_emb = self.extract_arg_embeddings(relations[:], relation_reps[:], trigger_subevent_embeddings)

                    # -- construct the current relation with actions -- #
                    # can produce more structures but relations will always be one
                    # TODO need to confirm is subscripting it will not destroy the computational graph
                    # args_structure_emb = self.extract_arg_embeddings([relations[timestep]], F.reshape(relation_reps[timestep], (1, -1)), trigger_subevent_embeddings)
                    # args_structure_emb = self.extract_arg_embeddings([relations[1]], F.reshape(relation_reps[1], (1, -1)), trigger_subevent_embeddings)
                    # args_structure_wd_actions_emb = self.add_action_embeddings(const.ACTION_LIST, args_structure_emb, actions_emb) # [structures x relations x emb]

                    # arg_structure_emb = trigger_subevent_embeddings[arg_id] if arg_id in trigger_subevent_embeddings else F.reshape(relation_reps[timestep], (1, 1, -1))
                    arg_structure_emb = self.extract_arg_embeddings([relations[timestep]], F.reshape(relation_reps[timestep], (1, -1)), trigger_subevent_embeddings)
                    log.debug("\t\targ count: %s", arg_structure_emb.shape[0])
                    arg_structure_wd_actions_emb = self.add_action_embeddings(const.ACTION_LIST, arg_structure_emb, actions_emb, 'A')

                    # loop thru the nbest states
                    # prev_structures = [] # only for this current set of nbest states
                    # prev_structures_dct = OrderedDict()
                    # assumption: the nbest states are all from the same time step
                    # TODO: multiply all the nbest with actions
                    '''
                    Since the nbest already contains all the best states up to the current timestep, expand from them only.
                    At the moment, I used the prev action only to generate actions
                    '''

                    # represent the structures from the nbest
                    log.debug("\t\textracting nbest..")
                    nbest_emb, prev_actions, cur_structures = self.extract_nbest_emb(actions_emb, relation_reps, nbest_states, states_dict)


                    # join: nbest + arg + buffer
                    # concat the common and arg to be the structure
                    # new_common_structure_wd_actions_emb = F.repeat(nbest_emb, arg_structure_wd_actions_emb.shape[0], axis=0)
                    # new_args_structure_wd_actions_emb = F.repeat(arg_structure_wd_actions_emb, nbest_emb.shape[0], axis=0)
                    structures_emb = F.concat((F.repeat(nbest_emb, arg_structure_wd_actions_emb.shape[0], axis=0), F.repeat(arg_structure_wd_actions_emb, nbest_emb.shape[0], axis=0)), axis=1) # [structures x relations x emb]
                    # expand buffer
                    new_buffer_wd_actions_emb = F.repeat(buffer_wd_action_emb, structures_emb.shape[0], axis=0)

                    # score the buffer + structure, final result will depend on the length of structure
                    log.debug("\t\tscoring nbest..")
                    raw_scores, softmax_scores, state_reps = self.score_nbest(structures_emb, new_buffer_wd_actions_emb)

                    # determine target labels
                    # num_sets = int(structures_emb.shape[0] / len(const.ACTION_LIST))
                    target_labels, new_actions = self.extract_target_labels(prev_actions, gold_actions, arg_structure_emb.shape[0])

                    # create the new structures with the entry
                    new_structures = self.create_newstructures(relation, actual_relation, cur_structures, arg_structure_emb.shape[0])

                    log.debug("\t\tnew structures: %s", len(new_structures))
                    # push scores to with target labels and new actions
                    log.debug("\t\tcreating a zip of new structures..")
                    if chainer.config.train:
                        temp = list(zip(new_structures, state_reps, new_actions, target_labels, raw_scores, softmax_scores))
                    else:
                        temp = list(zip(new_structures, state_reps, new_actions, [-1 for i in range(len(new_structures))], raw_scores, softmax_scores))
                    log.debug("\t\tpushing scores to the pq..")
                    for i in range(len(temp)):
                        states_dict[ctr] = temp[i]
                        pq.append((softmax_scores[i].data[1], ctr))
                        ctr += 1

                    # for k in range(len(new_structures)):
                    #     # if k == 0 or ( k > 0 and k % len(const.ACTION_LIST) != 0): # save only one IGNORE action for this timestep
                    #         # push its respective info to the pq
                    #     if chainer.config.train:
                    #         try:
                    #             states_dict[ctr] = (new_structures[k], state_reps[k], new_actions[k], target_labels[k], raw_scores[k], softmax_scores[k])
                    #         except:
                    #             print()
                    #     else:
                    #         states_dict[ctr] = (new_structures[k], state_reps[k], new_actions[k], -1, raw_scores[k], softmax_scores[k])
                    #
                    #     # push the score with the state primary key (ctr)
                    #     pq.append((softmax_scores[k].data[1], ctr))
                    #
                    #     # add ctr for the state index
                    #     ctr += 1



                    # for pqi, (_, state_ctr) in enumerate(nbest_states):
                    #     cur_structure, _, prev_action, _, _, _ = states_dict[state_ctr]
                    #     log.debug("nbest: %s cur_structure: %s prev_action: %s", pqi, cur_structure, prev_action)
                    #     if prev_action not in prev_structures: # to prevent repetition of representations of same structures
                    #         # -- add prev actions to common structure emb -- #
                    #         common_structure_wd_actions_emb = self.add_action_embeddings([(i + 1) for i in prev_action], common_structure_emb, actions_emb)
                    #
                    #         # concat the common and arg to be the structure
                    #         new_common_structure_wd_actions_emb = F.repeat(common_structure_wd_actions_emb, args_structure_wd_actions_emb.shape[0], axis=0)
                    #         new_args_structure_wd_actions_emb = F.repeat(args_structure_wd_actions_emb, common_structure_wd_actions_emb.shape[0], axis=0)
                    #         structures_emb = F.concat((new_common_structure_wd_actions_emb, new_args_structure_wd_actions_emb), axis=1) # [structures x relations x emb]
                    #         # expand buffer
                    #         new_buffer_wd_actions_emb = F.repeat(buffer_wd_action_emb, structures_emb.shape[0], axis=0)
                    #
                    #         # score the buffer + structure, final result will depend on the length of structure
                    #         raw_scores, softmax_scores, state_reps = self.perform_action(structures_emb, new_buffer_wd_actions_emb)
                    #
                    #
                    #         log.debug("num structures:%s", structures_emb.shape[0])
                    #
                    #         # determine target labels for all actions for this state
                    #         num_sets = int(structures_emb.shape[0] / len(const.ACTION_LIST))
                    #         target_labels, new_actions = self.extract_target_labels(prev_action, gold_actions, num_sets)
                    #
                    #         # create the new structures with the entry
                    #         new_structures = self.create_newstructures(relation, actual_relation, cur_structure, num_sets)
                    #
                    #         # add the prev action and store the values in the dict
                    #         prev_structures.append(prev_action)
                    #         indx = prev_structures.index(prev_action)
                    #         prev_structures_dct[indx] = (prev_action, raw_scores, softmax_scores, state_reps, new_structures, target_labels, new_actions)
                    #
                    #     # determine which structures to add to pq
                    #     indx = prev_structures.index(prev_action)
                    #     _, raw_scores, softmax_scores, state_reps, new_structures, target_labels, new_actions = prev_structures_dct[indx]
                    #
                    #     # push scores to with target labels and new actions
                    #     for k in range(len(new_structures)):
                    #         if k == 0 or ( k > 0 and k % len(const.ACTION_LIST) != 0): # save only one IGNORE action for this timestep
                    #             # push its respective info to the pq
                    #             if chainer.config.train:
                    #                 states_dict[ctr] = (new_structures[k], state_reps[k], new_actions[k], target_labels[k], raw_scores[k], softmax_scores[k])
                    #             else:
                    #                 states_dict[ctr] = (new_structures[k], state_reps[k], new_actions[k], -1, raw_scores[k], softmax_scores[k])
                    #
                    #             # push the score with the state primary key (ctr)
                    #             pq.append((softmax_scores[k].data[1], ctr))
                    #
                    #             # add ctr for the state index
                    #             ctr += 1

                    # # add margin to all non-gold actions
                    # '''
                    # idea: a margin value is added to the non-gold actions' scores and thus chances are they will
                    # be selected in nbest and then they will be updated until the scores are lower than the gold
                    # actions by the margin
                    # '''
                    # if chainer.config.train:
                    #     for k in range(len(pq)):
                    #         temp_score, temp_state_ctr = pq[k]
                    #         temp_buffer, temp_structure, temp_state_rep, temp_new_action, target_label, raw_score, softmax_score, combi = states_dict[temp_state_ctr]
                    #         if target_label == 0:  # non-gold
                    #             softmax_score = softmax_score + self.margin
                    #             raw_score = raw_score - self.margin #TODO - or +?
                    #             states_dict[temp_state_ctr] = (temp_buffer, temp_structure, temp_state_rep, temp_new_action, target_label, raw_score, softmax_score, combi)

                    # select n-best
                    log.debug("\t\tselecting nbest..")
                    nbest_states = self.select_nbest(pq)

                    # fix and remove events
                    log.debug("\t\tconstructing events..")
                    to_be_removed = self.fix_events(nbest_states, trigger_subevent_embeddings, final_event_structures, states_dict, trigger)

                    # check which states to update #TODO should the early update be on validation as well??
                    if chainer.config.train:
                        log.debug("\t\tcomputing loss..")
                        loss, early_update = self.compute_loss(pq, states_dict, nbest_states, early_update)
                        instance_loss += loss

                    log.debug("\t\ttrigger: %s timestep: %s entry: %s pq: %s nbest: %s to_remove: %s final_nbest: %s", trigger, timestep, actual_relation, len(pq), len(nbest_states), len(to_be_removed), len(nbest_states)-len(to_be_removed))

                    # -- remove fixed events from pq  -- #
                    ## idea: they won't be expanded anymore
                    log.debug("\t\tremoving constructed events..")
                    self.remove_fixed_events(nbest_states, to_be_removed)

                    # if a gold is out of the beam, don't proceed to next timestep/argument
                    log.debug("\t\tchecking if needs early update: %s", early_update)
                    if chainer.config.train:
                        if self.args['early_update']:
                            if early_update:
                                break

        # predictions = final_event_structures
        return final_event_structures, instance_loss, num_classification

    def __call__(self, batch):
        batch_predictions = []

        # prepare list of list of word ids and positions
        xs = []
        pos = []
        for i in batch:
            # add NONE word at the beginning
            i[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX].insert(0, const.ID_NONE_WORD_TYPE)
            i[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_WORD_INDICES].insert(0, const.NONE_WORD_TYPE)
            sen = i[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]
            xs.append(self.xp.array(sen))
            pos.append(self.xp.array([(j) for j in range(len(sen))]))

        # embed word and absolute position
        bilstm_batch, pos_embed = self.represent_input(xs, pos)

        batch_loss = 0
        batch_cnt = 0

        # structures = []
        # for buffer, arg in  buffer_arg_generator(batch):
        #     newstructures = apply_actions(structures, buffer, arg)

        for i in range(len(batch)):  # sentences
            instance = batch[i]
            input_rep = self.decide_input_rep(bilstm_batch, pos_embed, i)

            if chainer.config.train:
                predictions, instance_loss, _ = self._predict(instance, input_rep)

                batch_loss += instance_loss

                # # method profiling
                # self.profile_predict(instance, input_rep)
                # break

            else:
                predictions, _, count = self._predict(instance, input_rep)
                batch_cnt += count
            batch_predictions.append(predictions)

        log.debug("freeing memory..")
        gc.collect()

        return batch_predictions, batch_loss, batch_cnt
