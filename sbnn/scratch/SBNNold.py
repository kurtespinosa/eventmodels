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
            if 'pos_abs' in self.input_rep: # word + pos_abs
                self.arg_type_and_word_dim = self.dim_arg_type + (self.dim_bilstm * 2) + self.dim_pos
            else: # just word
                self.arg_type_and_word_dim = self.dim_arg_type + (self.dim_bilstm * 2)


            # add generalisation levels
            if self.args['level1_type_gen'] and self.args['level2_type_gen']:
                self.arg_type_and_word_dim = self.arg_type_and_word_dim + self.dim_level1_types + self.dim_level2_types
            elif self.args['level1_type_gen']:
                self.arg_type_and_word_dim = self.arg_type_and_word_dim + self.dim_level1_types
            elif self.args['level2_type_gen']:
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
            self.embed_positiontype = L.EmbedID(self.max_pos, self.dim_pos,  ignore_label=-1)
            self.embed_wordtype = L.EmbedID(n_word_types, self.dim_embed,  ignore_label=-1)
            # self.embed_trigtype = L.EmbedID(n_trig_types, self.dim_arg_type,  ignore_label=-1)
            self.embed_roletype = L.EmbedID(n_role_types, self.dim_role_type, ignore_label=-1)
            self.embed_argtype = L.EmbedID(n_arg_type2id, self.dim_arg_type, ignore_label=-1) # join the entity and trigger embedding matrix
            self.embed_action = L.EmbedID(len(const.ACTION_LIST)+1, self.action_dim,  ignore_label=-1)

            self.embed_rel_positiontype = L.EmbedID(self.max_pos*2, self.dim_pos,  ignore_label=-1) # *2 for positive and negative positions
            self.embed_level1_typegen = L.EmbedID(self.dim_level1_types, self.dim_level1_types,  ignore_label=-1)
            self.embed_level2_typegen = L.EmbedID(self.dim_level2_types, self.dim_level2_types,  ignore_label=-1)

            self.bilstm = L.NStepBiLSTM(1, self.dim_embed, self.dim_bilstm, 0)

            self.linear_structure = L.Linear(None, self.relation_dim)
            self.linear_buffer = L.Linear(None, self.relation_dim)

            self.state_representation = L.Linear(None, self.arg_type_and_word_dim)
            self.linear1 = L.Linear(None, self.hidden_dim)
            self.linear2 = L.Linear(None, self.hidden_dim)
            self.linear = L.Linear(None, 2)

            # for relation attention
            self.rel_attn_lin = L.Linear(None, self.relation_dim)
            self.rel_context_vec = L.Linear(1, self.relation_dim , nobias=True)

            # structure attention
            self.WQ = L.Linear(self.relation_dim, self.relation_dim, nobias=True)
            self.WK = L.Linear(self.relation_dim, self.relation_dim, nobias=True)
            self.WV = L.Linear(self.relation_dim, self.relation_dim, nobias=True)


    def load_pret_embed(self, path, vocab):
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
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
        xs_len = [len(x) for x in xs]
        xs_section = self.xp.cumsum(xs_len[:-1])
        xs_em = embed(F.concat(xs, axis=0))
        # if chainer.config.train: #TODO enable this in the final code as without this there is no dropout
        #     xs_em = F.dropout(xs_em, ratio=self.dropout)
        xs_embed = F.split_axis(xs_em, xs_section, axis=0)
        return xs_embed

    def extract_embed(self, xs, token_rep):
        """
        Extract embedding of sequences from token_rep.
        Args:
            xs: list of arrays of ids
            embed: embedding link
        Returns: the embedded sequence
        """
        xs_len = [len(x) for x in xs]
        xs_section = self.xp.cumsum(xs_len[:-1])
        xs_temp = F.concat(self.xp.array([self.xp.array(x) for x in xs]), axis=0)
        xs_em = F.embed_id(xs_temp, token_rep)
        xs_embed = F.split_axis(xs_em, xs_section, axis=0)
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
                try:
                    defn = instance[const.IDS_TRIGGERS_MERGED_IDX][arg_id]
                except:
                    print()
            triggerword = ' '.join(defn[const.IDS_ARG_MENTION:])
            sentencewords = instance[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_WORD_INDICES]
            word_inds = self.get_enclosing_word_ind(sentencewords, triggerword)
        return word_inds

    # def _represent_mentions(self, mention_inds, sentence_rep):
    #     '''
    #     Given the indices of the mention, retrieve their representation.
    #
    #     If the mention is the a multi-word, average the representation.
    #     Assumes that indices are present in the sentence_rep, otherwise,
    #     :param mention_ids:
    #     :param sentence_rep:
    #     :return:
    #     '''
    #     final_mention_representation = None
    #     try:
    #         # use embed_id function because the representations are already computed
    #         final_mention_representation = F.embed_id(self.xp.asarray(mention_inds).astype('i'), sentence_rep)
    #         if len(mention_inds) > 1: #TODO maybe this can be removed and maybe generalise this
    #             final_mention_representation = F.average(final_mention_representation, axis=0)
    #     except Exception as e:
    #         log.error("Error occured:" + str(e), exc_info=True)
    #         log.exception(str(e))
    #
    #     return final_mention_representation

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

    # def _represent_type_and_argument(self, instance, arg_ids, type_index, token_rep, arg_id, structures_above_threshold=None):
    #
    #     embedding_list = []
    #     if structures_above_threshold is not None:
    #         embedding_list = structures_above_threshold[arg_id]
    #     else:
    #         # embed type
    #         defn = arg_ids[arg_id]
    #         type_id = defn[const.IDS_ARG_TYPE]
    #         type_embedding = None
    #         if type_index == const.IDS_TRIGGERS_IDX:
    #             type_embedding = self.embed_argtype(np.array([type_id]).astype("i"))
    #         elif type_index == const.IDS_ENTITIES_IDX:
    #             type_embedding = self.embed_argtype(np.array([type_id]).astype("i"))
    #
    #         # embed type level
    #         if self.args['level1_type_gen']:
    #             level1id = instance[const.IDS_TYPE_GENERALISATION][arg_id][1]
    #             arg_level1_embed = self.embed_level1_typegen(self.xp.array([level1id]).astype("i"))
    #             # l1_embed = F.flatten(arg_level1_embed)
    #             type_embedding = F.concat((type_embedding, arg_level1_embed), axis=1)
    #         if self.args['level2_type_gen']:
    #             level2id = instance[const.IDS_TYPE_GENERALISATION][arg_id][2]
    #             arg_level2_embed = self.embed_level2_typegen(self.xp.array([level2id]).astype("i"))
    #             # l2_embed = F.flatten(arg_level2_embed)
    #             type_embedding = F.concat((type_embedding, arg_level2_embed), axis=1)
    #
    #         # embed mention
    #         mention = defn[const.IDS_ARG_MENTION]
    #         if [0] == mention: # if NONE, TODO but this will get the rep of the first word in the sentence!!
    #             # print("embed NONE here")
    #             mention_embedding, _ = self.represent_input(self.xp.array([mention]), None)
    #             mention_embedding = mention_embedding[0]
    #         else:
    #             mention_inds = self._get_mention_inds(instance, mention, arg_id)
    #             mention_embedding = self._represent_mentions(mention_inds, token_rep)
    #         flattened_type_embedding = F.flatten(type_embedding)
    #         flattened_mention_embedding = F.flatten(mention_embedding)
    #         type_and_argument_embedding = F.hstack([flattened_type_embedding, flattened_mention_embedding])
    #         reshaped_type_and_argument_embedding = F.reshape(type_and_argument_embedding, (1, self.arg_type_and_word_dim))
    #
    #         embedding_list.append(reshaped_type_and_argument_embedding)
    #     return embedding_list


    # def _construct_relation_embedding(self, instance, trig_embedding, pair, entities_ids, triggers_ids, token_rep, arg_rel_pos_embed=None, structures_above_threshold=None):
    #     relation = pair[0]
    #     action = pair[1]
    #
    #     # flatten trigger
    #     a = F.flatten(trig_embedding)
    #
    #     # embed role type
    #     role = relation[0]
    #     role_type_embedding = self.embed_roletype(np.array([role]).astype("i"))
    #     b = F.flatten(role_type_embedding)
    #
    #     # embed argument
    #     arg = relation[1]
    #     is_trigger = arg in triggers_ids
    #     if is_trigger:
    #         arg_embedding = self._represent_type_and_argument(instance, triggers_ids, const.IDS_TRIGGERS_IDX, token_rep, arg,
    #                                                           structures_above_threshold)
    #     else:
    #         arg_embedding = self._represent_type_and_argument(instance, entities_ids, const.IDS_ENTITIES_IDX, token_rep, arg)
    #
    #     # embed action
    #     # if action == const.ACTION_NONE:
    #     #     action_embedding = Variable(self.xp.zeros((self.action_dim), dtype=np.float32))
    #     # else:
    #     action_embedding = self.embed_action(self.xp.array([action+1]).astype("i"))  # +1 moves the ACTION_NONE to 0 TODO must fix this adhoc addition here
    #     d = F.flatten(action_embedding)
    #
    #     # create relation embedding for each argument embedding
    #     relation_embedding = []
    #     if len(arg_embedding) != 0:
    #         for i in range(len(arg_embedding)):
    #
    #             c = F.flatten(arg_embedding[i])
    #
    #             #condition for relative position embedding
    #             if arg_rel_pos_embed is not None:
    #                 e = F.flatten(arg_rel_pos_embed)
    #                 z = F.hstack([a, b, c, d, e])
    #                 emb = F.reshape(z, (1, self.relation_dim))
    #             else:
    #                 z = F.hstack([a, b, c, d])
    #                 emb = F.reshape(z, (1, self.relation_dim))
    #             relation_embedding.append(emb)
    #     return relation_embedding

    # def represent_arg_rel_pos(self, pos_id):
    #     """Return an embedding representing the relative position id.
    #
    #     :param pos_id: the index relative to the trigger
    #     :return: relative position embedding
    #     """
    #     return self.embed_rel_positiontype(np.array([pos_id + self.max_pos]).astype("i"))  # add max_pos so that if id is negative, index will be max_pos-id and vv

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

    # def embed_arg_rel_pos(self, trigger, arg, instance):
    #     rel_pos = self.get_arg_rel_pos(trigger, arg, instance)
    #     arg_rel_pos_embed = self.represent_arg_rel_pos(rel_pos)
    #     return arg_rel_pos_embed


    # def _construct_structure_embeddings(self, instance, token_rep, trigger, trig_embedding, structure, entities_ids, triggers_ids, structures_above_threshold=None):
    #     structure_embedding = []
    #     for pair in structure:
    #         arg_rel_pos_embed = None
    #         # embed relative position embedding
    #         if self.args['add_rel_pos']:
    #             if pair[0][1] is const.ID_NONE_ARG_TYPE:
    #                 arg = trigger
    #             else:
    #                 arg = pair[0][1]
    #             arg_rel_pos_embed = self.embed_arg_rel_pos(trigger, arg, instance)
    #         embedding = self._construct_relation_embedding(instance, trig_embedding, pair, entities_ids, triggers_ids, token_rep, arg_rel_pos_embed, structures_above_threshold)
    #         structure_embedding.append(embedding)
    #     return structure_embedding

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
        if rel_agg_attn: #do attention
            structure_embed = F.reshape(F.sum(self.relation_attention(rel_lin), 0), (1, self.relation_dim))
        else: #sum by default
            temp2 = F.reshape(rel_lin, (structure.shape[0], structure.shape[1], -1))
            structure_embed = F.sum(temp2, axis=1)
        return structure_embed

    def predict_score(self, state):
        state_rep = self.state_representation(F.relu(state))
        h1 = F.relu(self.linear1(state_rep))
        h2 = F.relu(self.linear2(h1)) #TODO do i need this extra layer?
        raw_score = self.linear(h1)
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
        softmax = F.softmax(F.matmul(Q, F.transpose(K))/F.sqrt(self.xp.array([K.shape[1]]).astype('float32')))
        finalscore = F.matmul(softmax, V)

        return finalscore

    def perform_action(self, S, B):
        if self.args['struct_attn']:
            temp_sb = self.structure_attn(B, S[0])
            for i in range(len(S)-1):
                temp = self.structure_attn(B, S[i+1])
                temp_sb = F.vstack((temp_sb, temp))

        else: # relation aggregation or sum
            temp_b = self.represent_structure(B, self.linear_buffer, self.args['rel_agg_attn'])
            # temp_b = F.tile(temp_b, (len(S), 1)) #TODO why?? because of concat? but we can concat along different axis
            # if S.shape[1] > 1:
            #     print()
            temp_s = self.represent_structure(S, self.linear_structure, self.args['rel_agg_attn'])

            # for i in range(len(S) - 1):
            #     temp = self.represent_structure(S[i+1], self.linear_structure, self.args['rel_agg_attn'])
            #     temp_s = F.vstack((temp_s, temp))
            # try:
            temp_sb = F.concat((temp_s, temp_b), axis=1)
            # except:
            #     print()
        # flattened = F.flatten(temp_sb)

        raw_score, softmax_score, state_rep = self.predict_score(temp_sb)

        return raw_score, softmax_score, state_rep

    # def check_if_all_subevents_present(self, cur_buffer, cur_structure, trigger_subevent_embeddings):
    #     present = True
    #     for rel in cur_buffer:
    #         entry = rel[0]
    #         if type(entry) is tuple:  # not None
    #             arg_id = str(entry[1])
    #             if arg_id.startswith("TR"):
    #                 if arg_id not in trigger_subevent_embeddings:
    #                     present = False
    #                     log.debug("Sub event not present: %s", arg_id)
    #                     break
    #
    #     for rel in cur_structure:
    #         entry = rel[0]
    #         if type(entry) is tuple:  # not None
    #             arg_id = str(entry[1])
    #             if arg_id.startswith("TR"):
    #                 if arg_id not in trigger_subevent_embeddings:
    #                     present = False
    #                     log.debug("Sub event not present: %s", arg_id)
    #                     break
    #     return present

    # def extract_nbest_structures(self, nbest_states, states_dict, entry, true_entry, gold_actions=None):
    #     structures = []
    #     # loop on n-best
    #     for pqi, (_, state_ctr) in enumerate(nbest_states):
    #         _, cur_structure, _, prev_action, temp_target_label, _, _, combi = states_dict[state_ctr]
    #
    #         # loop thru actions, TODO process all actions once
    #         for action in const.ACTION_LIST:  # don't include none
    #             action_entry = (entry, action, true_entry)
    #             if entry[0] == const.ID_NONE_ROLE_TYPE and action == const.ACTION_ADD:
    #                 continue  # TODO skip this adding NONE does not make sense, only IGNORE or ADDFIX, can't this be not a special case?
    #
    #             # determine target label
    #             new_action = str(prev_action) + str(action)
    #             target_label = 0
    #             if chainer.config.train:
    #                 for g, gold_action in gold_actions.items():
    #                     str_action = ''.join(gold_action)
    #                     if str_action == new_action:
    #                         target_label = 1
    #                         break
    #
    #             # create the structure with the entry
    #             new_structure = cur_structure.copy()
    #             new_structure.append(action_entry)
    #
    #             structures.append((new_structure, target_label))
    #     return structures

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
        l1_emb = self.embed_level1_typegen(self.xp.array([l1]).astype("i"))

        # embed level 2 ids
        l2_emb = self.embed_level2_typegen(self.xp.array([l2]).astype("i"))

        return F.concat((type_emb, mention_emb, l1_emb, l2_emb), axis=2)

    def embed_relations(self, zip_ids, token_rep):

        # embed trigger
        trigger = self.emb_arg(zip_ids[0], zip_ids[1], zip_ids[2], zip_ids[3], token_rep)

        # embed role
        role_emb = self.embed_roletype(self.xp.array([zip_ids[4]]).astype("i"))

        # embed arg TODO or replace this with the event embedding if it's been predicted down the level
        # check if arg has been predicted previously, if so create as many events
        arg = self.emb_arg(zip_ids[5], zip_ids[6], zip_ids[7], zip_ids[8], token_rep)

        # embed rel pos ids
        rel_pos_emb = self.embed_rel_positiontype(self.xp.array([zip_ids[9]]).astype("i"))

        return F.concat((trigger, role_emb, arg, rel_pos_emb), axis=2)

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

        if len(edges) == 0:
            stacked_rel_embs = F.reshape(Variable(np.zeros((self.relation_dim-self.action_dim), dtype=np.float32)), (1,1,-1))
        else:
            # if len(edges) == 1:
            #     edges = [edges]

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
                    if index != -1: # replace the arg rep with the event rep
                        event = trigger_subevent_embeddings[arg_id][index]
                        trigger_and_role = relation_reps[i][0:self.arg_type_and_word_dim + self.dim_role_type]
                        rest = relation_reps[i][self.arg_type_and_word_dim + self.dim_role_type + self.arg_type_and_word_dim:]
                        rep = F.concat((trigger_and_role, event, rest), axis=0)
                    new_rel_embs.append(rep)
                stacked_new_rel_embs = F.stack(new_rel_embs)
                new_structure_embs.append(stacked_new_rel_embs)
            stacked_rel_embs = F.stack(new_structure_embs)
        return stacked_rel_embs

    def add_action_embeddings(self, prev_action, common_structure_emb, actions_emb):
        '''
        add action to all structures and in each structure the action added to each relation is based on the sequence of the previous action sequence
        :param prev_action: list of sequence of actions
        :param common_structure_emb: the list of structures which is a list of relations
        :param actions_emb: embeddings for all actions
        :return:
        '''
        action_seq_emb = F.embed_id(self.xp.array(prev_action), actions_emb) # go thru each action and add 1 to move index 1 up
        new_action = F.expand_dims(action_seq_emb, axis=0) # for common, no repeats as rels as actions equals rels
        if new_action.shape[1] > common_structure_emb.shape[1]: # for the arg, repeat the structure to num actions then reshape new_action
            num_structures = common_structure_emb.shape[0]
            num_actions = new_action.shape[1]
            common_structure_emb = F.repeat(common_structure_emb, num_actions, axis=0)
            new_action = F.repeat(new_action, num_structures, axis=0)
            new_action = F.reshape(new_action, (new_action.shape[0]*new_action.shape[1], 1, -1))
            # new_action = F.reshape(new_action, (new_action.shape[1], 1, -1))
        elif new_action.shape[1] < common_structure_emb.shape[1]: # for buffer, repeat the actions
            new_action = F.repeat(new_action, common_structure_emb.shape[1], axis=1)
        assert new_action.shape[1] == common_structure_emb.shape[1], log.debug("new_action.shape[1] != common_structure_emb.shape[1]")
        common_structure_wd_action_emb = F.concat((new_action, common_structure_emb), axis=2)
        return common_structure_wd_action_emb

    def extract_target_labels(self, prev_action, gold_actions):
        targets = []
        new_actions = []
        if prev_action == [-1]: # remove the first state and set it to empty
            prev_action = []
        base = prev_action[:]
        for action in const.ACTION_LIST:
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

    def profile_predict(self, instance, token_rep):
        profile.runctx('self._predict(instance, token_rep)', globals(), locals())

    def _predict(self, instance, token_rep):

        # def _entry_in_buffer(entry, cur_buffer):
        #     yes = False
        #     for c in cur_buffer:
        #         edge = c[0]
        #         if entry == edge:
        #             yes = True
        #             break
        #     return yes

        # def _remove_entry(cur_buffer, entry):
        #     new_buffer = []
        #     for c in cur_buffer:
        #         edge = c[0]
        #         if entry != edge:
        #             new_buffer.append(c)
        #     return new_buffer

        # def _extract_sub_event_predictions_combinations(structure, predictions, combi):
        #     '''
        #         There's no need to generate combinations for the buffer as only the structure
        #         gets to be added in the final structures
        #     '''
        #     combination = []
        #
        #     last_relation = structure[-1]
        #     lst = []
        #
        #     # add the previous structure
        #     if combi is None:  # first step in search
        #         lst.append((last_relation[0], 0))
        #
        #     else:
        #         # print("test")
        #         for c in combi:
        #             lst = []
        #             lst.append(c)
        #             # combination.append(lst)
        #             combination.append(lst)
        #
        #         lst = []
        #         # add the combination of the last relation
        #         arg = last_relation[0][1]
        #         action = last_relation[1]
        #         if action != const.ACTION_IGNORE:
        #             if type(last_relation[0]) is tuple:
        #                 if arg.startswith("TR") and arg in predictions:
        #                     preds = predictions[arg]
        #                     for p in range(len(preds)):
        #                         lst.append((arg, p))  # pth index
        #                 else:
        #                     lst.append((arg, 0))  # the representation at index 0
        #             else:  # might not be executed at all
        #                 lst.append((last_relation[0], 0))
        #         else:
        #             lst.append((arg, 0))
        #
        #     combination.append(lst)
        #
        #     final_list_of_combination = []
        #     for comb in itertools.product(*combination):
        #         l = list(comb)
        #         final_list_of_combination.append(l)
        #
        #     return final_list_of_combination

        # def extract_combination(sub_event_prediction_combination, structure_emb):
        #     len_s = len(structure_emb)
        #     assert len_s == len(sub_event_prediction_combination), 'Error'
        #     temp_structure_emb = []
        #
        #     for i in range(len_s):
        #         rel = sub_event_prediction_combination[i]
        #         temp_structure_emb.append(structure_emb[i][rel[1]])
        #
        #     return temp_structure_emb


        # generator = buffer_arg_generator(instance)
        # for i in generator:
        #     print(i)

        # triggers = []
        # triggers_structures_mapping = collections.OrderedDict()

        # gather triggers in bottom-up sequence
        levels = instance[const.PRED_CAND_STRUCTURES_IDX]
        entities_ids = instance[const.PRED_ENTITIES_IDX]
        triggers_ids = instance[const.PRED_TRIGGERS_IDX]

        # add id of NONE TODO not sure if this should be here
        entities_ids[const.NONE_ARG_TYPE] = [const.ID_NONE_ARG_TYPE, '', '', [const.ID_NONE_WORD_TYPE]]

        # for level in range(len(levels)):
        #     for trig, trigstructures in levels[level].items():
        #         triggers.append((trig, g))
        #         triggers_structures_mapping[trig] = trigstructures

        # sentence level variables for sentence-level learning
        trigger_subevent_embeddings = DefaultOrderedDict(list)
        final_event_structures = DefaultOrderedDict(list)
        instance_loss = 0.0

        early_update = False

        # debugging variable
        file_id = instance[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_FILE_ID_IDX]
        log.debug("file_id: %s", file_id)

        num_classification = 0

        # extract action embs from ALL actions including NONE, since same for the batch it can be retrieve once
        actions = [0] + [a + 1 for a in const.ACTION_LIST]
        actions_emb = self.embed_action(self.xp.array(actions).astype("i"))

        # loop through triggers in sequence thru levels
        # TODO process all triggers in the same level at once
        for level in levels:

            for trigger, trigstructures in level.items():

                if trigger in ['T22', 'T25', 'T28', 'T32']:
                    print()

                # -- extract info -- #
                log.debug("trigger: %s", trigger)
                trig_structure = trigstructures[0] #there's only one
                true_edges = trig_structure[0]
                edges = trig_structure[1]

                # -- extract ids and reps -- #
                # TODO can be done once for all triggers in this level
                # trigger ids
                trig_ids = self.arg2ids(trigger, triggers_ids, entities_ids, instance)
                # extract relation ids from edges
                relation_ids = self.convert2ids(instance, edges, entities_ids, triggers_ids, trig_ids, trigger)
                # extract relation embs
                zip_ids = list(zip(*[i for i in relation_ids]))
                relation_reps = F.reshape(self.embed_relations(zip_ids, token_rep), (len(relation_ids), -1))
                # combine ids and edges
                relations = list(zip(relation_ids, edges, true_edges))

                # -- initial state -- #
                pq = []
                states_dict = DefaultOrderedDict()
                structure = []
                ctr = 0
                # must have a value previous action for the add_action_embeddings to work
                states_dict[ctr] = (structure, None, [const.ACTION_NONE], -1, 0, 0, None)
                pq.append((0.0, ctr))
                ctr += 1

                # init gold actions
                gold_actions = None
                if chainer.config.train:
                    num_events = len(trig_structure[2])
                    gold_actions = collections.OrderedDict() # accumulates the gold actions thru timesteps

                # select n-best
                sorted_list = sorted(pq, key=lambda state: state[0], reverse=True)  # sort in descending order
                nbest_states = sorted_list[:self.n_best] if len(sorted_list) >= self.n_best else sorted_list[:]

                # loop through timesteps
                for timestep, (_, relation, actual_relation) in enumerate(relations):

                    # reset pq: every time step has a new pq where to select the nbest from,
                    # TODO the idea is that having one pq for all time steps may slow down the search but can also be tried
                    pq = []

                    # check if at the current time step the entry has sub-events which were not predicted and stop the search (break)
                    # TODO can be changed to continue to next timestep as next argument may have been predicted down the level but too many things to consider below if so
                    arg_id = relation[1]
                    if arg_id.startswith("TR"):
                        if arg_id not in trigger_subevent_embeddings:
                            log.debug("arg id not predicted in previous level: %s, %s, %s", arg_id, trigger, file_id)
                            break

                    # extract gold labels
                    if chainer.config.train:
                        for l in range(num_events):
                            label = trig_structure[2][l][timestep][1]
                            label_action = label.index(1)
                            if l in gold_actions:
                                temp = gold_actions[l]
                                temp.append(label_action)
                                gold_actions[l] = temp
                            else:
                                gold_actions[l] = [label_action]

                    # -- extract buffer emb and add action-- #
                    # NOTE no need to create combinations based on the predicted argument events since this just serves as context and no final structure will be created from this
                    buffer_emb = self.extract_arg_embeddings(relations[timestep+1:], relation_reps[timestep+1:])
                    # buffer_emb = self.extract_arg_embeddings(relations[:], relation_reps[:])
                    buffer_wd_action_emb = self.add_action_embeddings([const.ACTION_NONE+1], buffer_emb, actions_emb)

                    # --- extract the common structure emb -- #
                    # can produce more structures but num relations in each will be the same up to before this time step
                    common_structure_emb = self.extract_arg_embeddings(relations[:timestep], relation_reps[:timestep], trigger_subevent_embeddings)
                    # common_structure_emb = self.extract_arg_embeddings(relations[:], relation_reps[:], trigger_subevent_embeddings)

                    # -- construct the current relation with actions -- #
                    # can produce more structures but relations will always be one
                    # TODO need to confirm is subscripting it will not destroy the computational graph
                    args_structure_emb = self.extract_arg_embeddings([relations[timestep]], F.reshape(relation_reps[timestep], (1,-1)), trigger_subevent_embeddings)
                    # args_structure_emb = self.extract_arg_embeddings([relations[1]], F.reshape(relation_reps[1], (1, -1)), trigger_subevent_embeddings)
                    args_structure_wd_actions_emb = self.add_action_embeddings(const.ACTION_LIST, args_structure_emb, actions_emb)

                    # loop thru the nbest states
                    for pqi, (_, state_ctr) in enumerate (nbest_states):
                        cur_structure, _, prev_action, temp_target_label, _, _, combi = states_dict[state_ctr]

                        # -- add prev actions to common structure emb -- #
                        common_structure_wd_actions_emb = self.add_action_embeddings([(i+1) for i in prev_action], common_structure_emb, actions_emb)

                        # determine target labels for all actions for this state
                        target_labels, new_actions = self.extract_target_labels(prev_action, gold_actions)

                        # concat the common and arg to be the structure
                        new_common_structure_wd_actions_emb = F.repeat(common_structure_wd_actions_emb, args_structure_wd_actions_emb.shape[0], axis=0)
                        new_args_structure_wd_actions_emb = F.repeat(args_structure_wd_actions_emb, common_structure_wd_actions_emb.shape[0], axis=0)
                        # try:
                        structures_emb = F.concat((new_common_structure_wd_actions_emb, new_args_structure_wd_actions_emb), axis=1)
                        # except:
                        #     print()

                        # score the buffer + structure, final result will depend on the length of structure
                        new_buffer_wd_actions_emb = F.repeat(buffer_wd_action_emb, structures_emb.shape[0], axis=0)
                        raw_scores, softmax_scores, state_reps = self.perform_action(structures_emb, new_buffer_wd_actions_emb)

                        # create the new structures with the entry
                        new_structures = []
                        for action in const.ACTION_LIST:
                            action_entry = (relation, action, actual_relation)
                            temp_s = list(cur_structure.copy())
                            temp_s.append(action_entry)
                            new_structures.append(temp_s)

                        # push scores to with target labels and new actions
                        for k in range(len(new_structures)):
                            # push its respective info to the pq
                            if chainer.config.train:
                                states_dict[ctr] = (new_structures[k], state_reps[k], new_actions[k], target_labels[k], raw_scores[k], softmax_scores[k], structures_emb[k])
                            else:
                                states_dict[ctr] = (new_structures[k], state_reps[k], new_actions[k], -1, raw_scores[k], softmax_scores[k], structures_emb[k])

                            # push the score with the state primary key (ctr)
                            pq.append((softmax_scores[k].data[1], ctr))

                            # add ctr for the state index
                            ctr += 1



                        # loop thru actions, TODO process all actions once
                        # for action in const.ACTION_LIST: #don't include none
                        #     action_entry = (relation, action, actual_relation)
                        #     if relation[0] == const.ID_NONE_ROLE_TYPE and action == const.ACTION_ADD:
                        #         continue  # TODO skip this adding NONE does not make sense, only IGNORE or ADDFIX, can't this be not a special case?
                        #
                        #     # # determine target label
                        #     # new_action = prev_action.append(action)
                        #     # target_label = 0
                        #     # if chainer.config.train:
                        #     #     for g, gold_action in gold_actions.items():
                        #     #         if gold_action == new_action:
                        #     #             target_label = 1
                        #     #             break
                        #
                        #     # create the action structure as many args
                        #     action_emb = F.reshape(actions_emb[action + 1], (1, -1))
                        #     repeated_action_emb = F.repeat(action_emb, args_structure_emb.shape[0], 0)
                        #     args_structure_wd_action_emb = F.concat((args_structure_emb, repeated_action_emb), axis=1) #TODO how to concat if len of structure emb is not equal to common_structure_emb
                        #
                        #     # create the structure with the entry
                        #     new_structure = cur_structure.copy()
                        #     new_structure.append(action_entry)
                        #
                        #     # concat common and the args_action_structure
                        #     new_structures_emb = F.concat((common_structure_wd_actions_emb, args_structure_wd_action_emb), axis=0) #TODO how to concat here if len structure emb is greater than 1
                        #
                        #     # score each structure (and combination) and buffer
                        #     raw_scores, softmax_scores, state_reps = self.perform_action(new_structures_emb, buffer_wd_action_emb) #[timestep+1:]
                        #
                        #     # debugging info
                        #     num_classification += 1
                        #
                        #     # loop thru the sub event prediction combination and
                        #     for k in range(softmax_scores.shape[0]):
                        #
                        #         # push its respective info to the pq
                        #         if chainer.config.train:
                        #             states_dict[ctr] = (new_structure, state_reps[k], new_action, target_label, raw_scores[k], softmax_scores[k], new_structure_emb[k])
                        #         else:
                        #             states_dict[ctr] = (new_structure, state_reps[k], new_action, -1, raw_scores[k], softmax_scores[k], new_structure_emb[k])
                        #
                        #         # push the score with the state primary key (ctr)
                        #         pq.append((softmax_scores[k].data[1], ctr))
                        #
                        #         # add ctr for the state index
                        #         ctr += 1

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
                    sorted_list = sorted(pq, key=lambda state:state[0], reverse=True) # sort in descending order
                    nbest_states = sorted_list[:self.n_best] if len(sorted_list) >= self.n_best else sorted_list[:]

                    # fix the event if it is greater than threshold
                    if trigger == 'TR29':
                        print()
                    to_be_removed = []
                    for n in range(len(nbest_states)):
                        _, temp_state_ctr = nbest_states[n]
                        temp_structure, temp_state_rep, temp_action, _, raw_score, softmax_score, combi = states_dict[temp_state_ctr]

                        # found an event, event before the last edge
                        if temp_action[-1] == const.ACTION_ADDFIX:
                            # always remove addfix from beam
                            to_be_removed.append(temp_state_ctr)

                            if softmax_score.data[1] >= self.threshold:
                                trigger_subevent_embeddings[trigger].append(temp_state_rep)
                                final_event_structures[trigger].append([temp_structure, raw_score])
                        if temp_action == [1]: # remove adding NONE in the first timestep
                            to_be_removed.append(temp_state_ctr)

                    # check which states to update #TODO should the early update be on validation as well??
                    if chainer.config.train:
                        # check 1: if gold is out of beam: gold but not nbest, early update
                        ## idea: score is closer to 0 but expected is 1 so move score closer to 1
                        to_be_updated = collections.OrderedDict()
                        for n in range(len(pq)):  # loop through all states
                            _, temp_state_ctr = pq[n]
                            _, _, temp_action, temp_target_label, raw_score, _, _ = states_dict[temp_state_ctr]
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
                            _, _, temp_action, temp_target_label, raw_score, softmax_score, _ = states_dict[temp_state_ctr]

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
                            _, _, _, temp_target_label, orig_score, _, _ = states_dict[key]
                            # value_label = value[1]
                            # assert value_label == temp_target_label, "Error: values not equal"
                            # new_target = np.array([temp_target_label], 'i')
                            orig_scores.append(orig_score)
                            targets.append(temp_target_label)

                        loss = F.softmax_cross_entropy(F.stack(orig_scores), self.xp.array(targets))
                        instance_loss += loss

                    # log.debug("trigger: timestep: entry: pq: nbest: to_remove: final_nbest: %s : %s : %s : %s: %s: %s : %s", trigger, t, true_entry, len(pq), len(nbest_states), len(to_be_removed), len(nbest_states)-len(to_be_removed))

                    # -- remove fixed events from pq  -- #
                    ## idea: they won't be expanded anymore
                    for item in to_be_removed:
                        for i, (_, temp_state_ctr) in enumerate(nbest_states):
                            if item == temp_state_ctr:
                                del nbest_states[i]
                                break


                    # if a gold is out of the beam, don't proceed to next timestep/argument
                    if chainer.config.train:
                        if self.args['early_update']:
                            if early_update:
                                break

        # predictions = final_event_structures
        return final_event_structures, instance_loss, num_classification

    def __call__(self, batch):
        batch_predictions = []


        #prepare list of list of word ids and positions
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

        for i in range(len(batch)):# sentences
            instance = batch[i]
            input_rep = self.decide_input_rep(bilstm_batch, pos_embed, i)

            if chainer.config.train:
                predictions, instance_loss, _ = self._predict(instance, input_rep)

                # # method profiling
                # self.profile_predict(instance, input_rep)
                # break

                batch_loss += instance_loss
            else:
                predictions, _, count = self._predict(instance, input_rep)
                batch_cnt += count
            batch_predictions.append(predictions)


        return batch_predictions, batch_loss, batch_cnt
