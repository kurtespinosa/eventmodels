#
import os
os.environ["CHAINER_SEED"]="0"
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
log = logging.getLogger(__name__)

# 3rd party
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

# modules
import src.util.constants as const

class SearchBasedModel(chainer.Chain):
    def __init__(self, n_word_types,n_trig_types, n_role_types, n_entity_types, params):
        super(SearchBasedModel, self).__init__()
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

            self.arg_type_and_word_dim = self.dim_arg_type + (self.dim_bilstm * 2) + self.dim_pos

            self.relation_dim = self.arg_type_and_word_dim + self.dim_role_type + self.arg_type_and_word_dim
            self.hidden_dim = int(self.arg_type_and_word_dim / 2)

            embed_init = chainer.initializers.Normal()
            self.embed_positiontype = L.EmbedID(self.max_pos, self.dim_pos, initialW=embed_init, ignore_label=-1)
            self.embed_wordtype = L.EmbedID(n_word_types, self.dim_embed, initialW=embed_init, ignore_label=-1)
            self.embed_trigtype = L.EmbedID(n_trig_types, self.dim_arg_type, initialW=embed_init, ignore_label=-1)
            self.embed_roletype = L.EmbedID(n_role_types, self.dim_role_type, initialW=embed_init, ignore_label=-1)
            self.embed_enttype = L.EmbedID(n_entity_types, self.dim_arg_type, initialW=embed_init, ignore_label=-1)
            self.embed_action = L.EmbedID(self.action_dim, self.action_dim, initialW=embed_init, ignore_label=-1)

            self.bilstm = L.NStepBiLSTM(1, self.dim_embed, self.dim_bilstm, 0)

            self.linear_structure = L.Linear(None, self.relation_dim+self.action_dim)
            self.linear_buffer = L.Linear(None, self.relation_dim+self.action_dim)

            self.state_representation = L.Linear(None, self.arg_type_and_word_dim)
            self.linear1 = L.Linear(None, self.hidden_dim)
            self.linear2 = L.Linear(None, self.hidden_dim)
            self.linear = L.Linear(None, 1)

    def load_glove(self, path, vocab):
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed_wordtype.W.data[vocab[word]] = vec
        log.info("Loaded embeddings: %s", path)

    def sequence_embed(self, xs, embed, TRAIN):
        """
        Embeds the sequences (batch).
        Args:
            xs: list of arrays of ids
            embed: embedding link
            TRAIN: flag for dropout which is applied during training but not in testing

        Returns: the embedded sequence
        """
        xs_len = [len(x) for x in xs]
        xs_section = self.xp.cumsum(xs_len[:-1])
        xs_em = embed(F.concat(xs, axis=0))
        if TRAIN:
            xs_em = F.dropout(xs_em, ratio=self.dropout)
        xs_embed = F.split_axis(xs_em, xs_section, axis=0)
        return xs_embed

    def _construct_structure_embeddings(self, instance, token_rep, trigger, structure, entities_ids, triggers_ids, structures_above_threshold=None):
        def _represent_mentions(mention_ids, token_rep):
            '''

            :param mention_ids:
            :param token_rep:
            :return:
            '''
            try:
                if len(mention_ids) == 0:
                    final_mention_representation = F.embed_id(self.xp.asarray([0]).astype('i'), token_rep)
                else:
                    final_mention_representation = F.embed_id(self.xp.asarray(mention_ids).astype('i'), token_rep)
                    if len(mention_ids) > 1:
                        final_mention_representation = F.average(final_mention_representation, axis=0)
            except:
                #TODO: when word is a substring of a word or when word is the NONE, can be split
                '''
                    #TODO:
                        If reason is because the word is a substring, use the index of the enclosing word. In this case, position is the index.
                        If reason is because the word is a NONE word, use the index of the NONE token. In this case, position is the last index.
                        
                        # xs_embed = self.sequence_embed([self.xp.asarray([3]).astype('i')], self.embed_wordtype, TRAIN=False) #index 3 is the NONE word
                        # h, c, word_embedding = self.bilstm(None, None, xs_embed)
                        # none_position_embedding = self.embed_positiontype(self.xp.asarray([self.max_pos - 1]))  # TODO: current is strict, better way is to get the position where it is a subset
                        # final_mention_representation = F.concat((word_embedding[0], none_position_embedding), axis=1)
                '''
                final_mention_representation = F.embed_id(self.xp.asarray([0]).astype('i'), token_rep)
            return final_mention_representation

        def _get_word_ids(xsi, mention):
            '''Returns the indices of the mention within the sentence.'''
            word_ind = []
            sentence_ids = xsi[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]
            for i in mention:
                if i in sentence_ids:
                    ind = sentence_ids.index(i)
                    word_ind.append(ind)
            return word_ind

        def _represent_argument_rel_pos(pos_id):
            """Return an embedding representing the relative position id.

            :param pos_id: the index relative to the trigger
            :return: relative position embedding
            """
            rel_pos_emb = None

            return rel_pos_emb

        def _compute_rel_pos(trig_idx, arg_idx):
            '''
            Computes the relative position of the argument wrt to the trig index.

            :param trig_idx: index of the trigger
            :param arg_idx: index of the argument
            :return: relative position
            '''

            rel_pos = -1
            return rel_pos

        def _represent_type_and_argument(arg_ids, type_index, token_rep, type_label, structures_above_threshold=None):

            embedding_list = []
            if structures_above_threshold is not None:
                embedding_list = structures_above_threshold[type_label]
            else:
                defn = arg_ids[type_label]
                type_id = defn[const.IDS_ARG_TYPE]
                type_embedding = None
                if type_index == const.IDS_TRIGGERS_IDX:
                    type_embedding = self.embed_trigtype(np.array([type_id]).astype("i"))
                elif type_index == const.IDS_ENTITIES_IDX:
                    type_embedding = self.embed_enttype(np.array([type_id]).astype("i"))
                mention = defn[const.IDS_ARG_MENTION]
                mention_ids = _get_word_ids(instance, mention)

                mention_embedding = _represent_mentions(mention_ids, token_rep)
                flattened_type_embedding = F.flatten(type_embedding)
                flattened_mention_embedding = F.flatten(mention_embedding)
                type_and_argument_embedding = F.hstack([flattened_type_embedding, flattened_mention_embedding])

                reshaped_type_and_argument_embedding = F.reshape(type_and_argument_embedding, (1, self.arg_type_and_word_dim))

                embedding_list.append(reshaped_type_and_argument_embedding)
            return embedding_list

        def _construct_relation_embedding(trig_embedding, pair, entities_ids, triggers_ids, token_rep, structures_above_threshold=None):
            relation = pair[0]
            action = pair[1]
            if action == const.ACTION_NONE:
                action_embedding = Variable(np.zeros((self.action_dim), dtype=np.float32))
            else:
                action_embedding = self.embed_action(np.array([action]).astype("i"))

            if relation == const.NONE_ROLE_TYPE:
                role = 0
                type_id = 0

                role_type_embedding = self.embed_roletype(np.array([role]).astype("i"))
                type_embedding = self.embed_enttype(np.array([type_id]).astype("i"))

                mention_ids = [self.max_pos-1] #NONE position index
                mention_embedding = _represent_mentions(mention_ids, token_rep)

                flattened_type_embedding = F.flatten(type_embedding)
                flattened_mention_embedding = F.flatten(mention_embedding)
                type_and_argument_embedding = F.hstack([flattened_type_embedding, flattened_mention_embedding])

                arg_embedding = F.reshape(type_and_argument_embedding,(1, self.arg_type_and_word_dim))
                relation_embedding = []
                a = F.flatten(trig_embedding)
                b = F.flatten(role_type_embedding)
                c = F.flatten(arg_embedding)
                d = F.flatten(action_embedding)
                z = F.hstack([a, b, c, d])
                emb = F.reshape(z, (1, self.relation_dim+self.action_dim))
                relation_embedding.append(emb)
            else:
                role = relation[0]
                arg = relation[1]
                role_type_embedding = self.embed_roletype(np.array([role]).astype("i"))
                is_trigger = arg in triggers_ids

                if is_trigger:
                    arg_embedding = _represent_type_and_argument(triggers_ids, const.IDS_TRIGGERS_IDX, token_rep, arg,
                                                                 structures_above_threshold)
                else:
                    arg_embedding = _represent_type_and_argument(entities_ids, const.IDS_ENTITIES_IDX, token_rep, arg)
                relation_embedding = []
                if len(arg_embedding) != 0:
                    for i in range(len(arg_embedding)):
                        a = F.flatten(trig_embedding)
                        b = F.flatten(role_type_embedding)
                        c = F.flatten(arg_embedding[i])
                        d = F.flatten(action_embedding)
                        z = F.hstack([a, b, c, d])
                        emb = F.reshape(z, (1, self.relation_dim+self.action_dim))
                        relation_embedding.append(emb)
            return relation_embedding

        trig_embedding = _represent_type_and_argument(triggers_ids, const.IDS_TRIGGERS_IDX, token_rep, trigger)[0]
        structure_embedding = []
        for pair in structure:
            embedding = _construct_relation_embedding(trig_embedding, pair, entities_ids, triggers_ids, token_rep, structures_above_threshold)
            structure_embedding.append(embedding)
        return structure_embedding

    def perform_action(self, S, B):
        S_ = np.zeros((self.relation_dim + self.action_dim), dtype=np.float32)
        final_b = np.zeros((self.relation_dim + self.action_dim), dtype=np.float32)

        #get reduced B representation
        if len(B) > 0:
            stack = F.vstack(B[0])
            for i in range(len(B)-1):
                stack = F.vstack((stack, F.vstack(B[i+1])))
            linear_b = F.relu(self.linear_buffer(stack))
            sum_b = F.sum(linear_b, 0)
            final_b = F.reshape(sum_b, (1, self.relation_dim+self.action_dim))

        #replicate B len(S) times
        B_ = F.tile(final_b, (len(S), 1))

        #compose len(s) S
        if len(S) > 0:
            S_ = F.reshape(F.sum(F.relu(self.linear_structure(F.vstack(S[0]))),0), (1, self.relation_dim+self.action_dim))
            for i in range(len(S)-1):
                temp = F.reshape(F.sum(F.relu(self.linear_structure(F.vstack(S[i+1]))),0), (1, self.relation_dim+self.action_dim))
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
    def _predict(self, instance, token_rep, TRAIN=False):

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
                There's no need to generate combinations for the buffer as only the structure
                gets to be added in the final structures
            '''
            combination = []

            last_relation = structure[-1]
            lst = []

            # add the previous structure
            if combi is None: #first step in search
                lst.append((last_relation[0], 0))

            else:
                # print("test")
                for c in combi:
                    lst = []
                    lst.append(c)
                    # combination.append(lst)
                    combination.append(lst)

                lst = []
                #add the combination of the last relation
                arg = last_relation[0][1]
                action = last_relation[1]
                if action != const.ACTION_IGNORE:
                    if type(last_relation[0]) is tuple:
                        if arg.startswith("TR") and arg in predictions:
                            preds = predictions[arg]
                            for p in range(len(preds)):
                                lst.append((arg, p)) # pth index
                        else:
                            lst.append((arg, 0)) # the representation at index 0
                    else: #might not be executed at all
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
                    print("analysis e_c")
                temp_structure_emb.append(structure_emb[i][rel[1]])

            return temp_structure_emb

        triggers = []
        action_list = [const.ACTION_IGNORE,const.ACTION_ADD,const.ACTION_ADDFIX]
        triggers_structures_mapping = dict()

        #gather triggers in bottom-up sequence
        groups = instance[const.PRED_CAND_STRUCTURES_IDX]
        entities_ids = instance[const.PRED_ENTITIES_IDX]
        triggers_ids = instance[const.PRED_TRIGGERS_IDX]
        for g in range(len(groups)):
            for trig, trigstructures in groups[g].items():
                triggers.insert(0, (trig, g))
                triggers_structures_mapping[trig] = trigstructures

        #sentence level variables for sentence-level learning
        ctr = 0
        states_dict = {}
        trigger_subevent_embeddings = collections.defaultdict(list)
        final_event_structures = collections.defaultdict(list)
        instance_loss = 0.0

        early_update = False

        num_classification = 0

        while triggers:
            trigger, level = triggers.pop()
            for s in range(len(triggers_structures_mapping[trigger])):
                #track n-best states
                pq = []

                edges = triggers_structures_mapping[trigger][s][1]
                true_edges = triggers_structures_mapping[trigger][s][0]

                #initial state
                structure = []
                buffer = []
                for e in range(len(edges)):
                    buffer.insert(0,(edges[e], const.ACTION_NONE, true_edges[e]))
                list_buffer = copy.deepcopy(buffer)
                new_buffer = copy.deepcopy(buffer)
                new_structure = copy.deepcopy(structure)
                states_dict[ctr] = (new_buffer[:], new_structure[:], None, '', -1, 0, 0, None)
                heapq.heappush(pq, (0.0, ctr))
                ctr += 1

                if TRAIN:
                    num_events = len(triggers_structures_mapping[trigger][s][2])
                    gold_actions = dict()

                for i in range(len(list_buffer)):
                    entry, _, true_entry = list_buffer.pop()

                    #check if entry is a sub-event
                    arg_id = None
                    if type(entry) is tuple: #not None
                        arg_id = entry[1]
                        if arg_id.startswith("TR"):
                            if arg_id not in trigger_subevent_embeddings:
                                early_update = True
                                break #can't perform the search since the argument event has not been detected

                    #extract gold labels
                    if TRAIN:
                        for l in range(num_events):
                            label = triggers_structures_mapping[trigger][s][2][l][i][1]
                            label_action = label.index(1)
                            if l in gold_actions:
                                prev_gold_action = gold_actions[l]
                                new_gold_action = ''.join(prev_gold_action)+str(label_action)
                                gold_actions[l] = new_gold_action
                            else:
                                gold_actions[l] = str(label_action)

                    new_pq = []

                    for pqi in range(len(pq)):
                        _, state_ctr = heapq.heappop(pq)
                        cur_buffer, cur_structure, _, prev_action, temp_target_label, _, _, combi = states_dict[state_ctr]

                        if not _entry_in_buffer(entry, cur_buffer): #check if edge in buffer if not skip this state
                            continue

                        cur_buffer = _remove_entry(cur_buffer, entry)
                        new_buffer = copy.deepcopy(cur_buffer)

                        # TODO check relations only that are found in the gold druin
                        if not self.check_if_all_subevents_present(cur_buffer, cur_structure, trigger_subevent_embeddings):
                            early_update = True
                            break

                        buffer_emb = self._construct_structure_embeddings(instance, token_rep, trigger, new_buffer,
                                                                          entities_ids, triggers_ids,
                                                                          trigger_subevent_embeddings)

                        new_structure = copy.deepcopy(cur_structure)
                        # process the structure_emb before action is added
                        common_structure_emb = self._construct_structure_embeddings(instance, token_rep, trigger, new_structure,
                                                                             entities_ids, triggers_ids,
                                                                             trigger_subevent_embeddings)

                        for a in range(len(action_list)):
                            structure_emb = []
                            new_structure = []
                            if entry == const.NONE_ROLE_TYPE and a == const.ACTION_ADD:
                                continue #skip this adding NONE does not make sense, only IGNORE or ADDFIX

                            new_structure.append((entry, action_list[a], true_entry))

                            #construct the relation embeddings
                            #TODO process all actions once
                            temp_emb = self._construct_structure_embeddings(instance, token_rep, trigger, new_structure, entities_ids, triggers_ids, trigger_subevent_embeddings)

                            # construct new action
                            new_action = str(prev_action) + str(a)

                            # get gold action
                            target_label = 0
                            if TRAIN:
                                for g, gold_action in gold_actions.items():
                                    str_action = ''.join(gold_action)
                                    if str_action == new_action:
                                        target_label = 1
                                        break

                            #construct the structure and the structure_emb
                            assert len(temp_emb) == 1, "ERROR: Cannot have more than one entry."
                            # if len(temp_emb) > 1:
                            #     print("analysis")
                            structure_emb.extend(common_structure_emb)
                            structure_emb.append(temp_emb[0])

                            #create the structure
                            new_structure = copy.deepcopy(cur_structure)
                            new_structure.append((entry, action_list[a], true_entry))

                            #score the action and push resulting states to PQ
                            sub_event_predictions_combinations = _extract_sub_event_predictions_combinations(
                                new_structure, trigger_subevent_embeddings, combi)

                            temp_structure_embs = []
                            for sub_event_prediction_combination in sub_event_predictions_combinations:
                                temp_structure_emb = extract_combination(sub_event_prediction_combination, structure_emb)
                                temp_structure_embs.append(temp_structure_emb)


                            raw_scores, sigmoid_scores, state_reps = self.perform_action(temp_structure_embs, buffer_emb)
                            num_classification += 1

                            for k in range(len(sub_event_predictions_combinations)):
                                #push into PQ resulting state with the score
                                if TRAIN:
                                    states_dict[ctr] = (new_buffer[:], new_structure[:], state_reps[k], new_action, target_label, raw_scores[k], sigmoid_scores[k], sub_event_predictions_combinations[k])
                                else:
                                    states_dict[ctr] = (new_buffer[:], new_structure[:], state_reps[k], new_action, -1, raw_scores[k], sigmoid_scores[k], sub_event_predictions_combinations[k])

                                new_score = 1.0- sigmoid_scores[k].data[0] #because min pq is used
                                heapq.heappush(new_pq, (new_score, ctr))
                                ctr += 1

                    # add margin to all non-gold actions
                    if TRAIN:
                        new_pq_with_margin = []
                        for k in range(len(new_pq)):
                            temp_score, temp_state_ctr = new_pq[k]
                            temp_buffer, temp_structure, temp_state_rep, temp_new_action, target_label, raw_score, sigmoid_score, combi = states_dict[temp_state_ctr]
                            if target_label == 0: #non-gold
                                before = sigmoid_score
                                sigmoid_score = sigmoid_score + self.margin
                                temp_score = temp_score - self.margin
                                states_dict[temp_state_ctr] = (temp_buffer, temp_structure, temp_state_rep, temp_new_action, target_label, raw_score, sigmoid_score, combi)
                                new_pq_with_margin.append((temp_score, temp_state_ctr))
                            else:
                                new_pq_with_margin.append((temp_score, temp_state_ctr))
                        new_pq = []
                        for n in range(len(new_pq_with_margin)):
                            temp_score, temp_state_ctr = new_pq_with_margin[n]
                            heapq.heappush(new_pq, (temp_score, temp_state_ctr))

                    #create new pq copy
                    new_pq_copy = []
                    for k in range(len(new_pq)):
                        new_pq_copy.append(new_pq[k])
                    # set the PQ size to N
                    nbest_pq = []
                    count = 0
                    if len(new_pq) > 0:
                        for _ in range(self.n_best):
                            temp_score, temp_state_ctr = heapq.heappop(new_pq)
                            heapq.heappush(nbest_pq, (temp_score, temp_state_ctr))
                            count += 1
                            if count == self.n_best:  # number of items in heap is greater than or equal to N
                                break
                            if not new_pq:  # pq is already empty, number of items in heap is less than N
                                break

                    # fix the event if it is greater than threshold
                    to_be_removed = []
                    for n in range(len(nbest_pq)):
                        _, temp_state_ctr = nbest_pq[n]
                        _, temp_structure, temp_state_rep, temp_action, _, raw_score, sigmoid_score, combi = states_dict[temp_state_ctr]

                        if temp_action.endswith(str(const.ACTION_ADDFIX)):  # found an event, event before the last edge
                            to_be_removed.append(temp_state_ctr)  # always remove addfix from beam
                            if sigmoid_score.data[0] >= self.threshold:
                                trigger_subevent_embeddings[trigger].append(temp_state_rep)
                                final_event_structures[trigger].append([temp_structure, raw_score])

                    if TRAIN:
                        #early update
                        # check 1: if gold is out of beam
                        to_be_updated = dict()
                        for n in range(len(new_pq_copy)):
                            _, temp_state_ctr = new_pq_copy[n]
                            _, _, _, temp_action, temp_target_label, raw_score, _, _ = states_dict[temp_state_ctr]
                            gold_predicted = False
                            if temp_target_label == 1: #it is gold
                                for q in range(len(nbest_pq)):
                                    _, temp_state_ctr_2 = nbest_pq[q]
                                    if temp_state_ctr == temp_state_ctr_2: #gold was predicted
                                        gold_predicted = True
                                        break
                                if not gold_predicted:
                                    early_update = True
                                    to_be_updated[temp_state_ctr]= (raw_score, temp_target_label, temp_action)

                        #check 2:
                        for n in range(len(nbest_pq)):
                            _, temp_state_ctr = nbest_pq[n]
                            _, _, _, temp_action, temp_target_label, raw_score, sigmoid_score, _ = states_dict[temp_state_ctr]
                            # predicted action is not gold,
                            if temp_target_label == 0: #not gold
                                to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)
                            #predicted, gold but less than threshold
                            if temp_action.endswith(str(const.ACTION_ADDFIX)) and temp_target_label == 1:
                                if sigmoid_score.data[0] < self.threshold:
                                    to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)
                                else:
                                    to_be_updated[temp_state_ctr] = (raw_score, temp_target_label, temp_action)

                        # compute the loss and update
                        for key, value in to_be_updated.items():
                            _, _, _, _, temp_target_label, orig_score, _, _ = states_dict[key]
                            value_score = value[0]
                            value_label = value[1]

                            # assert value_score.data[0][0] == orig_score.data[0][0], "Error: values not equal"
                            assert value_label == temp_target_label, "Error: values not equal"
                            new_target = np.array([[temp_target_label]], 'i')
                            loss = F.sigmoid_cross_entropy(F.reshape(orig_score, (1,1)), new_target) #use un-normalised probability
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


                    if early_update:
                        break

                if early_update:
                    break

            if early_update:
                break

        predictions = final_event_structures
        return predictions, instance_loss, num_classification

    def __call__(self, batch, TRAIN=False):
        batch_predictions = []

        # prepare list of list of word ids and positions
        xs = []
        pos = []
        for i in batch:
            xs.append(self.xp.array(i[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]))
            pos.append(self.xp.array([(j) for j in range(len(i[0][0]))]))

        # embed
        xs_embed = self.sequence_embed(xs, self.embed_wordtype, TRAIN)
        pos_embed = self.sequence_embed(pos, self.embed_positiontype, TRAIN)

        # bilstm
        h, c, bilstm_batch = self.bilstm(None, None, xs_embed)

        batch_loss = 0
        batch_cnt = 0

        for i in range(len(batch)):
            instance = batch[i]
            word_and_pos_rep = F.concat((bilstm_batch[i], pos_embed[i]), axis=1)
            if TRAIN:
                predictions, instance_loss, _ = self._predict(instance, word_and_pos_rep, TRAIN)
                batch_loss += instance_loss
            else:
                predictions, _, count = self._predict(instance, word_and_pos_rep)
                batch_cnt += count
            batch_predictions.append(predictions)
        return batch_predictions, batch_loss, batch_cnt
