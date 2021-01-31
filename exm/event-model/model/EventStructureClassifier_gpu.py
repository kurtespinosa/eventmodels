# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

import itertools
import collections

from datetime import datetime

import constants as const
from pipeline.util import Util

class Loader(chainer.Chain):
    def __init__(self):
        pass

    def load_glove(self, path, vocab):
        print("Loading embeddings...")
        with open(path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec
        print("Finished loading embeddings.")


class EventStructureClassifier(Loader):
    def __init__(self, n_word_types, n_trig_types, n_role_types, n_entity_types, trigger_type2id, entity_type2id,
                 DIM_EMBED, DIM_EVENT, DIM_BILSTM, DIM_TRIG_TYPE, DIM_ROLE_TYPE, DIM_ARG_TYPE, DIM_IO, DROPOUT,
                 REPLACE_TYPE, GENERALISATION, THRESHOLD):
        super(Loader, self).__init__()
        with self.init_scope():
            self.DIM_EMBED = DIM_EMBED
            self.DIM_EVENT = DIM_EVENT
            self.DIM_BILSTM = DIM_BILSTM
            self.DIM_TRIG_TYPE = DIM_TRIG_TYPE
            self.DIM_ROLE_TYPE = DIM_ROLE_TYPE
            self.DIM_ARG_TYPE = DIM_ARG_TYPE
            self.DIM_IO = DIM_IO
            self.DROPOUT = DROPOUT
            self.THRESHOLD = THRESHOLD
            self.GENERALISATION = GENERALISATION
            self.REPLACE_TYPE = REPLACE_TYPE
            self.DROPOUT = DROPOUT


            self.DIM_TREE_LSTM_INPUT = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2) + self.DIM_ROLE_TYPE + self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2) + self.DIM_IO
            self.DIM_ARG = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2)

            self.id2triggertype = {v: k for k, v in trigger_type2id.items()}
            self.id2entitytype = {v: k for k, v in entity_type2id.items()}

            self.embed = L.EmbedID(n_word_types, self.DIM_EMBED, ignore_label=-1)
            self.bilstm = L.NStepBiLSTM(1, self.DIM_EMBED, self.DIM_BILSTM, 0)
            self.embed_trigtype = L.EmbedID(n_trig_types, self.DIM_TRIG_TYPE, ignore_label=-1)
            self.embed_roletype = L.EmbedID(n_role_types, self.DIM_ROLE_TYPE, ignore_label=-1)
            self.embed_argtype = L.EmbedID(n_entity_types, self.DIM_ARG_TYPE, ignore_label=-1)
            self.embed_io = L.EmbedID(2, self.DIM_IO, ignore_label=-1)

            self.treelstm = L.ChildSumTreeLSTM(self.DIM_TREE_LSTM_INPUT, self.DIM_EVENT)

            self.l1 = L.Linear(None, self.DIM_EVENT)
            self.y = L.Linear(None, self.DIM_EVENT)
            self.final = L.Linear(None, 1)  # event or non-event
            self.reducedEvent = L.Linear(None, self.DIM_ARG)

            self.len_type_and_arg = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2)
            self.len_relation = self.DIM_TRIG_TYPE + (self.DIM_BILSTM * 2) + self.DIM_ROLE_TYPE + self.DIM_ARG_TYPE + \
                               (self.DIM_BILSTM * 2) + self.DIM_IO

            self.trigger_type2id = trigger_type2id
            self.entity_type2id = entity_type2id


    def _bilstm_layer(self, batch):
        #TODO: Implement batch mechanism
        xs = []
        for i in batch:
            xs.append(np.array(i[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]).astype("i"))

        #TODO: concat all xs into one big list and store lengths
        embed_xs = [F.dropout(self.embed(item),ratio=self.DROPOUT) for item in xs]

        #TODO: separate embed_xs back into list-of-list
        hy, cy, bilstm_xs = self.bilstm(None, None, embed_xs)

        #TODO: into batch again
        return bilstm_xs

    def _event_structure_network(self, trig_id, structure, bilstm_i, batch_i, structures_above_threshold=None):
        def _represent_mentions(mention_ids, bilstm_i):
            try:
                id = mention_ids[0]
                bi = bilstm_i[id]
            except:
                bi = np.zeros((self.DIM_BILSTM * 2), dtype=np.float32)

            mention_array = np.array([bi.data]).astype("f")

            for i in range(len(mention_ids) - 1):
                id = mention_ids[i + 1]
                bi = bilstm_i[id]
                temp = np.array([bi.data]).astype("f")
                mention_array = np.concatenate((mention_array, temp))

            final_mention_representation = F.average(mention_array, axis=0)

            return final_mention_representation

        def _represent_type_and_argument(batch_i, type_index, bilstm_i, type_label, structures_above_threshold=None):

            def _get_word_ids(xsi, mention):
                word_ind = []
                sentence_ids = xsi[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]
                for i in mention:
                    if i in sentence_ids:
                        ind = sentence_ids.index(i)
                        word_ind.append(ind)
                return word_ind

            embedding_list = []

            if structures_above_threshold is not None:
                embedding_list = structures_above_threshold[type_label]
            else:
                defn = batch_i[type_index][type_label]

                type_id = defn[const.IDS_ARG_TYPE]
                type_embedding = None
                if type_index == const.IDS_TRIGGERS_IDX:
                    if self.REPLACE_TYPE:
                        trig_word = self.id2triggertype[type_id]
                        new_arg = Util.extract_category(trig_word, self.GENERALISATION, const.TYPE_GENERALISATION)
                        assert new_arg != '', "ERROR: new_arg is '' "
                        type_id = self.trigger_type2id[new_arg]
                    type_embedding = self.embed_trigtype(np.array([type_id]).astype("i"))
                elif type_index == const.IDS_ENTITIES_IDX:
                    if self.REPLACE_TYPE:
                        ent_word = self.id2entitytype[type_id]
                        new_arg = Util.extract_category(ent_word, self.GENERALISATION, const.TYPE_GENERALISATION)
                        assert new_arg != '', "ERROR: new_arg is '' "
                        type_id = self.entity_type2id[new_arg]
                    type_embedding = self.embed_argtype(np.array([type_id]).astype("i"))

                mention = defn[const.IDS_ARG_MENTION]
                mention_ids = _get_word_ids(batch_i, mention)
                mention_embedding = _represent_mentions(mention_ids, bilstm_i)

                flattened_type_embedding = F.flatten(type_embedding)
                flattened_mention_embedding = F.flatten(mention_embedding)
                type_and_argument_embedding = F.hstack([flattened_type_embedding, flattened_mention_embedding])

                reshaped_type_and_argument_embedding = F.reshape(type_and_argument_embedding, (1, self.len_type_and_arg))
                embedding_list.append(reshaped_type_and_argument_embedding)

            return embedding_list

        def _construct_relation_embedding(trig_embedding, relation, batch_i, bilstm_i, structures_above_threshold=None):

            role = relation[0][0]
            arg = relation[0][1]
            io = relation[1]

            role_type_embedding = self.embed_roletype(np.array([role]).astype("i"))
            triggers = batch_i[const.IDS_TRIGGERS_IDX]
            is_trigger = arg in triggers

            if is_trigger:
                arg_embedding = _represent_type_and_argument(batch_i, const.IDS_TRIGGERS_IDX, bilstm_i, arg, structures_above_threshold)
            else:
                arg_embedding = _represent_type_and_argument(batch_i, const.IDS_ENTITIES_IDX, bilstm_i, arg)

            io_embedding = self.embed_io(np.array([io]).astype('i'))
            relation_embedding = []
            if len(arg_embedding) != 0:
                for i in range(len(arg_embedding)):
                    a = F.flatten(trig_embedding)
                    b = F.flatten(role_type_embedding)
                    c = F.flatten(arg_embedding[i])
                    d = F.flatten(io_embedding)
                    z = F.hstack([a, b, c, d])
                    emb = F.reshape(z, (1, self.len_relation))
                    relation_embedding.append(emb)
            return relation_embedding

        def _tree_lstm_layer(rel_emb):
            cs = []
            hs = []
            for i in range(len(rel_emb)):
                x = rel_emb[i]
                none_cs = none_hs = [None, None]
                c, h = self.treelstm(*none_cs, *none_hs, x)
                cs.append(c)
                hs.append(h)
            x_3 = None

            treelstmrep = None
            if len(cs) == 1 and len(hs) == 1:
                treelstmrep = hs[-1]
            else:
                c_3, h_3 = self.treelstm(*cs, *hs, x_3)
                treelstmrep = h_3
            return treelstmrep

        trig_embedding = _represent_type_and_argument(batch_i, const.IDS_TRIGGERS_IDX, bilstm_i, trig_id)[0]

        tree_representation = []
        if len(structure[0]) == 0:
            #TODO: is this correct?
            treelstmrep = np.zeros((1, self.DIM_EVENT), dtype=np.float32)
            tree_representation.append(treelstmrep)
        else:
            structure_embedding = []
            indices = []
            for relation in structure:
                embedding = _construct_relation_embedding(trig_embedding, relation, batch_i, bilstm_i, structures_above_threshold)
                structure_embedding.append(embedding)

                temp = []
                for i in range(len(embedding)):
                    temp.append(i)
                indices.append(temp)

            # generate all the candidate structures based on the relation representations
            all_rel_emb = []
            for combination in itertools.product(*indices):
                l = list(combination)
                temp = []
                for i in range(len(l)):
                    index = l[i]
                    r = structure_embedding[i][index]
                    temp.append(r)
                all_rel_emb.append(temp)

            for i in all_rel_emb:
                treelstmrep = _tree_lstm_layer(i)
                tree_representation.append(treelstmrep)

        event_representation = None
        prediction = None
        for i in tree_representation:
            event_representation = F.concat((i, trig_embedding))
            event_representation = self.reducedEvent(event_representation)
            h1 = F.relu(self.l1(event_representation))
            y = self.y(h1)
            prediction = self.final(y)
        return prediction, event_representation



    def _predict(self, instance, bilstm_i, target=None):
        def _add_to_predictions(trig, structure_id, structure_defn, pred, representation, predictions):
            def _is_an_event(pred):
                if pred.data > self.THRESHOLD:
                    return True
                return False

            norm_pred = F.sigmoid(pred)
            if _is_an_event(norm_pred):
                predictions[trig].append((structure_id, const.IS_EVENT, representation, structure_defn))
            else:
                predictions[trig].append((structure_id, const.IS_NON_EVENT, representation, structure_defn))

        def _all_sub_events_are_events(structure, predictions):
            all_are = True
            for relation in structure:
                if relation == ():
                    break
                arg = relation[0][1]
                if arg.startswith("TR"):
                    if arg in predictions:
                        preds = predictions[arg]
                        at_least_one_is_event = False
                        for p in preds:
                            if p[1] == const.IS_EVENT:
                                at_least_one_is_event = True
                                break
                        if not at_least_one_is_event:
                            all_are = False
                            break
                    else:
                        all_are = False
                        break
            return all_are

        def _extract_sub_event_predictions_combinations(structure, predictions):

            combination = []
            for relation in structure:
                if relation == ():
                    break
                arg = relation[0][1]
                io = relation[1]
                lst = []
                if arg.startswith("TR") and io == const.IN_EDGE:
                    if arg in predictions:
                        preds = predictions[arg]
                        for p in range(len(preds)):
                            if preds[p][1] == 1:
                                lst.append((arg, p))
                else:
                    lst.append((arg, 0))
                combination.append(lst)

            final_list_of_combination = []
            for comb in itertools.product(*combination):
                l = list(comb)
                final_list_of_combination.append(l)

            return final_list_of_combination

        def _extract_representations(sub_event_prediction_combination, predictions):
            trig_representation = dict()
            for i in sub_event_prediction_combination:
                arg = i[0]
                index = i[1]
                if arg.startswith("TR"):
                    representation = predictions[arg][index][2]
                    trig_representation[arg] = representation
            return trig_representation

        predictions = collections.defaultdict(list)
        instance_loss = 0
        instance_events = instance[const.IDS_EVENT_IDX]

        count = 0

        early_update = False
        for level in range(len(instance_events)):
            for trig, structures in instance_events[level].items():
                for s in range(len(structures)):
                    try:
                        structure_id = structures[s][const.IDS_INSTANCE_ID]
                        structure_defn = structures[s][const.IDS_INSTANCE_DEFN]
                    except:
                        structure_id = structure_defn = const.EMPTY_STRUCTURE
                    if level == 0:
                        pred, representation = self._event_structure_network(trig, structure_id, bilstm_i, instance)
                        _add_to_predictions(trig, structure_id, structure_defn, pred, representation, predictions)
                        if target:
                            current_label = target[level][trig][s]
                            loss = F.sigmoid_cross_entropy(pred, np.array([[current_label]], 'i'))
                            instance_loss += loss
                    else:
                        if _all_sub_events_are_events(structure_id, predictions):
                            sub_event_predictions_combinations = _extract_sub_event_predictions_combinations(structure_id, predictions)
                            for sub_event_prediction_combination in sub_event_predictions_combinations:
                                sub_event_representations = _extract_representations(sub_event_prediction_combination, predictions)
                                pred, representation = self._event_structure_network(trig, structure_id, bilstm_i, instance, sub_event_representations)
                                _add_to_predictions(trig, structure_id, structure_defn, pred, representation, predictions)
                                if target:
                                    current_label = target[level][trig][s]
                                    loss = F.sigmoid_cross_entropy(pred,
                                                                   np.array([[current_label]], 'i'))
                                    instance_loss += loss
                        else:
                            early_update = True
                            # print("early update")
                            break # proceed to next trigger
                    count += 1
            if early_update:
                # print("early update")
                break # stop prediction
        return predictions, instance_loss, count

    def __call__(self, batch, target=None):
        batch_predictions = []
        bilstm_batch = self._bilstm_layer(batch)
        batch_loss = 0
        batch_cnt = 0
        for i in range(len(batch)):
            if target:
                predictions, instance_loss, _ = self._predict(batch[i], bilstm_batch[i], target[i])
                batch_loss += instance_loss
            else:
                predictions, _, count = self._predict(batch[i], bilstm_batch[i])
            batch_predictions.append(predictions)
            batch_cnt += count
        return batch_predictions, batch_loss, batch_cnt