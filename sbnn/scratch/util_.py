import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import chainer.functions as F
import collections
import yaml
from collections import OrderedDict

import logging
log = logging.getLogger(__name__)
import argparse

import src.util.constants as const

def extract_yaml_name(yaml):
    '''
    Extracts the yaml filename.

    :param yaml: filepath of yaml file
    :return: filename
    '''
    temp1 = yaml.split("/")
    name = temp1[-1].split(".yaml")[0]
    return name

def parse_yaml(yaml):
    '''
    Parse event settings from yaml.

    :param yaml: yaml file with EVENT settings
    :return: EVENT settings
    '''
    with open(yaml, 'r') as stream:
        args = Util.ordered_load(stream)
    args = args['EVENT']
    log.info('Loaded parameters: %s', yaml)
    return args


class Util(object):
    def add_position_information(file_instances, boundaries):
        def map_words2pos(sen, start):
            map = dict()
            tokens = sen.split(" ")
            fr = start
            for t in range(len(tokens)):
                l = len(tokens[t])
                to = fr + l
                map[str(fr) + "_" + str(to)] = t
                fr = to + 1  # 1 for the space between tokens
            return map

        def map_dct2position(d, map):
            def get_best_idx(fr, to, map):
                #    This is the index where [fr,to] has the most overlap
                bestidx = -1
                bestcoverage = 0
                idx = -1
                a = -1
                b = -1
                for k, v in map.items():
                    idx += 1
                    fr_,to_ = k.split("_")
                    if int(fr_) <= fr or int(to_) >= to:
                        if fr < int(fr_):
                            a = int(fr_)
                        else:
                            a = fr
                        if to > int(to_):
                            b = int(to_)
                        else:
                            b = to
                        if (b-a) >  bestcoverage:
                            bestcoverage = b - a
                            bestidx = idx
                return bestidx


            newd = collections.defaultdict(list)
            for k, v in d.items():
                words = v[3:]
                fr = int(v[1])
                for i in range(len(words)):
                    l = len(words[i])
                    to = fr + l
                    if str(fr) + "_" + str(to) in map:
                        idx = map[str(fr) + "_" + str(to)]
                    else:  #
                        idx = get_best_idx(fr, to, map)
                    newd[k].append(idx)

                    fr = to + 1
            return newd

        for i in range(len(file_instances)):
            start, _ = boundaries[i]
            instance = file_instances[i]
            sen = instance[0][0]
            sen_dct = map_words2pos(sen, start)
            ent_dct = map_dct2position(instance[1], sen_dct)
            trig_dct = map_dct2position(instance[2], sen_dct)
            id2position_mapping = {**ent_dct, **trig_dct}
            file_instances[i][0].append(id2position_mapping)

    # @staticmethod
    # def exlude(instances, idx):
    #     new_instances = []
    #     for i in range(len(instances)):
    #         id = instances[i][0][1]
    #         sen = instances[i][0][0]
    #         found = False
    #         if id in idx:
    #             sens = idx[id]
    #             for s in sens:
    #                 if sen.startswith(s):
    #                     found = True
    #                     break
    #         if not found:
    #             new_instances.append(instances[i])
    #     return new_instances

    @staticmethod
    def merge(instances, instance_ids):
        new_instance_ids = []
        for i in range(len(instance_ids)):
            instance = instance_ids[i]
            entities = instances[i][const.IDS_ENTITIES_IDX]
            triggers = instances[i][const.IDS_TRIGGERS_IDX]
            instance.append(entities)
            instance.append(triggers)
            new_instance_ids.append(instance)
        return new_instance_ids

    @staticmethod
    def max_num_gold_actions(instance_ids):
        max_num_gold_actions_dict = dict()
        max_num_gold_actions = 0
        for i in instance_ids:
            triggers_structures_mapping = dict()
            # gather triggers in bottom-up sequence
            groups = i[const.PRED_CAND_STRUCTURES_IDX]
            for g in range(len(groups)):
                for trig, trigstructures in groups[g].items():
                    # triggers_structures_mapping[trig] = trigstructures

                    for s in range(len(trigstructures)):
                        edges = trigstructures[s][1]
                        num_events = len(trigstructures[s][2])
                        gold_actions = dict()
                        for j in range(len(edges)):
                            # all_unique = set()
                            for l in range(num_events):
                                label = trigstructures[s][2][l][j][1]
                                label_action = label.index(1)
                                # all_unique.add(label_action)
                                if l in gold_actions:
                                    prev_gold_action = gold_actions[l]
                                    new_gold_action = ''.join(prev_gold_action) + str(label_action)
                                    gold_actions[l] = new_gold_action
                                else:
                                    gold_actions[l] = str(label_action)
                            all_unique = set()
                            for k, v in gold_actions.items():
                                all_unique.add(v)
                            max_num = len(all_unique)
                            if max_num > max_num_gold_actions:
                                max_num_gold_actions = max_num
                            if max_num in max_num_gold_actions_dict:
                                max_num_gold_actions_dict[max_num] += 1
                            else:
                                max_num_gold_actions_dict[max_num] = 1
        return max_num_gold_actions_dict,max_num_gold_actions

    @staticmethod
    def count_gold_events_in_ids(instance_ids, USE_FILTER):
        count = 0
        ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
        for i in instance_ids:
            events = i[ind]
            for level in range(len(events)):
                event_structures = events[level]
                for trigger, _ in event_structures.items():
                    trigger_num_events = len(event_structures[trigger][0][2])
                    count += trigger_num_events
        return count

    @staticmethod
    def compute_nestedness(instances, gold, USE_FILTER):
        def count_events_in_gold(gold_instance):
            trigger_count = dict()
            for event_id, structure in gold_instance.items():
                trigger = structure[0].split(":")[1]
                if trigger in trigger_count:
                    trigger_count[trigger] += 1
                else:
                    trigger_count[trigger] = 1
            return trigger_count
        ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
        event_count = dict()

        for i in range(len(instances)):
            eventstructures = instances[i][ind]
            trigger_count = count_events_in_gold(gold[i][const.GOLD_EVENTS_IDX])
            for level in range(len(eventstructures)):
                events = eventstructures[level]
                num_events = 0
                for trigger, _ in events.items():
                    if trigger in trigger_count:
                        events_with_trigger = trigger_count[trigger]
                        num_events += events_with_trigger
                if level in event_count:
                    event_count[level] += num_events
                else:
                    event_count[level] = num_events
        return event_count
    @staticmethod
    def replace_event_arg_with_trigger(c_gold, gold_events):
        new_structure = []
        new_structure.append(c_gold[0])

        new_args = []
        args = c_gold[1:]
        for a in args:
            role = a.split(":")[0]
            arg = a.split(":")[1]
            if arg.startswith("E"):
                event = gold_events[arg]
                arg = event[0].split(":")[1]
            new_args.append(role+":"+arg)
        new_structure.extend(new_args)
        return new_structure

    @staticmethod
    def create_trig_event_map(events):
        trigger_event_map = collections.defaultdict(list)
        for _, structure in events.items():
            trigger = structure[0].split(":")[1]
            trigger_event_map[trigger].append(structure)
        return trigger_event_map

    @staticmethod
    def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(stream, OrderedLoader)

    @staticmethod
    def load_txt_to_list_with_boundaries(file, IS_EMO, main_txt_file=None):
        sentences = []
        boundaries = []
        cur_len = 0

        main_txt = None
        # if IS_EMO:
        #     main_txt = open(main_txt_file, 'r').read()
        with open(file, 'r') as file_read:
            lines = file_read.readlines()

            if IS_EMO:
                i = 0
                while i < len(lines):
                    line = lines[i]
                    start, end = line.split(":")
                    boundaries.append((int(start), int(end)))
                    sentence = lines[i + 1]  # add 1 to include the last character
                    sentences.append(sentence)
                    i += 2

            else:
                for i in range(len(lines)):
                    line = lines[i].rstrip()
                    sentences.append(line)
                    boundaries.append((cur_len, cur_len + len(line)))
                    cur_len = cur_len + len(line) + 1

        return sentences, boundaries
    # @staticmethod
    # def load_txt_to_list_with_boundaries(file):
    #     sentences = []
    #     boundaries = []
    #     cur_len = 0
    #
    #     with open(file, 'r') as file_read:
    #         lines = file_read.readlines()
    #         for i in range(len(lines)):
    #             line = lines[i].strip()
    #             sentences.append(line)
    #             boundaries.append((cur_len, cur_len + len(line)))
    #             cur_len = cur_len + len(line) + 1
    #
    #     return sentences, boundaries

    @staticmethod
    def get_index(e_from, e_to, boundaries):
        ind = -1
        for i in range(len(boundaries)):
            fr = boundaries[i][0]
            to = boundaries[i][1]
            if e_from >= fr and e_to <= to:
                ind = i
                break
        return ind

    @staticmethod
    def get_indices(dic, boundaries):
        lst_indices = collections.defaultdict(dict)
        for k, v in dic.items():
            e_from = v[1]
            e_to = v[2]
            index = Util.get_index(int(e_from), int(e_to), boundaries)
            # assert index != -1, "Error: index == -1"
            lst_indices[index][k] = v
        return lst_indices

    @staticmethod
    def get_arg_index(trig, indices):
        ind = -1
        for k, v in indices.items():
            if trig in v:
                ind = k
                break
        return ind

    @staticmethod
    def get_rel_indices(rels, ent_indices, trig_indices):

        rel_index = collections.defaultdict(dict)
        invalid_count = 0
        total_count = 0
        for id, defn in rels.items():
            trig = defn[1]
            arg = defn[2]

            trig_ind = Util.get_arg_index(trig, trig_indices)
            arg_ind = Util.get_arg_index(arg, ent_indices)
            if arg_ind == -1:
                arg_ind = Util.get_arg_index(arg, trig_indices)

            if trig_ind == arg_ind:
                rel_index[trig_ind][id] = defn
                total_count += 1
            else:
                log.debug("Not supported: intersentence relation: %s, %s, %s, %s", str(id),
                          defn, str(trig_ind), str(arg_ind))
                invalid_count += 1
        return rel_index, invalid_count, total_count

    @staticmethod
    def get_event_indices(events, ent_indices, trig_indices):

        def _same_indices(indices):
            same = True
            cur = indices[0]
            for i in indices[1:]:
                if i != cur:
                    same = False
                    break
            return same

        def _get_event_trig_map(events):
            event_trig_map = dict()
            for id, defn in events.items():
                trig = defn[0].split(":")[1]
                event_trig_map[id] = trig
            return event_trig_map

        event_index = collections.defaultdict(dict)

        event_trig_id_map = _get_event_trig_map(events)

        excluded_events = []

        count = 0
        for e_id, defn in events.items():
            indices = []
            for t in range(len(defn)):
                pair = defn[t].split(":")
                id = pair[1]
                if id.startswith("E"):  # a sub-event so get the trigger id
                    if id in event_trig_id_map:
                        id = event_trig_id_map[id]
                    else:
                        id = None
                ind = Util.get_arg_index(id, trig_indices)
                if ind == -1:
                    ind = Util.get_arg_index(id, ent_indices)
                indices.append(ind)
            if _same_indices(indices):
                event_index[indices[0]][e_id] = defn
            else:
                log.debug("Not supported: intersentence event: %s, %s, %s", id, defn, indices)
                count += 1
                excluded_events.append(e_id)
        return event_index, count, excluded_events

    @staticmethod
    def extract_category(arg, level, TYPE_GENERALISATION):
        new_arg = ''
        # assert arg in TYPE_GENERALISATION, ("ERROR: type not in the CG types. %s", arg)
        if arg in TYPE_GENERALISATION:
            if level == 0:
                new_arg = arg
            elif level == 1:
                new_arg = TYPE_GENERALISATION[arg][0]
            elif level == 2:
                new_arg = TYPE_GENERALISATION[arg][1]
        else:
            log.warning("Using input type as type not in the CG types: %s ", arg)
            new_arg = arg
        return new_arg

    @staticmethod
    def prepareBatch(train_prediction, train_target=None):
        plain_labels = None
        trig_labels = None
        if train_target:
            plain_labels, trig_labels = _to_matrix(train_target)
        plain_preds,  trig_preds= _to_matrix(train_prediction, is_target=False)
        if train_target:
            temp = _handle_extra_predictions(trig_labels, trig_preds)
            plain_labels = np.array(temp, 'i')
        preds_matrix = F.vstack(plain_preds)
        return preds_matrix, plain_labels

    @staticmethod
    def _to_matrix(batch, is_target=True):
        def _get_labels(structures, result):
            trig_labels = dict()
            for trig, targets in structures.items():
                labels = []
                for i in targets:
                    if is_target:
                        result.append(i)
                        labels.append(i)
                    else:
                        try:
                            if type(i) is list:
                                for item in i:
                                    result.append(item)
                                    labels.append(i)
                            else:
                                result.append(i)
                                labels.append(i)
                        except:
                            print("except")
                            # i = [1,0] #if it is not a target,
                            # then if prediction is zero that means there is no structure predicted
                            # so non-event
                trig_labels[trig] = labels
            return trig_labels

        trig_labels = []
        plain_labels = []
        for groups in batch:
            groups_labels = []
            for group in groups:
                group_labels = _get_labels(group, plain_labels)
                groups_labels.append(group_labels)
            trig_labels.append(groups_labels)
        return plain_labels, trig_labels

    @staticmethod
    def _handle_extra_predictions(trig_labels, trig_preds):
        overallTarget = []
        for i in range(len(trig_labels)):
            for g in trig_labels[i]:
                for k, v in g.items():
                    lenv = len(v)
                    v_pred = trig_preds[i][0][k]
                    lenv_pred = len(v_pred)
                    for ind in range(len(v)):
                        preds = v_pred[ind]
                        target = v[ind]
                        for _ in range(len(preds)):
                            overallTarget.append(target)
        return overallTarget

    @staticmethod
    def format_filtered_instances(filtered_test_instances, role_type2id):
        def _label(structures, role_type2id):
            labeled_structures = collections.defaultdict(list)
            for group in structures:
                for trig, structures in group.items():
                    for structure in structures:
                        new_structure = []
                        for relation in structure:
                            if relation == ():
                                new_structure = config.EMPTY_STRUCTURE
                                break
                            role = relation[0][0]
                            arg = relation[0][1]
                            io = relation[1]
                            new_relation = ((role_type2id[role], arg), io)
                            new_structure.append(new_relation)
                        final_structure = [new_structure, config.IS_EVENT, 0]
                        labeled_structures[trig].append(final_structure)
            return labeled_structures


        predictions = []
        for i in range(len(filtered_test_instances)):
            structures = filtered_test_instances[i][config.PRED_FILTERED_STRUCTURES_IDX]
            labeled_structures = _label(structures, role_type2id)
            predictions.append(labeled_structures)
        return predictions

    @staticmethod
    def count_events_non_events(train_target_labels):
        zeros = 0
        ones = 0
        for instance in train_target_labels:
            for group in instance:
                for trig, structures in group.items():
                    for label in structures:
                        if label == 1:
                            ones += 1
                        else:
                            zeros += 1
        return ones, zeros

    @staticmethod
    def count_event_structures(data):
        flat = 0
        nested = 0
        for i in data:
            ind = 5
            if config.USE_FILTER:
                ind = 6
            groups = i[ind]
            for g in groups:
                for trig, structures in g.items():
                    for structure in structures:
                        is_flat = True
                        for rels in structure:
                            try:
                                arg = rels[0][1]
                            except:
                                arg = ''
                            if arg.startswith("TR"):
                                is_flat = False
                                break
                        if is_flat:
                            flat += 1
                        else:
                            nested += 1
        return flat, nested

    @staticmethod
    def count_event_flat_nested_in_event_structures(data):
        flat = 0
        nested = 0
        for g in data:
            for trig, structures in g.items():
                for structure in structures:
                    is_flat = True
                    for rels in structure:
                        try:
                            arg = rels[0][1]
                        except:
                            arg = ''
                        if arg.startswith("TR"):
                            is_flat = False
                            break
                    if is_flat:
                        flat += 1
                    else:
                        nested += 1
        return flat, nested

    @staticmethod
    def count_predicted_flat_nested_events(index, instances, predictions, target):
        '''
            Count the predicted flat and nested events.

        '''
        flat = 0
        nested = 0
        flat_tp = flat_tn = flat_fp = flat_fn = 0
        nested_tp = nested_tn = nested_fp = nested_fn = 0
        for i in range(len(instances)):
            groups = instances[i][index]
            for g in range(len(groups)):
                for trig, structures in groups[g].items():
                    for s in range(len(structures)):
                        is_flat = True
                        label = target[i][g][trig][s]
                        prediction = 0
                        if trig in predictions[i]:
                            try:
                                prediction = predictions[i][trig][s][1]
                            except:
                                print("analysis")
                        for rel in structures[s]:
                            if rel == ():
                                break
                            arg = rel[0][1]
                            if arg.startswith("TR"):
                                is_flat = False
                                break
                        if is_flat:
                            flat += 1
                            if prediction == 1:
                                if label == 1:
                                    flat_tp += 1
                                else:
                                    flat_fp += 1
                            else:
                                if label == 1:
                                    flat_fn += 1
                                else:
                                    flat_tn += 1

                        else:
                            nested += 1
                            if prediction == 1:
                                if label == 1:
                                    nested_tp += 1
                                else:
                                    nested_fp += 1
                            else:
                                if label == 1:
                                    nested_fn += 1
                                else:
                                    nested_tn += 1

        return [flat, nested, flat_tp, flat_tn, flat_fp, flat_fn, nested_tp, nested_tn, nested_fp, nested_fn]

    @staticmethod
    def prec_rec_f(info):
        flat = info[0]
        nested = info[1]
        flat_tp = info[2]
        flat_tn = info[3]
        flat_fp = info[4]
        flat_fn = info[5]
        nested_tp = info[5]
        nested_tn = info[6]
        nested_fp = info[7]
        nested_fn = info[8]
        tp = flat_tp + nested_tp
        fn = flat_fn + nested_fn
        fp = flat_fp + nested_fp
        recall = tp / (tp + fn)
        prec = tp / (tp + fp)
        fscore = (2*recall*prec) / (prec+recall)
        return prec, recall, fscore


    @staticmethod
    def check_instances_with_diff(gold, target, instances):

        def count_instance_events(instance):
            ones = 0
            for group in instance:
                for trig, structures in group.items():
                    for label in structures:
                        if label == 1:
                            ones += 1
            return ones

        assert len(gold) == len(target) == len(instances), "ERROR: lengths must be the same"

        not_the_same = []
        for i in range(len(gold)):
            g_events = len(gold[i][3])
            t_events = count_instance_events(target[i])
            if g_events != t_events:
                not_the_same.append((instances[i], gold[i], target[i]))
        return not_the_same


    ################## DEBUGGING CODE SNIPPETS #################
    '''
    Count candidate structures before and after filtering.
    Change the index accordingly.
    '''
    # flat = 0
    # nested = 0
    # for i in new_instances:
    #     groups = i[5]
    #     for g in groups:
    #         for trig, structures in g.items():
    #             for structure in structures:
    #                 is_flat = True
    #                 for rel in structure:
    #                     if rel == () or rel == ((), 1):
    #                         break
    #                     arg = rel[0][1]
    #                     if arg.startswith("TR"):
    #                         is_flat = False
    #                         break
    #                 if is_flat:
    #                     flat += 1
    #                 else:
    #                     nested += 1

    '''
    Count simple and nested events in gold.
    '''
    # flat = 0
    # nested = 0
    # for i in test_new_gold:
    #     events = i[3]
    #     for id, structure in events.items():
    #         is_nested = False
    #         for s in structure:
    #             arg = s.split(":")[1]
    #             if arg.startswith("E"):
    #                 is_nested = True
    #                 break
    #         if is_nested:
    #             nested += 1
    #         else:
    #             flat += 1

    '''
    Count events in gold.
    '''
    # count = 0
    # for i in train_new_gold:
    #     count += len(i[3])

    '''
    Count flat and nested event structures and gold events after labelling. 
    Change the index of the candidate structures accordingly.
    '''
    # flat = 0
    # nested = 0
    # true_flat_events = 0
    # true_nested_events = 0
    # for i in range(len(train_instances)):
    #     groups = train_instances[i][6]
    #     for g in range(len(groups)):
    #         for trig, structures in groups[g].items():
    #             for s in range(len(structures)):
    #                 is_flat = True
    #                 label = train_target_labels[i][g][trig][s]
    #                 for rel in structures[s]:
    #                     if rel == () or rel == ((), 1):
    #                         break
    #                     arg = rel[0][1]
    #                     if arg.startswith("TR"):
    #                         is_flat = False
    #                         break
    #                 if is_flat:
    #                     flat += 1
    #                     if label == 1:
    #                         true_flat_events += 1
    #                 else:
    #                     nested += 1
    #                     if label == 1:
    #                         true_nested_events += 1

    '''
    Counts for prec, rec, fscore
    '''
    # flat = 0
    # nested = 0
    # f_tp = 0
    # f_fp = 0
    # f_tn = 0
    # f_fn = 0
    # n_tp = 0
    # n_fp = 0
    # n_tn = 0
    # n_fn = 0
    # for i in range(len(train_instances)):
    #     groups = train_instances[i][6]
    #     for g in range(len(groups)):
    #         for trig, structures in groups[g].items():
    #             for s in range(len(structures)):
    #                 is_flat = True
    #                 label = train_target_labels[i][g][trig][s]
    #                 prediction = 0
    #                 if trig in train_predictions[i]:
    #                     prediction = train_predictions[i][trig][s][1]
    #                 for rel in structures[s]:
    #                     if rel == () or rel == ((), 1):
    #                         break
    #                     arg = rel[0][1]
    #                     if arg.startswith("TR"):
    #                         is_flat = False
    #                         break
    #                 if is_flat:
    #                     flat += 1
    #                     if label == 1:
    #                         if prediction == 1:
    #                             f_tp += 1
    #                         else:
    #                             f_fn += 1
    #                     else:
    #                         if prediction == 1:
    #                             f_fp += 1
    #                         else:
    #                             f_tn += 1
    #                 else:
    #                     nested += 1
    #                     if label == 1:
    #                         if prediction == 1:
    #                             n_tp += 1
    #                         else:
    #                             n_fn += 1
    #                     else:
    #                         if prediction == 1:
    #                             n_fp += 1
    #                         else:
    #                             n_tn += 1
    '''
    Retrieve events of certain type from gold.
    '''
    # temp = []
    # for i in train_gold_instances:
    #     events = i[3]
    #     for id, value in events.items():
    #         type = value[0].split(":")[0]
    #         if type == 'Planned_process':
    #             if len(value) > 3:
    #                 temp.append(i)

from collections import OrderedDict, Callable

class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))