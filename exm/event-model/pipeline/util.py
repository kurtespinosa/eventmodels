import numpy as np
import chainer.functions as F
import constants as config
import collections
import yaml
from collections import OrderedDict
import constants as const
import random

class Util(object):

    @staticmethod
    def exlude(instances, idx):
        new_instances = []
        for i in range(len(instances)):
            id = instances[i][0][1]
            sen = instances[i][0][0]
            found = False
            if id in idx:
                sens = idx[id]
                for s in sens:
                    if sen.startswith(s):
                        found = True
                        break
            if not found:
                new_instances.append(instances[i])
        return new_instances

    @staticmethod
    def count_event_predictions(predictions):
        cnt = 0
        for i in predictions:
            for trig, structures in i.items():
                for structure in structures:
                    if structure[1] == 1:
                        cnt += 1
        return cnt

    @staticmethod
    def count_written_events(ready_for_writing):
        cnt = 0
        for fileid, contents in ready_for_writing.items():
            events = contents[2]
            cnt += len(events)
        return cnt

    @staticmethod
    def shuffle(instances_ids, instances, target_labels=None):
        if target_labels == None:
            bundle = list(zip(instances_ids, instances))
        else:
            bundle = list(zip(instances_ids, instances, target_labels))
        random.shuffle(bundle)
        result = zip(*bundle)
        return result

    @staticmethod
    def print_event_type_dist(train_type_dist, test_type_dist):
        print("\nTrigger\tTr_Pos\tTs_Pos\tTr_Neg\tTs_Neg")
        for trigger, values in train_type_dist.items():
            tr_pos, tr_neg = values
            if trigger in test_type_dist:
                ts_pos, ts_neg = test_type_dist[trigger]
                print(trigger,"\t", tr_pos, "\t", ts_pos, "\t", tr_neg, "\t", ts_neg)
            else:
                print(trigger, "\t", tr_pos, "\t", 0, "\t", tr_neg, "\t", 0)

        print("\n")

    @staticmethod
    def extract_event_sample_distribution(instances, targets, USE_FILTER):
        ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX

        event_type_dist = dict()
        for i in range(len(instances)):
            events = instances[i][ind]
            triggers = instances[i][const.PRED_TRIGGERS_IDX]
            target_labels = targets[i]
            for g in range(len(events)):
                for trigger, structures in events[g].items():
                    labels = target_labels[g][trigger]
                    trigger_type = triggers[trigger][0]
                    for s in range(len(structures)):
                        structure = structures[s]
                        label = labels[s]
                        if label == const.IS_EVENT:
                            # print("Event")
                            if trigger_type in event_type_dist:
                                pos, neg = event_type_dist[trigger_type]
                                pos +=1
                                event_type_dist[trigger_type] = [pos, neg]
                            else:
                                event_type_dist[trigger_type] = [1, 0]
                        else:
                            # print("Non-Event")
                            if trigger_type in event_type_dist:
                                pos, neg = event_type_dist[trigger_type]
                                neg +=1
                                event_type_dist[trigger_type] = [pos, neg]
                            else:
                                event_type_dist[trigger_type] = [0, 1]
        return event_type_dist

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

        # main_txt = None
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
                    sentence = lines[i+1]  # add 1 to include the last character
                    sentences.append(sentence)
                    i += 2

            else:
                for i in range(len(lines)):
                    line = lines[i].rstrip()
                    sentences.append(line)
                    boundaries.append((cur_len, cur_len + len(line)))
                    cur_len = cur_len + len(line) + 1

        return sentences, boundaries
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
    def get_rel_indices(rels, ent_indices, trig_indices, VERBOSE):

        rel_index = collections.defaultdict(dict)
        count = 0
        trig_arg_pairs = []
        for id, defn in rels.items():
            trig = defn[1]
            arg = defn[2]

            trig_ind = Util.get_arg_index(trig, trig_indices)
            arg_ind = Util.get_arg_index(arg, ent_indices)
            if arg_ind == -1:
                arg_ind = Util.get_arg_index(arg, trig_indices)

            if trig_ind == arg_ind:
                rel_index[trig_ind][id] = defn
            else:
                if VERBOSE:
                    print("Not supported: intersentence relations", id, defn, trig_ind, arg_ind)
                count += 1
                trig_arg_pairs.append((trig, arg))

        return rel_index, count, trig_arg_pairs

    @staticmethod
    def get_event_indices(events, ent_indices, trig_indices, VERBOSE):

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
                if VERBOSE:
                    print("Not supported: intersentence events", id, defn, indices)
                count += 1
                excluded_events.append(e_id)
        return event_index, count, excluded_events

    @staticmethod
    def extract_category(arg, level, TYPE_GENERALISATION, VERBOSE):
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
            if VERBOSE:
                print("ERROR: type not in the CG types > ", arg)
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
                                print("debug")
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