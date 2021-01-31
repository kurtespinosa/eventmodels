#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019 18:58
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""


from collections import OrderedDict, Counter
import os
import logging
log = logging.getLogger(__name__)

from src.util import util as util
from src.util import constants as const
from src.util.util import DefaultOrderedDict


def remove_instances_with_attributes(instances, empty_indices):
    new_instances = []
    for i in range(len(instances)):
        if i not in empty_indices:
            new_instances.append(instances[i])
    return new_instances

def load(DIR, IS_EMO):
    def get_arg_index(trig, indices):
        '''

        :param trig:
        :param indices:
        :return:
        '''
        ind = -1
        for k, v in indices.items():
            if trig in v:
                ind = k
                break
        return ind

    def get_event_indices(events, ent_indices, trig_indices):
        def _same_indices(indices):
            '''

            :param indices:
            :return:
            '''
            same = True
            cur = indices[0]
            for i in indices[1:]:
                if i != cur:
                    same = False
                    break
            return same

        def _get_event_trig_map(events):
            '''

            :param events:
            :return:
            '''
            event_trig_map = OrderedDict()
            for id, defn in events.items():
                trig = defn[0].split(":")[1]
                event_trig_map[id] = trig
            return event_trig_map

        event_index = DefaultOrderedDict(dict)
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
                ind = get_arg_index(id, trig_indices)
                if ind == -1:
                    ind = get_arg_index(id, ent_indices)
                indices.append(ind)
            if _same_indices(indices):
                event_index[indices[0]][e_id] = defn
            else:
                log.debug("Not supported: intersentence event: %s, %s, %s", id, defn, indices)
                count += 1
                excluded_events.append(e_id)
        return event_index, count, excluded_events

    def _load_to_list_ann(file):
        '''

        :param file:
        :return:
        '''
        entities = OrderedDict()
        triggers = OrderedDict()
        rels = OrderedDict()
        events = OrderedDict()

        with open(file, 'r') as file_read:
            lines = file_read.readlines()
            for line in lines:
                tokens = line.split()
                id = tokens[0]
                defn = tokens[1:]
                if line.startswith("R"):
                    rels[id] = defn
                elif line.startswith("TR"):
                    triggers[id] = defn
                elif line.startswith("T"):
                    entities[id] = defn
                elif line.startswith("E"):
                    events[id] = defn
        return [entities, triggers, rels, events]

    def _associate_ann_to_sentences(sentences, boundaries, quad, file):
        '''

        :param sentences:
        :param boundaries:
        :param quad:
        :param file:
        :return:
        '''
        def _remove_events(events, excluded_events):
            '''

            :param events:
            :param excluded_events:
            :return:
            '''
            new_events = OrderedDict()
            for id, defn in events.items():
                if id not in excluded_events:
                    new_events[id] = defn
            return new_events

        inter_count = 0
        quad_ent_indices = util.get_indices(quad[0], boundaries)
        quad_trig_indices = util.get_indices(quad[1], boundaries)
        event_indices, count, excluded_events = get_event_indices(quad[3],
                                                                  quad_ent_indices,
                                                                  quad_trig_indices)
        inter_count += count
        events = quad[3]

        # loop to remove events cascade
        while len(excluded_events) > 0:
            new_events = _remove_events(events, excluded_events)
            event_indices, count, excluded_events = get_event_indices(new_events,
                                                                    quad_ent_indices,
                                                                    quad_trig_indices)
            inter_count += count
            events = new_events

        instances = []

        for i in range(len(sentences)):
            sentence_at_i = [sentences[i], file, i]
            events_ents_at_i = quad_ent_indices[i]
            events_trigs_at_i = quad_trig_indices[i]
            events_at_i = event_indices[i]
            instance = [sentence_at_i, events_ents_at_i, events_trigs_at_i, events_at_i]
            instances.append(instance)
        return instances, inter_count

    def check_for_invalid_event_structures(quad, result_file, file):
        '''

        :param quad:
        :param result_file:
        :param file:
        :return:

        TODO: generalise this to read a list of invalid structures
        '''

        def is_valid_roles_combination(roles):
            '''

            :param roles:
            :return:
            '''
            flag = True
            if 'CSite' in roles:
                if 'Cause' not in roles:
                    flag = False
            return flag

        # For now this only handles this: every structure that has CSite role must have a Cause role as well

        events = quad[3]
        for id, structure in events.items():
            roles = []
            for rel in structure[1:]:
                role, arg = rel.split(":")
                roles.append(role)
            if not is_valid_roles_combination(roles):
                log.error("ERROR: invalid structure found in gold:%s, %s, %s", file,
                          id, structure)

    def load_txt_to_list_with_boundaries(file, IS_EMO, main_txt_file=None):
        '''

        :param file:
        :param IS_EMO:
        :param main_txt_file:
        :return:
        '''
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

    def add_position_information(file_instances, boundaries):
        '''

        :param file_instances:
        :param boundaries:
        :return:
        '''
        def map_words2pos(sen, start):
            '''

            :param sen:
            :param start:
            :return:
            '''
            map = OrderedDict()
            tokens = sen.split(" ")
            fr = start
            for t in range(len(tokens)):
                l = len(tokens[t])
                to = fr + l
                map[str(fr) + "_" + str(to)] = t
                fr = to + 1  # 1 for the space between tokens
            return map

        def map_dct2position(d, map):
            '''

            :param d:
            :param map:
            :return:
            '''
            def get_best_idx(fr, to, map):
                '''

                :param fr:
                :param to:
                :param map:
                :return:
                '''
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


            newd = DefaultOrderedDict(list)
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

    instances = []
    files = os.listdir(DIR)
    file_id_events = DefaultOrderedDict(list)

    intersentence_event_count = 0
    num_events = 0
    main_txt_file = None
    for file in files:
        if file.endswith(const.GOLD_TXT_OUTPUT_EXT):
            file = file.split(const.GOLD_TXT_OUTPUT_EXT)[0]
            if IS_EMO:
                main_txt_file = DIR + file + const.GOLD_MAIN_TXT_EXT
            txt_file = DIR + file + const.GOLD_TXT_OUTPUT_EXT
            sentences, boundaries = load_txt_to_list_with_boundaries(txt_file,
                                                                          IS_EMO,
                                                                          main_txt_file)
            ann_file = DIR + file + const.GOLD_ANN_OUTPUT_EXT
            quad = _load_to_list_ann(ann_file)
            num_events += len(quad[3])

            # check for invalid event structures
            # check_for_invalid_event_structures(quad, result_file, file)

            file_id_events[file].append(quad)
            file_instances, count = _associate_ann_to_sentences(sentences, boundaries,
                                                                quad, file)

            # add position info to sentence and entities/triggers
            add_position_information(file_instances, boundaries)

            intersentence_event_count += count
            instances.extend(file_instances)
    log.info("%s: %s gold events, %s (%.2f%%) intersentence events", DIR,
             str(num_events), str(intersentence_event_count),
             intersentence_event_count / num_events * 100)
    return instances, file_id_events

def denumber_roles(gold):
    '''
    Removes the numbers in roles and changes the gold instance.

    :param gold: contains the instances with the sentence, entities, triggers and events
    :return: modified events in the gold
    '''
    for i in gold:
        events = i[const.GOLD_EVENTS_IDX]
        new_events = OrderedDict()
        for id, event in events.items():
            new_event = []
            for rel in event:
                temp = rel.split(":")
                role = temp[0]
                arg = temp[1]
                if role[-1].isdigit():
                    role = role[:-1]
                new_event.append(role+":"+arg)
            new_events[id] = new_event
        i[const.GOLD_EVENTS_IDX] = new_events
    return gold

def extract_templates(file_id_events):
    '''

    :param file_id_events:
    :return:
    '''
    def _extract_all_unique_structure_combinations(file_id_events):
        '''

        :param file_id_events:
        :return:
        '''

        def extractEventStructures(file_id_events):
            '''

            :param file_id_events:
            :return:
            '''
            def extractFullString(fullstring, mentionLevel=False):
                '''

                :param fullstring:
                :param mentionLevel:
                :return:
                '''

                '''
                Given a trigger/entity definition, extract the string.
                '''
                typ = fullstring[0]
                new_word = []
                if mentionLevel:
                    word = fullstring[3]
                    new_word.append(typ)
                    new_word.append(word)
                else:
                    new_word.append(typ)
                return new_word

            def extractEventStructure(v):
                '''

                :param v:
                :return:
                '''

                def _remove_number_in_role(role):
                    '''

                    :param role:
                    :return:
                    '''
                    last_char = role[-1]
                    new_role = role
                    if last_char.isdigit():
                        new_role = role[0:-1]
                    return new_role

                structure = []
                is_valid = True
                if len(v) == 1:  # event does not have arguments/participants
                    structure.append(())
                else:
                    for i in v[1:]: #skips trigger info
                        role = [i.split(":")[0]][0]
                        role = _remove_number_in_role(role)
                        arg = i.split(":")[1]
                        isSubevent = False
                        if arg.startswith("T"):
                            try:
                                arg = extractFullString(entities[arg], False)
                            except(KeyError):
                                log.debug("Discarding structure %s as it contains intersentence argument: %s", v, arg)
                                is_valid = False
                                break
                        elif arg.startswith("E"):
                            try:
                                args = events[arg]
                                trig = triggers[args[0].split(":")[1]]
                                arg = extractFullString(trig, False)
                                isSubevent = True
                            except(KeyError):
                                log.debug("Discarding structure %s as it contains intersentence argument: %s", v, arg)
                                is_valid = False
                                break
                        if isSubevent:
                            structure.append((role, arg[0], True))
                        else:
                            structure.append((role, arg[0], False))
                return structure, is_valid

            structures = DefaultOrderedDict(list)

            for fid, quad in file_id_events.items():
                entities = quad[0][0]
                triggers = quad[0][1]
                relations = quad[0][2]
                events = quad[0][3]

                for k, v in events.items():
                    triggerstring = triggers[v[0].split(":")[1]]
                    trigger = extractFullString(triggerstring, True)
                    structure, is_valid = extractEventStructure(v)
                    if is_valid:
                        structures[trigger[0]].append(structure)
                    else: # assumes that invalid events have intersentence arguments
                        log.debug("Event has intersentence arguments")
            return structures

        def getLevel(structures, level):
            '''

            :param structures:
            :param level:
            :return:
            '''
            '''
            Loops thru all the structures and using the hierarchy of types, changes the argument
            based on the level.
            '''
            newstructures = DefaultOrderedDict(list)

            for k, v in structures.items():
                newv = []
                for i in v:
                    newi = []
                    for j in i:
                        if len(j) == 0:
                            newj = ()
                        else:
                            newarg = util.extract_category(j[1], level, const.TYPE_GENERALISATION)
                            newj = (j[0], newarg)
                        newi.append(newj)
                    newv.append(newi)
                newstructures[k].append(newv)
            return newstructures

        structures = extractEventStructures(file_id_events)

        abstractedStructures = []
        abstractedStructures.append(getLevel(structures, 0))
        abstractedStructures.append(getLevel(structures, 1))
        abstractedStructures.append(getLevel(structures, 2))

        return abstractedStructures

    def _remove_duplicates(structures):
        '''

        :param structures:
        :return:
        '''
        finallist = []
        for m in range(3):  # 3 levels

            uniquelist = DefaultOrderedDict(list)
            for trigger, v in structures[m].items():

                biglist = []
                for i in structures[m][trigger][0]:
                    lst = []  # permits repeated arguments

                    for j in i:
                        if len(j) == 0:
                            biglist.append(lst)
                            break

                        role = j[0]
                        type = j[1]
                        joined = role + "__" + type
                        lst.append(joined)
                    biglist.append(lst)

                # put them in hashtable based on the length
                newdict = DefaultOrderedDict(list)
                for i in biglist:
                    leni = len(i)
                    newdict[leni].append(i)

                # going thru each hashtable index, remove the duplicates by using multiset to represent each structure
                '''
                For each structure, check if it is already in the final list by going thru the final list and representing
                each element as a multiset then comparing it with the current structure.
                If they are the same, that means, the structure is a duplicate so do not add it to the final list
                Otherwise, add it.
                '''
                newdictlist = DefaultOrderedDict(list)
                for k, v in newdict.items():
                    newlist = []
                    for j in newdict[k]:
                        s = Counter(j)
                        found = False
                        for n in newlist:
                            sk = Counter(n)
                            ss = Counter(s)
                            if ss == sk:
                                found = True
                                break
                        if not found:
                            newlist.append(dict(s))
                    newdictlist[str(k)].append(newlist)
                uniquelist[trigger].append(newdictlist)
            finallist.append(uniquelist)
        return finallist


    templates = _extract_all_unique_structure_combinations(file_id_events)
    templates = _remove_duplicates(templates)

    # log.info("There are %s intersentence events in the train gold excluded by filter.", str(intersentence_event_count))
    return templates

def generate_gold_actions(train_instances, train_new_gold=None):

    def _generate(instance, group, gold_structures=None):

        def _loop_thru_relations_and_assign_actions(structure, gold_structure=None):
            def _check_action_against_gold(gold_structure, rel):
                action = [0,0,0]

                if rel in gold_structure:
                    if len(gold_structure) == 1:
                        action[const.ACTION_ADDFIX] = 1
                    else:
                        action[const.ACTION_ADD] = 1
                    gold_structure.remove(rel)
                else:
                    action[const.ACTION_IGNORE] = 1
                return action, gold_structure

            structure_actions = []
            # indices_addfix = dict()
            if gold_structure is None:
                for r in range(len(structure)):
                    rel = structure[r]
                    action = [1,0,0] #ignore
                    structure_actions.append((rel, action))

            else:
                for g in gold_structure:
                    structure_action = []
                    for r in range(len(structure)):
                        rel = structure[r]
                        action, g = _check_action_against_gold(g[:], rel)
                        structure_action.append((rel, action))
                    structure_actions.append(structure_action)

            return structure_actions

        group = instance[const.PRED_CAND_STRUCTURES_IDX][group]
        new_group = DefaultOrderedDict(list)

        for trig, data in group.items():
            for d in range(len(data)):
                structure = data[d][0]
                id = data[d][1]

                assert len(id) == len(structure), "Error: length of structure != id"
                #gather the mapping for role and args since the comparison has to be made with the gold
                #because of the numbered roles
                arg_roleid_mapping = OrderedDict()
                role_arg_mapping = OrderedDict()
                roleid_arg_mapping = OrderedDict()
                for s in structure:
                    if s != ():
                        role = s[0]
                        arg = s[1]
                        role_arg_mapping[arg] = role
                for s in id:
                    if s != ():
                        roleid = s[0]
                        arg = s[1]
                        roleid_arg_mapping[arg] = roleid
                for k,v in role_arg_mapping.items():
                    if k in roleid_arg_mapping:
                        arg_roleid_mapping[k] = roleid_arg_mapping[k]

                if structure == [()]:
                    structure = []
                    id = []
                structure.insert(0, (const.NONE_ROLE_TYPE, const.NONE_ARG_TYPE))
                id.insert(0, (const.ID_NONE_ROLE_TYPE, const.NONE_ARG_TYPE))

                gold = None
                if gold_structures:
                    if trig in gold_structures:
                        gold = gold_structures[trig]
                        if gold == [[()]]:
                            gold = [[(const.NONE_ROLE_TYPE, const.NONE_ARG_TYPE)]]
                        actions = _loop_thru_relations_and_assign_actions(structure, gold)
                    else:
                        actions = _loop_thru_relations_and_assign_actions(structure, None)
                else:
                    actions = _loop_thru_relations_and_assign_actions(structure, None)

                #replace role word with ids
                gold_actions = []
                if gold_structures:
                    if gold is not None:
                        for g in range(len(gold)):
                            new_actions = []
                            for a in actions[g]:
                                edge = a[0]
                                action = a[1]
                                # if type(edge) is tuple:
                                edge_arg = edge[1]
                                if edge_arg == const.NONE_ROLE_TYPE:
                                    edge_roleid = const.ID_NONE_ROLE_TYPE
                                else:
                                    edge_roleid = arg_roleid_mapping[edge_arg]
                                # else:
                                #     edge_roleid = const.ID_NONE_ROLE_TYPE
                                #     edge_arg = None #just a
                                new_edge = (edge_roleid, edge_arg)
                                new_actions.append((new_edge, action))
                            gold_actions.append(new_actions)
                        new_group[trig].append((structure, id, gold_actions))
                    else:
                        gold_actions.append(actions)
                        new_group[trig].append((structure, id, gold_actions))
                else:
                    gold_actions.append(actions)
                    new_group[trig].append((structure, id, gold_actions))

        return new_group

    def _index_using_triggers(gold_structures):
        def _transform_to_tuples(gold_structures):
            new_gold_structures = DefaultOrderedDict(list)
            for trig_id, structures in gold_structures.items():
                for s in structures:
                    if s == []:
                        new_structure = [()]
                    else:
                        new_structure = []
                        for rel in s:
                            role, arg = rel.split(":")
                            new_rel = (role, arg)
                            new_structure.append(new_rel)
                    new_gold_structures[trig_id].append(new_structure)
            return new_gold_structures

        new_gold_structures = DefaultOrderedDict(list)
        event_trigger_mapping = OrderedDict()
        for event_id, structure in gold_structures.items():
            trigger_id = structure[0].split(":")[1]
            new_gold_structures[trigger_id].append(structure[1:])
            event_trigger_mapping[event_id] = trigger_id

        transformed_structures = _transform_to_tuples(new_gold_structures)

        new_gold = DefaultOrderedDict(list)
        for trig_id, structures in transformed_structures.items():

            for structure in structures:
                if structure == const.EMPTY_STRUCTURE:
                    new_structure = [()]
                else:
                    new_structure = []
                    for rel in structure:
                        role = rel[0]
                        arg = rel[1]
                        if arg in event_trigger_mapping:
                            arg = event_trigger_mapping[arg]
                        new_rel = (role, arg)
                        new_structure.append(new_rel)
                new_gold[trig_id].append(new_structure)
        return new_gold

    def _count_gold_structures(gold_structures):
        triggers = []
        num_structures = 0
        for trig, structures in gold_structures.items():
            triggers.append(trig)
            num_structures += len(structures)
        return triggers, num_structures

    def _count_cand_structures(groups):
        triggers = []
        num_structures = 0
        for g in groups:
            for trig, structures in g.items():
                triplet = structures[0]
                structs = triplet[2]
                num_structures += len(structs)
                # num_structures += len(structures)
                triggers.append(trig)
        return triggers, num_structures

    def _are_gold_covered(gold_triggers, cand_triggers):
        covered = True
        for t in gold_triggers:
            if t not in cand_triggers:
                covered = False
                break
        return covered

    new_instances = []
    structures_idx = const.PRED_CAND_STRUCTURES_IDX
    for ind in range(len(train_instances)):
        orig_groups = train_instances[ind][structures_idx]

        gold_structures1 = None
        gold_structures = None
        if train_new_gold:
            gold_structures1 = train_new_gold[ind][const.GOLD_EVENTS_IDX]
            gold_structures = _index_using_triggers(gold_structures1)
        groups = []
        for g in range(len(orig_groups)):
            group = _generate(train_instances[ind], g, gold_structures)
            groups.append(group)
        #check all gold structures are covered
        # gold_triggers, gold_num_structures = _count_gold_structures(gold_structures)
        # cand_triggers, cand_num_structures = _count_cand_structures(groups)

        #checking for GOLD relations only
        # assert gold_num_structures <= cand_num_structures, ("ERROR: gold_num_structures !<= cand_num_structures", ind)
        # assert _are_gold_covered(gold_triggers, cand_triggers), ("ERROR: some gold structures not covered in cand_num_structures", ind)

        instance = []
        instance.extend(train_instances[ind][0:structures_idx])
        instance.append(groups)
        new_instances.append(instance)
    return new_instances

