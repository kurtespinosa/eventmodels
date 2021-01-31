import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)


import os
import collections

from src.util.util_ import Util
import src.util.constants as const

import logging
log = logging.getLogger(__name__)

class GoldDataProcessor(object):

    @staticmethod
    def denumber_roles(gold):
        '''
        Removes the numbers in roles and changes the gold instance.
        :param gold: contains the instances with the sentence, entities, triggers and events
        :return: modified events in the gold
        '''
        for i in gold:
            events = i[const.GOLD_EVENTS_IDX]
            new_events = dict()
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
    @staticmethod
    def get_gold_entities(DIRECTORY):
        files = os.listdir(DIRECTORY)
        entities_dict = dict()
        for file in files:
            if file.endswith(const.GOLD_TXT_OUTPUT_EXT):
                file = file.split(const.GOLD_TXT_OUTPUT_EXT)
                filename = DIRECTORY + file[0]
                a1file = filename + const.GOLD_ANN_OUTPUT_EXT
                entities = Gold._readEntitiesIntoDict(a1file)
                entities_dict[file[0]] = entities
        return entities_dict

    @staticmethod
    def extract_unique_triggers(gold_instances):
        trig_types = set()
        for g in gold_instances:
            triggers = g[2]
            for k,v in triggers.items():
                typ = v[0]
                trig_types.add(typ)
        return trig_types

    @staticmethod
    def _parseTriggerOrEntityLine(line):
        tokens = line.split()
        id = tokens[0]
        role = tokens[1]
        offset_from = tokens[2]
        offset_to = tokens[3]
        word = ' '.join(tokens[4:])
        return id, role, offset_from, offset_to, word

    @staticmethod
    def _readEntitiesIntoDict(filename):
        file = open(filename, 'r')
        entities = dict()
        for line in file:
            if line.startswith("T"):
                if line.startswith("TR"):
                    continue
                else:
                    id, role, offset_from, offset_to, word = Gold._parseTriggerOrEntityLine(line)
                    entities[id] = (role, int(offset_from), int(offset_to), word)
        return entities

    @staticmethod
    def load(DIR, IS_EMO):
        def _load_to_list_ann(file):
            entities = dict()
            triggers = dict()
            rels = dict()
            events = dict()

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

            def _remove_events(events, excluded_events):
                new_events = dict()
                for id, defn in events.items():
                    if id not in excluded_events:
                        new_events[id] = defn
                return new_events

            inter_count = 0
            quad_ent_indices = Util.get_indices(quad[0], boundaries)
            quad_trig_indices = Util.get_indices(quad[1], boundaries)
            event_indices, count, excluded_events = Util.get_event_indices(quad[3],
                                                    quad_ent_indices,
                                                    quad_trig_indices)

            inter_count += count
            events = quad[3]

            #loop to remove events cascade
            while len(excluded_events) > 0:
                new_events = _remove_events(events, excluded_events)
                event_indices, count, excluded_events = Util.get_event_indices(new_events,
                                                        quad_ent_indices,
                                                        quad_trig_indices)
                inter_count += count
                events = new_events

            instances = []

            for i in range(len(sentences)):
                sentence_at_i = [sentences[i], file]
                events_ents_at_i = quad_ent_indices[i]
                events_trigs_at_i = quad_trig_indices[i]
                events_at_i = event_indices[i]
                instance = [sentence_at_i, events_ents_at_i, events_trigs_at_i, events_at_i]
                instances.append(instance)
            return instances, inter_count

        def check_for_invalid_event_structures(quad, result_file, file):
            #TODO: generalise this to read a list of invalid structures
            def is_valid_roles_combination(roles):
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

        instances = []
        files = os.listdir(DIR)
        file_id_events = collections.defaultdict(list)

        intersentence_event_count = 0
        num_events = 0
        main_txt_file = None
        for file in files:
            if file.endswith(const.GOLD_TXT_OUTPUT_EXT):
                file = file.split(const.GOLD_TXT_OUTPUT_EXT)[0]
                if IS_EMO:
                    main_txt_file = DIR + file + const.GOLD_MAIN_TXT_EXT
                txt_file = DIR + file + const.GOLD_TXT_OUTPUT_EXT
                sentences, boundaries = Util.load_txt_to_list_with_boundaries(txt_file,
                                                                              IS_EMO,
                                                                              main_txt_file)
                ann_file = DIR + file + const.GOLD_ANN_OUTPUT_EXT
                quad = _load_to_list_ann(ann_file)
                num_events += len(quad[3])
                # check_for_invalid_event_structures(quad, result_file, file)
                file_id_events[file].append(quad)
                file_instances, count = _associate_ann_to_sentences(sentences, boundaries,
                                                                    quad, file)

                #add position info to sentence and entities/triggers
                Util.add_position_information(file_instances, boundaries)

                intersentence_event_count += count
                instances.extend(file_instances)
        log.info("%s: %s gold events, %s (%.2f%%) intersentence events", DIR,
                 str(num_events), str(intersentence_event_count),
                 intersentence_event_count/num_events*100)
        return instances, file_id_events

    @staticmethod
    def extract_templates(file_id_events):

        def _extract_all_unique_structure_combinations(file_id_events):

            def extractEventStructures(file_id_events):
                def extractFullString(fullstring, mentionLevel=False):

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

                    def _remove_number_in_role(role):
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

                structures = collections.defaultdict(list)

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
                Loops thru all the structures and using the hierarchy of types, changes the argument
                based on the level.
                '''
                newstructures = collections.defaultdict(list)

                for k, v in structures.items():
                    newv = []
                    for i in v:
                        newi = []
                        for j in i:
                            if len(j) == 0:
                                newj = ()
                            else:
                                newarg = Util.extract_category(j[1], level, const.TYPE_GENERALISATION)
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
            finallist = []
            for m in range(3):  # 3 levels

                uniquelist = collections.defaultdict(list)
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
                    newdict = collections.defaultdict(list)
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
                    newdictlist = collections.defaultdict(list)
                    for k, v in newdict.items():
                        newlist = []
                        for j in newdict[k]:
                            s = collections.Counter(j)
                            found = False
                            for n in newlist:
                                sk = collections.Counter(n)
                                ss = collections.Counter(s)
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


    @staticmethod
    def extract_types_with_no_arguments(filter):
        types = []
        for trig, lst in filter[0].items():
            if '0' in lst[0]:
                types.append(trig)
        return types

    @staticmethod
    def remove_instances_with_attributes(instances, empty_indices):
        new_instances = []
        for i in range(len(instances)):
            if i not in empty_indices:
                new_instances.append(instances[i])
        return new_instances

    @staticmethod
    def remove_intersentence_events(instances):
        new_instances = []

        return new_instances



    @staticmethod
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
            new_group = collections.defaultdict(list)

            for trig, data in group.items():
                for d in range(len(data)):
                    structure = data[d][0]
                    id = data[d][1]

                    assert len(id) == len(structure), "Error: length of structure != id"
                    #gather the mapping for role and args since the comparison has to be made with the gold
                    #because of the numbered roles
                    arg_roleid_mapping = dict()
                    role_arg_mapping = dict()
                    roleid_arg_mapping = dict()
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
                    structure.insert(0, const.NONE_ROLE_TYPE)
                    id.insert(0, const.NONE_ROLE_TYPE)

                    gold = None
                    if gold_structures:
                        if trig in gold_structures:
                            gold = gold_structures[trig]
                            if gold == [[()]]:
                                gold = [[(const.NONE_ROLE_TYPE)]]
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
                                    if type(edge) is tuple:
                                        edge_arg = edge[1]
                                        edge_roleid = arg_roleid_mapping[edge_arg]
                                    else:
                                        edge_roleid = 0  # because const.NONE_ROLE_TYPE id is 0
                                        edge_arg = None #just a
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
                new_gold_structures = collections.defaultdict(list)
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

            new_gold_structures = collections.defaultdict(list)
            event_trigger_mapping = dict()
            for event_id, structure in gold_structures.items():
                trigger_id = structure[0].split(":")[1]
                new_gold_structures[trigger_id].append(structure[1:])
                event_trigger_mapping[event_id] = trigger_id

            transformed_structures = _transform_to_tuples(new_gold_structures)

            new_gold = collections.defaultdict(list)
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


    @staticmethod
    def label_instances(instances, gold_instances, USE_FILTER, VERBOSE):
        def _convert_cand_structures(train_instances, USE_FILTER):
            def _convert(train_instance, index, structures_idx):
                '''
                Converts each structure into a multiset for easy comparison.
                :param train_instance:
                :param index:
                :return:
                '''
                all_events = collections.defaultdict(list)

                entities = train_instance[const.PRED_ENTITIES_IDX]
                triggers = train_instance[const.PRED_TRIGGERS_IDX]

                candidate_structures = train_instance[structures_idx]

                if len(candidate_structures) != 0:
                    for trig_id, structures in candidate_structures[index].items():

                        if structures[0] == const.EMPTY_STRUCTURE or structures[0] == [((),1)]:
                            multiset = collections.Counter([])
                            all_events[trig_id].append(multiset)
                        else:
                            for structure in structures:
                                new_structure = []
                                for relation in structure:
                                    edge_label = relation[1]
                                    role_n_type = relation[0]
                                    role = role_n_type[0]
                                    arg_id = role_n_type[1]
                                    if arg_id in triggers:
                                        defn = triggers[arg_id]
                                    else:
                                        defn = entities[arg_id]
                                    arg_type = defn[0]
                                    offset_start = defn[1]
                                    offset_end = defn[2]

                                    if edge_label == 1:
                                        joined = role + "__" + arg_type + "__"+str(offset_start)+str(offset_end)
                                        new_structure.append(joined)

                                multiset = collections.Counter(new_structure)

                                all_events[trig_id].append(multiset)
                return all_events

            all_structures = []

            structures_idx = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
            for ind in range(len(train_instances)):
                orig_groups = train_instances[ind][structures_idx]

                groups = []
                for g in range(len(orig_groups)):
                # for g in range(len(train_instances[ind][const.PRED_CAND_STRUCTURES_IDX])):
                    group =  _convert(train_instances[ind], g, structures_idx)
                    groups.append(group)
                all_structures.append(groups)
            return all_structures

        def _convert_events(gold_instances):

            new_gold_instances = []

            count_tr = 0
            count_inter = 0
            for ind in range(len(gold_instances)):
                entities = gold_instances[ind][const.GOLD_ENTITIES_IDX]
                triggers = gold_instances[ind][const.GOLD_TRIGGERS_IDX]
                events = gold_instances[ind][const.GOLD_EVENTS_IDX]
                all_events = collections.defaultdict(list)
                for eve_id, structure in events.items():
                    new_structure = []
                    trig_id = structure[0].split(":")[1]
                    arguments = structure[1:]
                    is_convertible = True
                    for i in range(len(arguments)):
                            role, type = arguments[i].split(":")
                            joined = ''
                            if type.startswith("TR"):
                                if VERBOSE:
                                    print("Error: invalid argument ", arguments[i], "skipping event ", eve_id)
                                count_tr += 1
                                is_convertible = False
                                break
                            elif type.startswith("T"):
                                arg_type = entities[type][0]
                                offset_start = entities[type][1]
                                offset_end = entities[type][2]

                                joined = role + "__" + arg_type+ "__"+str(offset_start)+str(offset_end)
                            elif type.startswith("E"):
                                if type not in events:
                                    if VERBOSE:
                                        print("Not supported: cannot convert intersentence events", eve_id, structure, type)
                                    is_convertible = False
                                    count_inter += 1
                                    break
                                event_def = events[type]
                                arg_type = event_def[0].split(":")[0]
                                arg_id = event_def[0].split(":")[1]
                                trig_def = triggers[arg_id]
                                offset_start = trig_def[1]
                                offset_end = trig_def[2]


                                joined = role + "__" + arg_type+ "__"+str(offset_start)+str(offset_end)
                            new_structure.append(joined)
                    if is_convertible:
                        multiset = collections.Counter(new_structure)
                        all_events[trig_id].append(multiset)
                new_gold_instances.append(all_events)
            return new_gold_instances, count_tr, count_inter


        def _get_labels(events, gold_events):
            labels = collections.defaultdict(list)
            for id, structures in events.items():
                for structure in structures:
                    if id in gold_events:
                        gold_structures = gold_events[id]
                        is_event = False
                        for gold_structure in gold_structures:
                            if structure == gold_structure:
                                is_event = True
                                break
                        if is_event:
                            labels[id].append(const.IS_EVENT)
                        else:
                            labels[id].append(const.IS_NON_EVENT)
                    else:
                        labels[id].append(const.IS_NON_EVENT)
            return labels

        train_structures = _convert_cand_structures(instances, USE_FILTER)
        gold_structures, count_tr, count_inter = _convert_events(gold_instances)

        assert len(train_structures) == len(gold_structures), "ERROR: len(train_structures) != len(gold_structures)"

        instances_labels = []
        for index in range(len(train_structures)):
            gold_events = gold_structures[index]
            groups_labels = []
            for i in range(len(train_structures[index])):
                group = train_structures[index][i]
                group_label = _get_labels(group, gold_events)
                groups_labels.append(group_label)
            instances_labels.append(groups_labels)
        return instances_labels, count_tr, count_inter