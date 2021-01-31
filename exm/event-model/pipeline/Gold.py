import os
import collections

from pipeline.util import Util
import constants as const
import numpy as np

class Gold(object):

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
    def load(DIR, VERBOSE, IS_EMO):
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
            event_indices, count, excluded_events = Util.get_event_indices(quad[3], quad_ent_indices, quad_trig_indices, VERBOSE)

            inter_count += count
            events = quad[3]

            #loop to remove events cascade
            while len(excluded_events) > 0:
                new_events = _remove_events(events, excluded_events)
                event_indices, count, excluded_events = Util.get_event_indices(new_events, quad_ent_indices, quad_trig_indices, VERBOSE)
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

        instances = []

        files = os.listdir(DIR)

        file_id_events = collections.defaultdict(list)

        intersentence_event_count = 0
        main_txt_file = None
        for file in files:
            if file.endswith(const.GOLD_TXT_OUTPUT_EXT):
                file = file.split(const.GOLD_TXT_OUTPUT_EXT)[0]
                # if IS_EMO:
                #     main_txt_file = DIR + file + const.GOLD_MAIN_TXT_EXT
                txt_file = DIR + file + const.GOLD_TXT_OUTPUT_EXT
                sentences, boundaries = Util.load_txt_to_list_with_boundaries(txt_file, IS_EMO, main_txt_file)
                ann_file = DIR + file + const.GOLD_ANN_OUTPUT_EXT
                quad = _load_to_list_ann(ann_file)
                file_id_events[file].append(quad)
                file_instances, count = _associate_ann_to_sentences(sentences, boundaries, quad, file)
                intersentence_event_count += count
                instances.extend(file_instances)
        return instances, intersentence_event_count, file_id_events

    @staticmethod
    def extract_filter(file_id_events, VERBOSE):

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
                                    if VERBOSE:
                                        print("Error: discarding structure ", v, " as it contains intersentence argument:", arg)
                                    is_valid = False
                                    break
                            elif arg.startswith("E"):
                                try:
                                    args = events[arg]
                                    trig = triggers[args[0].split(":")[1]]
                                    arg = extractFullString(trig, False)
                                    isSubevent = True
                                except(KeyError):
                                    if VERBOSE:
                                        print("Error: discarding structure ", v, " as it contains intersentence argument:", arg)
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
                            if VERBOSE:
                                print("ERROR: event has intersentence arguments")
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
                                newarg = Util.extract_category(j[1], level, const.TYPE_GENERALISATION, VERBOSE)
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


        filter = _extract_all_unique_structure_combinations(file_id_events)
        filter = _remove_duplicates(filter)

        return filter


    # @staticmethod
    # def extract_filter(instances, VERBOSE):
    #
    #     def _extract_all_unique_structure_combinations(dataset):
    #         '''
    #         Extract into a set all unique combinations of argument structures associated per trigger
    #         from the dataset.
    #         '''
    #
    #         def extractEventStructures(instances):
    #
    #             def extractFullString(fullstring, mentionLevel=False):
    #                 '''
    #                 Given a trigger/entity definition, extract the string.
    #                 '''
    #                 typ = fullstring[0]
    #                 new_word = []
    #                 if mentionLevel:
    #                     word = fullstring[3]
    #                     new_word.append(typ)
    #                     new_word.append(word)
    #                 else:
    #                     new_word.append(typ)
    #                 return new_word
    #
    #             def extractEventStructure(v):
    #
    #                 def _remove_number_in_role(role):
    #                     last_char = role[-1]
    #                     new_role = role
    #                     if last_char.isdigit():
    #                         new_role = role[0:-1]
    #                     return new_role
    #
    #                 structure = []
    #                 is_valid = True
    #                 if len(v) == 1:  # event does not have arguments/participants
    #                     structure.append(())
    #                 else:
    #                     for i in v[1:]: #skips trigger info
    #                         role = [i.split(":")[0]][0]
    #
    #                         role = _remove_number_in_role(role)
    #
    #                         arg = i.split(":")[1]
    #                         isSubevent = False
    #                         if arg.startswith("T"):
    #                             try:
    #                                 arg = extractFullString(entities[arg], False)
    #                             except(KeyError):
    #                                 if VERBOSE:
    #                                     print("Error: discarding structure ", v, " as it contains intersentence argument:", arg)
    #                                 is_valid = False
    #                                 break
    #                         elif arg.startswith("E"):
    #                             try:
    #                                 args = events[arg]
    #                                 trig = triggers[args[0].split(":")[1]]
    #                                 arg = extractFullString(trig, False)
    #                                 isSubevent = True
    #                             except(KeyError):
    #                                 if VERBOSE:
    #                                     print("Error: discarding structure ", v, " as it contains intersentence argument:", arg)
    #                                 is_valid = False
    #                                 break
    #                         if isSubevent:
    #                             structure.append((role, arg[0], True))
    #                         else:
    #                             structure.append((role, arg[0], False))
    #                 return structure, is_valid
    #
    #             structures = collections.defaultdict(list)
    #
    #             intersentence_event_count = 0
    #             for i in range(len(instances)):
    #                 file_id  = instances[i][const.GOLD_SENTENCE_INFO_IDX][const.GOLD_SENTENCE_FILE_ID_IDX]
    #                 events = instances[i][const.GOLD_EVENTS_IDX]
    #                 #uses the gold entities and triggers
    #                 entities = instances[i][const.GOLD_ENTITIES_IDX]
    #                 triggers = instances[i][const.GOLD_TRIGGERS_IDX]
    #
    #                 for k, v in events.items():
    #                     triggerstring = triggers[v[0].split(":")[1]]
    #                     trigger = extractFullString(triggerstring, True)
    #                     structure, is_valid = extractEventStructure(v)
    #
    #                     if is_valid:
    #                         structures[trigger[0]].append(structure)
    #                     else: # assumes that invalid events have intersentence arguments
    #                         intersentence_event_count += 1
    #             return structures, intersentence_event_count
    #
    #         def join_argument(p):
    #             pair = ""
    #             for a in p:
    #                 pair += a[0] + "_"
    #             return pair
    #
    #         def extractUniqueRoleArgPairs(structures):
    #             pairs = set()
    #             for k, v in structures.items():
    #
    #                 regustruct = structures[k]
    #                 for r in regustruct:
    #
    #                     for p in r:
    #                         pair = join_argument(p)
    #
    #                         pairs.add(pair)
    #
    #             listpairs = list(pairs)
    #             return listpairs
    #
    #         def convertToSet(structures):
    #             newstructure = dict()
    #             for s, v in structures.items():
    #                 items = structures[s]
    #                 unique = list(set(str(i) for i in structures[s]))
    #                 newstructure[s] = unique
    #
    #             return newstructure
    #
    #         def getLevel(structures, level):
    #             '''
    #             Loops thru all the structures and using the hierarchy of types, changes the argument
    #             based on the level.
    #             '''
    #             newstructures = collections.defaultdict(list)
    #
    #             for k, v in structures.items():
    #                 newv = []
    #                 for i in v:
    #                     newi = []
    #                     for j in i:
    #                         if len(j) == 0:
    #                             newj = ()
    #                         else:
    #                             newarg = Util.extract_category(j[1], level, const.TYPE_GENERALISATION)
    #                             newj = (j[0], newarg)
    #                         newi.append(newj)
    #                     newv.append(newi)
    #                 newstructures[k].append(newv)
    #             return newstructures
    #
    #         structures, count = extractEventStructures(dataset)
    #
    #         abstractedStructures = []
    #         abstractedStructures.append(getLevel(structures, 0))
    #         abstractedStructures.append(getLevel(structures, 1))
    #         abstractedStructures.append(getLevel(structures, 2))
    #
    #         return abstractedStructures, count
    #
    #     def _remove_duplicates(structures):
    #         finallist = []
    #         for m in range(3):  # 3 levels
    #
    #             uniquelist = collections.defaultdict(list)
    #             for trigger, v in structures[m].items():
    #
    #                 biglist = []
    #                 for i in structures[m][trigger][0]:
    #                     lst = []  # permits repeated arguments
    #
    #                     for j in i:
    #                         if len(j) == 0:
    #                             biglist.append(lst)
    #                             break
    #
    #                         role = j[0]
    #                         type = j[1]
    #                         joined = role + "__" + type
    #                         lst.append(joined)
    #                     biglist.append(lst)
    #
    #                 # put them in hashtable based on the length
    #                 newdict = collections.defaultdict(list)
    #                 for i in biglist:
    #                     leni = len(i)
    #                     newdict[leni].append(i)
    #
    #                 # going thru each hashtable index, remove the duplicates by using multiset to represent each structure
    #                 '''
    #                 For each structure, check if it is already in the final list by going thru the final list and representing
    #                 each element as a multiset then comparing it with the current structure.
    #                 If they are the same, that means, the structure is a duplicate so do not add it to the final list
    #                 Otherwise, add it.
    #                 '''
    #                 newdictlist = collections.defaultdict(list)
    #                 for k, v in newdict.items():
    #                     newlist = []
    #                     for j in newdict[k]:
    #                         s = collections.Counter(j)
    #                         found = False
    #                         for n in newlist:
    #                             sk = collections.Counter(n)
    #                             ss = collections.Counter(s)
    #                             if ss == sk:
    #                                 found = True
    #                                 break
    #                         if not found:
    #                             newlist.append(dict(s))
    #                     newdictlist[str(k)].append(newlist)
    #                 uniquelist[trigger].append(newdictlist)
    #             finallist.append(uniquelist)
    #         return finallist
    #
    #
    #     filter, count = _extract_all_unique_structure_combinations(instances)
    #     filter = _remove_duplicates(filter)
    #     return filter, count

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