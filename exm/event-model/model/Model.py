# -*- coding: utf-8 -*-
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

import itertools
import collections
import os
from datetime import datetime
import numpy as np
import math
from numpy import binary_repr
from functools import reduce as _reduce

from model.EventStructureClassifier import EventStructureClassifier
from pipeline.util import Util


import constants as const
import heapq


class Model(object):

    @staticmethod
    def load(DIR, DIR_NER, DIR_REL, VERBOSE, IS_EMO):
        def _load_to_list_ner_ann(file):
            entities = dict()
            triggers = dict()
            with open(file, 'r') as file_read:
                lines = file_read.readlines()
                for line in lines:
                    tokens = line.split()
                    if line.startswith("TR"):
                        triggers[tokens[0]] = [tokens[1], tokens[2], tokens[3], ' '.join(tokens[4:])]
                    elif line.startswith("T"):
                        entities[tokens[0]] = [tokens[1], tokens[2], tokens[3], ' '.join(tokens[4:])]
            return entities, triggers

        def _load_to_list_rel_ann(file):
            rels = dict()
            with open(file, 'r') as file_read:
                lines = file_read.readlines()
                for line in lines:
                    tokens = line.split()
                    rels[tokens[0]] = tokens[1:]
            return rels

        def _associate_ann_to_sentences(sentences, boundaries, entities, triggers, rels, file):

            def _extract_unique_nodes(rels):
                nodes = set()
                for k, v in rels.items():
                    nodes.add(v[1])
                    nodes.add(v[2])
                return nodes

            ent_indices = Util.get_indices(entities, boundaries)
            trig_indices = Util.get_indices(triggers, boundaries)
            rel_indices, count, trig_arg_pairs = Util.get_rel_indices(rels, ent_indices, trig_indices, VERBOSE)

            instances = []

            for i in range(len(sentences)):
                sentence_at_i = [sentences[i], file]
                entities_at_i = ent_indices[i]
                triggers_at_i = trig_indices[i]
                relations_at_i = rel_indices[i]
                nodes = _extract_unique_nodes(relations_at_i)
                instance = [sentence_at_i, entities_at_i, triggers_at_i, relations_at_i, nodes]
                instances.append(instance)
            return instances, count, trig_arg_pairs

        instances = []

        files = os.listdir(DIR)

        invalid_rel_count = 0
        all_file_ids = []
        main_txt_file = None
        for file in files:
            if file.endswith(const.GOLD_TXT_OUTPUT_EXT):
                file = file.split(const.GOLD_TXT_OUTPUT_EXT)[0]
                # if file == 'PMID-15776477':
                #     print("debug load")

                all_file_ids.append(file)

                txt_file = DIR + file + const.GOLD_TXT_OUTPUT_EXT
                rel_ann_file = DIR_REL + file + const.PRED_REL_EXT
                ner_ann_file = DIR_NER + file + const.PRED_NER_EXT
                if os.path.isfile(rel_ann_file) and os.path.isfile(ner_ann_file):
                    if IS_EMO:
                        main_txt_file = DIR + file + const.GOLD_MAIN_TXT_EXT
                    sentences, boundaries = Util.load_txt_to_list_with_boundaries(txt_file, IS_EMO, main_txt_file)
                    entities, triggers = _load_to_list_ner_ann(ner_ann_file)
                    rels = _load_to_list_rel_ann(rel_ann_file)
                    file_instances, count, trig_arg_pairs = _associate_ann_to_sentences(sentences, boundaries, entities, triggers, rels, file)
                    '''
                    Uncomment the lines below to reveal the intersentence events.
                    
                    '''
                    # print("File:", file, "count:", count)
                    # print("trig-arg-pairs:", trig_arg_pairs)

                    instances.extend(file_instances)
                    invalid_rel_count += count
        return instances, all_file_ids, invalid_rel_count

    @staticmethod
    def create_vocabulary(instances, UNK_MIN_FREQUENCY, UNK_ASSIGNMENT_PROBABILITY):

        def _add_words_to_collection(counter, s):
            tokens = s.split()
            for k in tokens:
                counter.update(k.strip().split())

        def _add_types_to_collection(counter, s):
            counter.update(s.strip().split())

        words = collections.Counter()
        role_types = collections.Counter()
        trigger_types = collections.Counter()
        entity_types = collections.Counter()

        # Stores words with frequency MIN_FREQ
        singleton = set()

        for instance in instances:
            sentence = instance[const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_IDX]

            _add_words_to_collection(words, sentence)
            entities = instance[const.PRED_ENTITIES_IDX]
            triggers = instance[const.PRED_TRIGGERS_IDX]
            relations = instance[const.PRED_RELATIONS_IDX]

            for k, v in triggers.items():
                typ = v[0]
                _add_types_to_collection(trigger_types, typ)
                word = v[3]
                _add_words_to_collection(words, word)

            for k, v in entities.items():
                typ = v[0]
                _add_types_to_collection(entity_types, typ)
                word = v[3]
                _add_words_to_collection(words, word)

            # extract roles from events
            # events = instance[const.GOLD_EVENTS_IDX]
            # for k, v in events.items():
            #     lenv = len(v)
            #     for i in range(lenv - 1):
            #         arg = v[i + 1]
            #         roleType = arg.split(":")[0]
            #         _add_types_to_collection(role_types, roleType)

            # extract roles from relations
            for k,v in relations.items():
                roleType = v[0]
                _add_types_to_collection(role_types, roleType)

        word_type2id = collections.OrderedDict()
        role_type2id = collections.OrderedDict()
        trigger_type2id = collections.OrderedDict()
        entity_type2id = collections.OrderedDict()

        for c in const.CONSTANT_WORDS:
            word_type2id[c] = len(word_type2id)

        word_count_list = words.most_common()
        for (word, count) in word_count_list:
            if count <= UNK_MIN_FREQUENCY:
                if np.random.random_sample() > UNK_ASSIGNMENT_PROBABILITY:
                    word_type2id[word] = len(word_type2id)
                else:
                    word_type2id[word] = word_type2id[const.UNK_TOKEN]
                    singleton.add(word)
            else:
                word_type2id[word] = len(word_type2id)

        # for roles
        word_count_list = role_types.most_common()
        for (word, count) in word_count_list:
            role_type2id[word] = len(role_type2id)

        # For trigger
        word_count_list = trigger_types.most_common()
        for (word, count) in word_count_list:
            trigger_type2id[word] = len(trigger_type2id)

        # For entity
        word_count_list = entity_types.most_common()
        for (word, count) in word_count_list:
            entity_type2id[word] = len(entity_type2id)

        return word_type2id, role_type2id, trigger_type2id, entity_type2id, singleton

    @staticmethod
    def generate_candidate_structures(instances,filter, GENERALISATION, INCLUDE_MENTION, USE_FILTER, PARTIAL_ARGUMENT_MATCHING, VERBOSE):
        model = Model()
        new_instances = model._generate_candidate_structures(instances)
        if USE_FILTER:
            new_instances = model._filter_structures(new_instances, filter, GENERALISATION, INCLUDE_MENTION, PARTIAL_ARGUMENT_MATCHING, VERBOSE)
        return new_instances

    @staticmethod
    def get_indices_with_empty_triggers(instances):
        indices = []
        for i in range(len(instances)):
            relations = instances[i][const.PRED_TRIGGERS_IDX]
            if len(relations) == 0:
                indices.append(i)
        return indices

    @staticmethod
    def remove_instances_with_empty_triggers(instances, empty_indices):
        new_instances = []
        for i in range(len(instances)):
            if i not in empty_indices:
                new_instances.append(instances[i])
        return new_instances

    @staticmethod
    def get_indices_with_empty_relations(instances):
        indices = []
        for i in range(len(instances)):
            relations = instances[i][const.PRED_RELATIONS_IDX]
            if len(relations) == 0:
                indices.append(i)
        return indices

    @staticmethod
    def remove_instances_with_empty_relations(instances, empty_indices):
        new_instances = []
        for i in range(len(instances)):
            if i not in empty_indices:
                new_instances.append(instances[i])
        return new_instances

    @staticmethod
    def get_indices_with_empty_events(instances, USE_FILTER):
        indices = []
        ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
        for i in range(len(instances)):
            events = instances[i][ind]
            all_groups_empty = True
            for e in range(len(events)):
                if len(events[e]) != 0:
                    all_groups_empty = False
                    break
            if all_groups_empty:
                indices.append(i)
        return indices

    @staticmethod
    def remove_instances_with_empty_events(instances, empty_indices):
        new_instances = []
        for i in range(len(instances)):
            if i not in empty_indices:
                new_instances.append(instances[i])
        return new_instances

    @staticmethod
    def generate_numbered_roles_for_some_events(instances, EVENTS_WITH_NUMBERED_ROLES, USE_FILTER):

        def _generate_numbered_roles(EVENTS_WITH_NUMBERED_ROLES, eventstructure, triggers):
            def extract_numbered_arguments(event):
                in_sites = []
                in_themes = []
                out_sites = []
                out_themes = []
                in_instruments = []
                out_instruments = []
                in_participants= []
                out_participants = []
                in_products = []
                out_products = []
                for i in event:
                    role = i[0][0]
                    io = i[1]
                    if role == 'Site':
                        if io == const.IN_EDGE:
                            in_sites.append(i)
                        else:
                            out_sites.append(i)
                    if role == 'Theme':
                        if io == const.IN_EDGE:
                            in_themes.append(i)
                        else:
                            out_themes.append(i)
                    if role == 'Instrument':
                        if io == const.IN_EDGE:
                            in_instruments.append(i)
                        else:
                            out_instruments.append(i)
                    if role == 'Participant':
                        if io == const.IN_EDGE:
                            in_participants.append(i)
                        else:
                            out_participants.append(i)
                    if role == 'Product':
                        if io == const.IN_EDGE:
                            in_products.append(i)
                        else:
                            out_products.append(i)
                return in_sites, in_themes, in_instruments, in_participants, out_sites, out_themes, out_instruments, out_participants, in_products, out_products

            def number(lst, i=None):
                new_list = []
                if i is None:
                    i = 0
                for e in lst:
                    role = e[0][0]
                    arg = e[0][1]
                    io = e[1]
                    if i > 0:
                        new_role = role + str(i + 1)
                    else:
                        new_role = role
                    new_list.append(((new_role, arg), io))
                    i += 1
                return new_list

            def number_participants(participants):
                lst = []
                args = []
                for i in participants:
                    role = i[0][0]
                    arg = i[0][1]
                    io = i[1]
                    num = 0
                    typ = ''
                    if arg.startswith("TR"):
                        num = arg[2:]
                        typ = "TR"
                    else:
                        num = arg[1:]
                        typ = "T"
                    heapq.heappush(args, (num, typ))

                for j in range(len(args)):
                    arg, typ = heapq.heappop(args)
                    part = ''
                    if j == 0:
                        part = "Participant"
                    else:
                        part = "Participant"+str(j+1)
                    new_edge = ((part, str(typ)+str(arg)), io)
                    lst.append(new_edge)

                return lst


            new_event_structure = collections.defaultdict(list)
            if eventstructure != {}:
                for trig, argstructures in eventstructure.items():
                    structure = argstructures[0]
                    if structure == const.EMPTY_STRUCTURE or structure == [((), 1)]:
                        new_event_structure[trig].append(const.EMPTY_STRUCTURE)
                        continue
                    trigger_type = triggers[trig][0]
                    if trigger_type in EVENTS_WITH_NUMBERED_ROLES:
                        for s in argstructures:
                            in_sites, in_themes, in_instruments, in_participants, out_sites, out_themes, out_instruments, out_participants, in_products, out_products = extract_numbered_arguments(s)

                            '''
                            PMID-3872182.a2: 
                                E9	Binding:T25 Theme:T8 Theme:T5 Theme:T7
                            Therefore, no numbering if Binding has no site.
                            Previous comment on that line:  
                            do numbering, even if there are not sites but more than one Theme
                            So that is False: but I wonder there was this comment. 
                            It must have appeared that Binding with no site and only themes have numbering in one of the samples.
                            Yes: 
                            'PMID-21536653'
                                E20: <class 'list'>: ['Binding:TR65', 'Theme:T26', 'Theme2:T27']
                            '''
                            # if len(in_sites)  == 0 and len(in_themes) > 1:
                            #     print("remove numbering here")
                            # elif len(in_themes) == 0 and len(in_instruments) > 1:
                            #     print("remove numbering here")
                            if  len(in_sites) > 0 and len(in_themes) > 1:
                                theme_combinations = itertools.permutations(in_themes, len(in_sites))
                                for comb in theme_combinations:
                                    remaining = list(set(in_themes) - set(comb))
                                    new_sites = number(in_sites)
                                    new_themes = number(comb)
                                    new_remaining = number(remaining, len(in_sites))
                                    new_structure = []
                                    for site in new_sites:
                                        new_structure.append(site)
                                    for t in new_themes:
                                        new_structure.append(t)
                                    for r in new_remaining:
                                        new_structure.append(r)
                                    for out_s in out_sites:
                                        new_structure.append(out_s)
                                    for out_t in out_themes:
                                        new_structure.append(out_t)
                                    new_event_structure[trig].append(new_structure)
                            #for planned process
                            elif trigger_type == 'Planned_process' and len(in_themes) >= 0 and len(in_instruments) == 0:
                                new_themes = number(in_themes)
                                new_structure = []
                                for theme in new_themes:
                                    new_structure.append(theme)
                                for out_s in out_themes:
                                    new_structure.append(out_s)
                                for out_t in out_instruments:
                                    new_structure.append(out_t)
                                new_event_structure[trig].append(new_structure)
                            # elif len(in_themes) >= 0 and len(in_instruments) >= 0:
                            #     instrument_combinations = itertools.permutations(in_instruments, len(in_themes))
                            #     for comb in instrument_combinations:
                            #         remaining = list(set(in_instruments) - set(comb))
                            #         new_themes = number(in_themes)
                            #         new_instruments = number(comb)
                            #         new_remaining = number(remaining, len(in_themes))
                            #         new_structure = []
                            #         for theme in new_themes:
                            #             new_structure.append(theme)
                            #         for t in new_instruments:
                            #             new_structure.append(t)
                            #         for r in new_remaining:
                            #             new_structure.append(r)
                            #         for out_s in out_themes:
                            #             new_structure.append(out_s)
                            #         for out_t in out_instruments:
                            #             new_structure.append(out_t)
                            #         new_event_structure[trig].append(new_structure)
                            elif len(in_participants) > 1:
                                lst = number_participants(in_participants)
                                new_event_structure[trig].append(lst)
                            elif len(in_products) > 1:
                                product_combinations = itertools.permutations(in_products, len(in_products))
                                for comb in product_combinations:
                                    new_products = number(comb)
                                    new_structure = []
                                    for product in new_products:
                                        new_structure.append(product)
                                    for out_p in out_products:
                                        new_structure.append(out_p)
                                    new_event_structure[trig].append(new_structure)
                            elif len(in_themes) > 1:
                                # added to handle GE13
                                theme_combinations = itertools.permutations(in_themes, len(in_themes))
                                for comb in theme_combinations:
                                    new_themes = number(comb)
                                    new_structure = []
                                    for theme in new_themes:
                                        new_structure.append(theme)
                                    for out_t in out_themes:
                                        new_structure.append(out_t)
                                    new_event_structure[trig].append(new_structure)
                            else:
                                new_event_structure[trig].append(s)
                    else:
                        new_event_structure[trig] = argstructures
            return new_event_structure

        for i in range(len(instances)):
            file_id = instances[i][const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_FILE_ID_IDX]
            # if file_id == 'PMID-1505125':
            #     print("debug numbered roles")
            triggers = instances[i][const.PRED_TRIGGERS_IDX]
            ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
            eventstructures = instances[i][ind]

            # b_flat, b_nested = main_util.count_event_flat_nested_in_event_structures(eventstructures)
            # if file_id == 'PMID-15586242':
            #     print("debug")


            new_event_structures = []
            for e in eventstructures:
                group_event_structures = _generate_numbered_roles(EVENTS_WITH_NUMBERED_ROLES, e, triggers)
                new_event_structures.append(group_event_structures)

            # a_flat, a_nested = Util.count_event_flat_nested_in_event_structures(new_event_structures)

            # if a_flat != b_flat or b_nested != a_nested:
            #     print("")
            instances[i][ind] = new_event_structures
        return instances


    def _generate_candidate_structures(self, instances):

        def _generate_candidate_structure(relations, nodes, triggers):

            def _topo_sort(data):
                """Dependencies are expressed as a dictionary whose keys are items
                and whose values are a set of dependent items. Output is a list of
                sets in topological order. The first set consists of items with no
                dependences, each subsequent set consists of items that depend upon
                items in the preceeding sets.
                From: https://pypi.python.org/pypi/toposort/1.0
                """
                # Special case empty input.
                if len(data) == 0:
                    return

                # Copy the input so as to leave it unmodified.
                data = data.copy()

                # Ignore self dependencies.
                for k, v in data.items():
                    v.discard(k)
                # Find all items that don't depend on anything.
                extra_items_in_deps = _reduce(set.union, data.values()) - set(data.keys())
                # Add empty dependences where needed.
                data.update({item: set() for item in extra_items_in_deps})
                while True:
                    ordered = set(item for item, dep in data.items() if len(dep) == 0)
                    if not ordered:
                        break
                    yield ordered
                    data = {item: (dep - ordered)
                            for item, dep in data.items()
                            if item not in ordered}
                if len(data) != 0:
                    # raise ValueError('ERROR: Cyclic dependencies exist among these items: {}'.format(
                    #     ', '.join(repr(x) for x in data.items())))
                    print('ERROR: Cyclic dependencies exist among these items, hence, excluded: {}'.format(', '.join(repr(x) for x in data.items())))

            def _create_adjacency_list(relations, triggers):
                def _trigarg_triggers(trigargs):
                    trigs = set()
                    for k, v in trigargs.items():
                        trigs.add(k)
                    return trigs

                trigargs = collections.defaultdict(set)

                for k,v in relations.items():
                    trigger = v[const.PRED_RELATION_TRIG_IDX]
                    role = v[const.PRED_RELATION_ROLE_IDX]
                    arg = v[const.PRED_RELATION_ARG_IDX]
                    trigargs[trigger].add((role, arg))

                # Add triggers with no arguments
                foundtrigs = _trigarg_triggers(trigargs)
                noevent_triggers = set(triggers) - set(foundtrigs)
                for i in list(noevent_triggers):
                    trigargs[i].add(())

                return trigargs

            def _generate_arg_combinations(trigargs):
                def _append_zeros(i, length):
                    dif = length - len(i)
                    s = i
                    for i in range(dif):
                        s = "0" + s
                    return s

                def _combinations(num):

                    total = int(math.pow(2, num))
                    c = []
                    for t in range(total):
                        c.insert(0, binary_repr(t))
                    length = len(c[0])
                    newc = []
                    for i in c:
                        if len(i) < length:
                            i = _append_zeros(i, length)
                        newc.append(i)
                    return newc

                def _generate_structure(com, v):
                    l = list(com)
                    structure = []
                    for i in range(len(v)):
                        if l[i] == str(const.IN_EDGE):  # IN edge
                            structure.append((v[i], 1))
                        else:  # OUT edge
                            structure.append((v[i], 0))
                    return structure

                trigstructures = collections.defaultdict(list)
                for k, v in trigargs.items():
                    newc = _combinations(len(v))
                    if v == {()}:
                        newc = newc[0]
                    for a in newc:
                        s = _generate_structure(a, list(v))  # convert v to list to support indexing

                        trigstructures[k].append(s)

                return trigstructures

            def _transform(trigargs):
                new_trig_args = collections.defaultdict(set)
                for trig, pairs in trigargs.items():
                        for p in pairs:
                            if len(p) != 0:
                                arg = p[1]
                                new_trig_args[trig].add(arg)
                            else:
                                new_trig_args[trig].add('')
                return new_trig_args

            def _make_sequence_of_structures(sequence, trigstructures):
                final_structures = []
                for triggers in sequence:
                    group = dict()
                    for t in triggers:
                        if t.startswith("TR"):
                            group[t] = trigstructures[t]
                    final_structures.append(group)
                return final_structures

            trigargs = _create_adjacency_list(relations, triggers)
            trigstructures = _generate_arg_combinations(trigargs)
            dict_of_sets = _transform(trigargs)
            sequence = list(_topo_sort(dict_of_sets))
            final_structures = _make_sequence_of_structures(sequence, trigstructures)
            return final_structures

        for i in range(len(instances)):
            file_id = instances[i][const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_FILE_ID_IDX]
            # if file_id == 'PMID-1388088':
            #     print("debug generate")
            relations = instances[i][const.PRED_RELATIONS_IDX]
            nodes = instances[i][const.PRED_NODES_IDX]
            triggers = instances[i][const.PRED_TRIGGERS_IDX]
            s = _generate_candidate_structure(relations, list(nodes), triggers.keys())
            instances[i].append(s) # can't use the index directly because the list must be resized first to do that
        return instances

    def _filter_structures(self, instances, filter, GENERALISATION, INCLUDE_MENTION, PARTIAL_ARGUMENT_MATCHING, VERBOSE):
        def _filter_out(trig_structures, filter, entities, triggers, generalisation, include_mention):
            '''
            Retain structures found in the filter
            '''

            def _is_IN_edge(relation):
                if relation[1] == 1:  # use IN edges only to define a structure
                    return True
                else:
                    return False

            def _convert_to_list(perm):
                perms = []
                for i in perm:
                    tl = []
                    for j in i:
                        tl.append(j)
                    perms.append(tl)
                return perms

            def _remove_mentions(inedges):
                newinedges = []
                for i in inedges:
                    role = i[0]
                    type = i[1][0]
                    edge = (role, type)
                    newinedges.append(edge)
                return newinedges

            def _remove_labels(inedges):
                newinedges = []
                for i in inedges:
                    role = i[0][0]
                    type = i[0][1]
                    edge = (role, type)
                    newinedges.append(edge)
                return newinedges

            def _into_list(inedges):
                lst = []  # permits repeated arguments
                for j in inedges:
                    if len(j[0]) == 0:
                        lst.append('')
                    else:
                        role = j[0][0]
                        type = j[0][1]
                        joined = role + "__" + type
                        lst.append(joined)
                return lst

            def _transform_inedges(inedges, triggers, entities, generalisation,
                                                         include_mention, PARTIAL_ARGUMENT_MATCHING):
                new_inedges = []
                for i in inedges:
                    rel = i[0]
                    if len(rel) == 0:
                        new_inedges.append((rel, i[1]))
                    else:
                        id = rel[1]
                        type = ''
                        if id in triggers:
                            type = triggers[id][0]
                        elif id in entities:
                            type = entities[id][0]
                        new_type = Util.extract_category(type, generalisation, const.TYPE_GENERALISATION, VERBOSE)
                        new_inedges.append(((rel[0], new_type), i[1]))
                return new_inedges

            def has_site_without_theme(inedges):
                roles = []
                flag = False
                for i in inedges:
                    rel = i[0]
                    role = rel[0]
                    roles.append(role)
                if 'Site' in roles:
                    if 'Theme' not in roles:
                        flag = True
                    else:
                        flag = False

                return flag #or found

            def has_csite_without_cause(inedges):
                roles = []
                flag = False
                for i in inedges:
                    rel = i[0]
                    role = rel[0]
                    roles.append(role)
                if 'CSite' in roles:
                    if 'Cause' not in roles:
                        flag = True
                    else:
                        flag = False
                return flag

            newtrigstructures = collections.defaultdict(list)

            for trig_id, structures in trig_structures.items():
                trig_type = triggers[trig_id][0]

                # if trig_type == 'Localization':
                #     print("debug")

                # skip because filter does not contain this trigger type
                if trig_type not in filter[generalisation]:
                    continue

                # if len(structures) == 2:
                #     if structures[0][0][0] == ():
                #         str_len = '0'
                #         if str_len in filter[generalisation][trig_type][0]:
                #             if len(filter[generalisation][trig_type][0][str_len]) > 0:
                #                 newtrigstructures[trig_id].append(const.EMPTY_STRUCTURE)
                #                 continue
                if structures == [[((), 1)]] or structures == [[()]]:
                    str_len = '0'
                    if str_len in filter[generalisation][trig_type][0]:
                        if len(filter[generalisation][trig_type][0][str_len]) > 0:
                            newtrigstructures[trig_id].append(const.EMPTY_STRUCTURE)
                else:
                    for structure in structures:
                        inedges = []

                        # remove the OUT edges
                        for relation in structure:
                            try:

                                if _is_IN_edge(relation):
                                    inedges.append(relation)
                            except:
                                print("")

                        # if len(inedges) != 0:

                            # transform inedges
                        inedges = _transform_inedges(inedges, triggers, entities, generalisation,
                                                     include_mention, PARTIAL_ARGUMENT_MATCHING)

                        # match the structure permutation in the filter
                        found = False
                        if trig_type in filter[generalisation]:

                            #checking of site and theme, csite and cause
                            if has_csite_without_cause(inedges):
                                found = False
                            else:
                                if trig_type != 'Mutation' and has_site_without_theme(inedges):
                                    found = False
                                else:
                                    list_inedges = _into_list(inedges)

                                    multiset_inedges = collections.Counter(list_inedges)

                                    len_multiset_inedges = sum(multiset_inedges.values())

                                    str_len = str(len_multiset_inedges)
                                    if str_len in filter[generalisation][trig_type][0]:
                                        if len(filter[generalisation][trig_type][0][str_len]) > 0:
                                            for s in filter[generalisation][trig_type][0][str_len][0]:
                                                s1 = collections.Counter(s)
                                                if s1 == multiset_inedges:
                                                    found = True
                                                    # if len_multiset_inedges == 2 and len(list_inedges) == 1:
                                                    #     print("debug filter")
                                                    break

                                    #additional programming check using rules
                                    # if not found:
                                    #     found = filter_programmatically(inedges, trig_type)


                        if found:
                            newtrigstructures[trig_id].append(structure)
            return newtrigstructures



        def filter_programmatically(inedges, trig_type):
            def is_valid_regulation_type(inedges):
                has_one_theme = False
                has_one_cause = False
                if len(inedges) == 2:
                    first = inedges[0][0]
                    second = inedges[1][0]
                    first_role = first[0]
                    second_role = second[0]
                    if first_role != second_role:
                        if first_role == 'Theme' or second_role == 'Theme':
                            has_one_theme = True
                        if first_role == 'Cause' or second_role == 'Cause':
                            has_one_cause = True
                elif len(inedges) == 1:
                    first = inedges[0][0]
                    first_role = first[0]
                    if first_role == 'Theme':
                        has_one_theme = True
                    else:
                        has_one_theme = False
                return has_one_theme, has_one_cause

            def is_valid_plannedprocess(inedges):
                flag = True
                instrument_args = []
                for i in inedges:
                    rel = i[0]
                    role = rel[0]
                    if role not in ['Instrument', 'Theme']:
                        flag = False
                        break
                    if role == 'Instrument':
                        arg = rel[1]
                        instrument_args.append(arg)
                for s in instrument_args:
                    if s not in const.ENTITY_LIST:
                        flag = False
                        break
                return flag

            def is_valid(inedges, trigtype):
                flag = False
                if trig_type == 'Metastasis':
                    if len(inedges) == 1:
                        rel = inedges[0][0]
                        role = rel[0]
                        arg = rel[1]
                        if role == 'ToLoc' and arg == 'Cell':
                            flag = True
                elif trig_type =='Binding':
                    bindings = [['Theme_Simple_chemical', 'Theme_Multi-tissue_structure'],
                                ['Theme_Gene_or_gene_product', 'Site_DNA_domain_or_region']]
                    event = []
                    if len(inedges) == 2:
                        first = inedges[0]
                        rel1 = first[0]
                        role1 = rel1[0]
                        arg1 = rel1[1]
                        event.append(role1+"_"+arg1)
                        second = inedges[1]
                        rel2 = second[0]
                        role2 = rel2[0]
                        arg2 = rel2[1]
                        event.append(role2 + "_" + arg2)
                    binding1 = collections.Counter(bindings[0])
                    binding2 = collections.Counter(bindings[1])
                    event_set = collections.Counter(event)
                    if event_set == binding1 or event_set == binding2:
                        flag = True
                elif trig_type == 'Cell_transformation':
                    celltransfo = ['AtLoc_Cell', 'Theme_Cell']
                    event = []
                    if len(inedges) == 2:
                        first = inedges[0]
                        rel1 = first[0]
                        role1 = rel1[0]
                        arg1 = rel1[1]
                        event.append(role1 + "_" + arg1)
                        second = inedges[1]
                        rel2 = second[0]
                        role2 = rel2[0]
                        arg2 = rel2[1]
                        event.append(role2 + "_" + arg2)
                    celltrans = collections.Counter(celltransfo)
                    event_set = collections.Counter(event)
                    if celltrans == event_set:
                        flag = True
                return flag

            found = False
            if trig_type in ['Regulation', 'Positive_regulation', 'Negative_regulation']:
                '''
                Theme(Any), Cause?(Any) 
                '''
                has_one_theme, has_one_cause = is_valid_regulation_type(inedges)
                if has_one_theme or has_one_cause:
                    found = True
            elif trig_type == 'Planned_process':
                found = is_valid_plannedprocess(inedges)

            else:
                found = is_valid(inedges, trig_type)

            return found




        def _extract_events_with_no_arguments(instance_events):
            trigs = set()
            no_arg_trigs = set()
            for level in range(len(instance_events)):
                for trig, structures in instance_events[level].items():
                    for structure in structures:
                        for relation in structure:
                            if relation[0] == ():
                                break
                            arg = relation[0][1]
                            if arg.startswith("TR"):
                                if arg not in trigs:
                                    no_arg_trigs.add(arg)
                    trigs.add(trig)
            return no_arg_trigs

        for i in range(len(instances)):
            file_id = instances[i][const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_FILE_ID_IDX]

            groups = instances[i][const.PRED_CAND_STRUCTURES_IDX]
            triggers = instances[i][const.PRED_TRIGGERS_IDX]
            entities = instances[i][const.PRED_ENTITIES_IDX]

            no_arg_triggers = _extract_events_with_no_arguments(groups)

            if len(no_arg_triggers) > 0:
                empty_events = collections.defaultdict(list)
                for trig in no_arg_triggers:
                    empty_events[trig].append(const.EMPTY_STRUCTURE)
                groups.insert(0, empty_events)

            # add events with no arguments and filter them here
            # then remove the one in predict

            # if file_id == 'PMID-18559522':
            #     print("debug filter")


            new_groups = []
            for g in range(len(groups)):
                group = groups[g]
                new_group = _filter_out(group, filter, entities, triggers, GENERALISATION, INCLUDE_MENTION)
                new_groups.append(new_group)
            instances[i].append(new_groups)
        return instances

    @staticmethod
    def instances_to_ids(instances, word_type2id, role_type2id, trigger_type2id,entity_type2id, USE_FILTER, VERBOSE):

        def _instance2id(instance, word_type2id, role_type2id, trigger_type2id,entity_type2id):
            def getRelID(i, trigs, ents):
                a = i[0]
                b = i[2]

                a_id = ''
                b_id = ''

                for k, v in trigs.items():
                    if a == v:
                        a_id = k
                    if b == v:
                        b_id = k

                for k, v in ents.items():
                    if a == v:
                        a_id = k
                    if b == v:
                        b_id = k
                return a_id, b_id

            def getRelsID(relations, ents, trigs):
                newrels = []
                for i in relations:
                    role = i[1]
                    a_id, b_id = getRelID(i, trigs, ents)
                    s = a_id + "_" + b_id
                    newrels.append([s, role])
                return newrels

            def getID(i, trigs, ents):
                id = ''
                for k, v in trigs.items():
                    if i == v:
                        id = k
                for k, v in ents.items():
                    if i == v:
                        id = k
                return id

            def getIDs(nodes, trigs, ents):
                dic = set()
                for i in nodes:
                    id = getID(i, trigs, ents)
                    dic.add(id)
                return dic

            def _structure_2ids(eventstructure, roletype2id, ents, trigs):
                new_event_structure = collections.defaultdict(list)
                if eventstructure == {}:
                    eventstructure = new_event_structure
                for trig, argstructures in eventstructure.items():
                    for structure in argstructures:
                        if structure == const.EMPTY_STRUCTURE or structure == [((),1)]:
                            new_event_structure[trig].append(const.EMPTY_STRUCTURE)
                            break
                        new_structure = []
                        for edge in structure:
                            relation = edge[0]
                            roleType = relation[0]

                            ending1 = roleType[-1]
                            if ending1.isdigit():  # remove the numbering
                                roleType = roleType[0:-1]


                            if roleType not in roletype2id:
                                print("Not Supported: Missing role types in events", roleType)
                                role = 0
                            else:
                                role = roletype2id[roleType]
                            io = edge[1]

                            new_edge = ((role, relation[1]), io)

                            new_structure.append(new_edge)
                        new_event_structure[trig].append((structure, new_structure))
                return new_event_structure

            def _text2id(text, word2id):
                ids = []
                count = 0
                word_count = 0
                for k in text:
                    if k.rstrip() in word2id:
                        ids.append(word2id[k.rstrip()])
                    else:
                        if VERBOSE:
                            print("Error: using UNK token ID as word was not found, ", k)
                        ids.append(word2id[const.UNK_TOKEN])
                        count += 1
                    word_count += 1
                return ids, count, word_count

            def arg2id(v, argtype2id, word2id):
                if v[0] not in argtype2id:
                    if VERBOSE:
                        print("Not supported: Missing types for Entities/Triggers. Assigning random.", v[0])
                    val = 0
                else:
                    val = argtype2id[v[0]]
                ids, count, word_count = _text2id(v[3].split(), word2id)
                return [val, v[1], v[2], ids , count, word_count]

            def _args2id(arguments, argtype2id, word2id):
                arg = dict()
                total_cnt = 0
                total_word_cnt = 0
                for k, v in arguments.items():
                    arg[k] = arg2id(v, argtype2id, word2id)[0:4]
                    count = arg2id(v, argtype2id, word2id)[4]
                    total_word_cnt += arg2id(v, argtype2id, word2id)[5]
                    total_cnt += count
                return arg, total_cnt, total_word_cnt

            def _relations_2id(relations, roletype2id):
                new_relations = dict()
                for id, triplet in relations.items():
                    role = triplet[0]
                    if role not in role_type2id:
                        role_id = 0
                        print("Not supported: Missing types for Roles.", role)
                    else:
                        role_id = role_type2id[role]
                    new_triplet = [role_id, triplet[1], triplet[2]]
                    new_relations[id] = new_triplet
                return new_relations



            sentence, unk_word_count, txt_word_count  = _text2id(instance[const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_IDX].split(), word_type2id)
            file_id = instance[const.PRED_SENTENCE_INFO_IDX][const.PRED_SENTENCE_FILE_ID_IDX]
            sentence = [sentence, file_id]

            triggers = instance[const.PRED_TRIGGERS_IDX]
            new_triggers, unk_trig_count, trig_word_count = _args2id(triggers, trigger_type2id, word_type2id)

            entities = instance[const.PRED_ENTITIES_IDX]
            new_entities, unk_ent_count, ent_word_count = _args2id(entities, entity_type2id, word_type2id)

            relations = instance[const.PRED_RELATIONS_IDX]
            new_relations = _relations_2id(relations, role_type2id)

            ind = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
            eventstructures = instance[ind]

            unk_count = unk_word_count + unk_trig_count + unk_ent_count
            total_word_count = txt_word_count + trig_word_count + ent_word_count

            new_event_structures = []
            for e in eventstructures:
                group_event_structures = _structure_2ids(e, role_type2id, new_entities, new_triggers)
                new_event_structures.append(group_event_structures)

            newinstance = [sentence, new_entities, new_triggers, new_relations, new_event_structures]
            return newinstance, unk_count, total_word_count

        new_instances = []
        total_unk_count = 0
        total_word_count = 0
        for i in range(len(instances)):
            instance_id, unk_count, word_count = _instance2id(instances[i], word_type2id, role_type2id,trigger_type2id,entity_type2id)
            total_unk_count += unk_count
            total_word_count += word_count
            new_instances.append(instance_id)
        return new_instances, total_unk_count, total_word_count

    @staticmethod
    def predictor(n_word_types, n_trig_types, n_role_types, n_entity_types, trigger_type2id, entity_type2id, DIM_EMBED,
                  DIM_EVENT,DIM_BILSTM, DIM_TRIG_TYPE,DIM_ROLE_TYPE,DIM_ARG_TYPE,DIM_IO,DROPOUT, REPLACE_TYPE,
                  GENERALISATION, THRESHOLD):

        model = EventStructureClassifier(n_word_types, n_trig_types, n_role_types, n_entity_types, trigger_type2id, entity_type2id,
                                         DIM_EMBED, DIM_EVENT,DIM_BILSTM, DIM_TRIG_TYPE,DIM_ROLE_TYPE,DIM_ARG_TYPE,DIM_IO,DROPOUT,
                                         REPLACE_TYPE, GENERALISATION, THRESHOLD)
        return model

    @staticmethod
    def generate_prediction_file(instances, predictions, OUTPUT_DIR, all_file_ids, SPECIAL_ENTITIES, USE_FILTER):

        def _extract_event_structures_and_map_to_file_ids(predictions, instances):

            def _extract_event_trig_structures(predictions):

                def _convert_to_multiset(structure):
                    l = []
                    for rel in structure:
                        role = rel[0][0]
                        type = rel[0][1]
                        io = rel[1]
                        if io == const.IN_EDGE:
                            l.append(role + "__" + type)
                    multiset = collections.Counter(l)
                    return multiset

                # def _extract_events(instance):
                #     structures_idx = const.PRED_FILTERED_STRUCTURES_IDX if USE_FILTER else const.PRED_CAND_STRUCTURES_IDX
                #     groups = instance[structures_idx]
                #     trig_structures = collections.defaultdict(list)
                #     for g in groups:
                #         for trig, structures in g.items():
                #             for structure in structures:
                #
                #                 trig_structures[trig].append((multiset_struct, structure))
                #     return trig_structures

                def _extract_predictions(predictions):
                    trig_structures = collections.defaultdict(list)
                    for trig, structures in predictions.items():
                        for structure in structures:
                            struct = structure[0]
                            pred = structure[1]
                            multiset_struct = _convert_to_multiset(struct)
                            trig_structures[trig].append((multiset_struct, pred))
                    return trig_structures

                trig_structures =  collections.defaultdict(list)
                # preds = _extract_predictions(predictions)
                #
                # for trig, structures in events.items():
                #     multiset = structures[0]
                #     struct = structures[1]
                #
                #     pred_structures = preds[trig]
                #     # check if for all sub-events they were predicted then consider this as predicted
                #     is_event = const.IS_EVENT
                #     for p in pred_structures:
                #         p_multiset = p[0]
                #         p_pred = p[1]
                #         if multiset == p_multiset:
                #             if p_pred != const.IS_EVENT:
                #                 is_event = const.IS_NON_EVENT
                #                 break
                #     if is_event:
                #         trig_structures[trig].append(struct)


                for trig, structures in predictions.items():
                    for s in range(len(structures)):
                        prediction = structures[s][1]
                        if prediction == const.IS_EVENT:
                            pred_structure = structures[s][3]
                            trig_structures[trig].append(pred_structure)
                return trig_structures

            def _combine_trig_structures(cur_trig_structures, trig_structures):
                for trig, structures in cur_trig_structures.items():
                    for structure in structures:
                        trig_structures[trig].append(structure)
                return trig_structures


            file_id_events_mapping = dict()

            assert len(predictions) == len(instances), "ERROR: len(predictions) != len(instances)"
            for p in range(len(predictions)):
                file_id = instances[p][const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_FILE_ID_IDX]


                trig_structures = _extract_event_trig_structures(predictions[p])

                cur_file_id_entities = dict()
                cur_file_id_triggers = dict()
                cur_trig_structures = collections.defaultdict(list)

                file_id_entities = instances[p][const.IDS_ENTITIES_IDX]
                file_id_triggers = instances[p][const.IDS_TRIGGERS_IDX]

                if file_id in file_id_events_mapping:
                    cur_file_id_entities = file_id_events_mapping[file_id][0]
                    cur_file_id_triggers = file_id_events_mapping[file_id][1]
                    cur_trig_structures = file_id_events_mapping[file_id][2]

                combined_entities = {**file_id_entities, **cur_file_id_entities}
                combined_triggers = {**file_id_triggers, **cur_file_id_triggers}

                combined_trig_structures = _combine_trig_structures(cur_trig_structures, trig_structures)

                triplet = [combined_entities, combined_triggers, combined_trig_structures]
                file_id_events_mapping[file_id] = triplet

            return file_id_events_mapping


        def _convert_to_event_structures_and_assign_ids(file_id_events_mapping):

            def _create_event_ids(triggers, input_events):

                def _extract_IN_relations(structure):
                    IN_structure = []
                    for relation in structure:
                        if len(relation) != 0:
                            role = relation[0][0]
                            trigger = relation[0][1]
                            io = relation[1]
                            if io == const.IN_EDGE:
                                IN_structure.append((role, trigger))
                    return IN_structure

                event_id = 1
                events = dict()
                trigger_event_mapping = collections.defaultdict(list)
                for trig, structures in input_events.items():
                    for structure in structures:
                        if len(structure) == 0:
                            IN_structure = [trig]
                        else:
                            IN_structure = _extract_IN_relations(structure)
                            IN_structure.insert(0, trig)
                        new_event = "E" + str(event_id)
                        events[new_event] = IN_structure
                        trigger_event_mapping[trig].append(new_event)
                        event_id += 1
                return events, trigger_event_mapping

            def _replace_trigger_arg_with_event_ids(events_with_ids, trigger_event_mapping):

                def search_structure(same_structure, structure):
                    idx = -1
                    for id, structures in same_structure.items():
                        structure1 = structures[0][1] #just get the first structure
                        structure1_ = collections.Counter(structure1)
                        structure_ = collections.Counter(structure)
                        if structure1 == structure:
                            idx = id
                            break
                    return idx

                def extract_event_ids(arg_structures):
                    eventids = []
                    for struct in arg_structures:
                        eventids.append(struct[0])
                    return eventids
                #gather all with same trigger
                same_trigger = collections.defaultdict(list)
                for event_id, structure in events_with_ids.items():
                    trigger = structure[0]
                    event_structure = structure[1:]
                    same_trigger[trigger].append([event_id, event_structure])
                #gather all of same structures
                same_structure = collections.defaultdict(list)
                ctr = 0
                for event_id, structure in events_with_ids.items():
                    idx = search_structure(same_structure, structure)
                    if idx == -1:
                        same_structure[ctr].append([event_id, structure])
                        ctr += 1
                    else:
                        same_structure[idx].append([event_id, structure])
                #checking
                for id, structures in same_structure.items():
                    cnt = len(structures)
                    exp_cnt = 1
                    structure = structures[0][1][1:]
                    for rel in structure:
                        arg = rel[1]
                        if arg in same_trigger:
                            cnt_in_trigger = len(same_trigger[arg])
                            exp_cnt *= cnt_in_trigger
                    # if cnt != exp_cnt:
                    #     print("debug")
                    # assert cnt <= exp_cnt, "Error: more sub-event structures than the main event structures"

                #generate the combinations
                #TODO: By default this gets the first n sub-event if cnt < exp_cnt
                '''
                In theory, the main event structures should be equal to the number of sub-events but 
                if there are less main events than the sub-events, then we assume that it is safe
                to take the first n sub-events as the argument of the main events
                See example in notebook 2018-08-06
                '''
                new_events = dict()
                for id, structures in same_structure.items():
                    #create combinations
                    structure = structures[0][1][1:]
                    combinations = []
                    for rel in structure:
                        arg = rel[1]
                        if arg in same_trigger:
                            arg_structures = same_trigger[arg]
                            eventids = extract_event_ids(arg_structures)
                            combinations.append(eventids)
                    actual_combinations = []
                    for comb in itertools.product(*combinations):
                        l = list(comb)
                        actual_combinations.append(l)

                    for s in range(len(structures)):
                        try:
                            comb = actual_combinations[s]
                        except:
                            trigger = structures[s][1][0]
                            event_id = structures[s][0]
                            new_structure = []
                            new_events[event_id] = [trigger, new_structure]
                            continue
                        #extract triggers
                        trigger = structures[s][1][0]
                        event_id = structures[s][0]
                        triggers = []
                        structure = structures[s][1][1:]
                        for rel in structure:
                            arg = rel[1]
                            if arg in same_trigger:
                                triggers.append(arg)

                        # create mapping
                        mapping = dict()
                        for c in comb:
                            for g in triggers:
                                if c in trigger_event_mapping[g]:
                                    mapping[g] = c
                        new_structure = []
                        for rel in structure:
                            role = rel[0]
                            arg = rel[1]
                            if arg in mapping:
                                arg = mapping[arg]
                            new_structure.append((role, arg))
                        new_events[event_id] = [trigger, new_structure]
                    # exp_events = 0
                    # for relation in event_structure:
                    #     arg = relation[1]
                    #     if arg.startswith("TR"):
                    #         if arg in trigger_event_mapping:
                    #             arg_ids = trigger_event_mapping[arg]
                    #             exp_events += len(arg_ids)


                    #
                    # combinations = []
                    # for relation in structure[1:]:
                    #     arg = relation[1]
                    #     if arg.startswith("TR"):
                    #         if arg in trigger_event_mapping:
                    #             arg_ids = trigger_event_mapping[arg]
                    #             combinations.append(arg_ids)
                    #
                    # list_combs = []
                    # for comb in itertools.product(*combinations):
                    #     l = list(comb)
                    #     list_combs.append(l)
                    #
                    # for relation in structure[1:]:
                    #     role = relation[0]
                    #     arg = relation[1]
                    #     if arg.startswith("TR"):
                    #         if arg in trigger_event_mapping:
                    #             arg_ids = trigger_event_mapping[arg]
                    #             try:
                    #                 arg = arg_ids[idx]
                    #             except:
                    #                 arg = arg_ids[0] #structure is repeated but only one sub-event
                    #                 print("debug")
                    #     new_structure.append((role, arg))
                    # new_events[event_id] = [trigger, new_structure]
                return new_events

            file_id_with_new_events = []
            for file_id, defns in file_id_events_mapping.items():
                entities = defns[0]
                triggers = defns[1]
                events = defns[2]
                events_with_ids, trigger_event_mapping = _create_event_ids(triggers, events)
                new_events = _replace_trigger_arg_with_event_ids(events_with_ids, trigger_event_mapping)
                file_id_with_new_events.append((file_id, entities, triggers, new_events))
            return file_id_with_new_events

        def _make_readable(file_id_with_new_events, id2role_type):

            def _transform_events(events, triggers, id2role_type):
                def _transform_structure(structure, triggers, id2role_type):
                    new_structure = []
                    trigger = structure[0]
                    trig_type = triggers[trigger][0]

                    relations = structure[1]
                    new_relations = []
                    for relation in relations:
                        role = relation[0]
                        arg = relation[1]
                        role_type = id2role_type[role]
                        new_relations.append((role_type, arg))
                    new_structure.append(((trig_type, trigger), new_relations))
                    return new_structure


                new_events = dict()
                for event_id, structure in events.items():
                    new_structure = _transform_structure(structure, triggers, id2role_type)
                    new_events[event_id] = new_structure
                return new_events

            readable = []
            for i in file_id_with_new_events:
                file_id = i[0]
                entities = i[1]
                triggers = i[2]
                events = i[3]
                new_events = _transform_events(events, triggers, id2role_type)
                readable.append((file_id, entities, triggers, new_events))
            return readable

        def _write_to_file(readable, OUTPUT_DIR, all_file_ids, SPECIAL_ENTITIES):

            pred_dir = os.listdir(OUTPUT_DIR)
            for f in pred_dir:
                os.remove(os.path.join(OUTPUT_DIR, f))

            present_files = []

            for file_id, contents in readable.items():

                triggers = contents[1]
                events = contents[2]
                ne = contents[3]
                present_files.append(file_id)

                file = open(OUTPUT_DIR + file_id + const.OUTPUT_EXTENSION, 'w')
                for trig_id, content in triggers.items():
                    trig_type = content[0]
                    offset_from = content[1]
                    offset_to = content[2]
                    mention_words = content[3]
                    line = trig_id + "\t" + trig_type + " " + str(offset_from) + " " + str(offset_to) + "\t" + mention_words.rstrip()
                    file.write(line+"\n")

                for event_id, structure in events.items():
                    trig_id = structure[0]
                    trig_type = triggers[trig_id][0]
                    arg_str = ''
                    if len(structure) > 1:
                        for relation in structure[1:]:
                            relation = relation[0]+":"+relation[1] + " "
                            arg_str += relation
                    line = event_id + "\t" + trig_type + ":" + trig_id + " " + arg_str.rstrip()
                    file.write(line + "\n")

                for id, defn in ne.items():
                    type = defn[0]
                    offset_from = defn[1]
                    offset_to = defn[2]
                    mention_words = defn[3]
                    line = id + "\t" + type + " " + str(offset_from) + " " + str(
                        offset_to) + "\t" + mention_words.rstrip()
                    file.write(line + "\n")

                file.close()


            missing_files = set(all_file_ids) - set(present_files)
            for f in missing_files:
                file = open(OUTPUT_DIR + f + const.OUTPUT_EXTENSION, 'w')
                file.close()


        def _adjust_ids(readable):
            def _get_max_min_id(triggers):
                max = -1
                min = const.MAX_INT_SIZE_FOR_IDS
                for key, _ in triggers.items():
                    num = int(key[2:])
                    if num > max:
                        max = num
                    if num < min:
                        min = num
                return max, min

            def _change_entity_ids(events, file_id, file_id_trigger_ids):
                '''
                Change the entity ids that are greater than the min trigger id for this file
                and change it to start from max trig id  + 1.
                :param events:
                :param file_id_trigger_ids:
                :return:
                '''

                def _change_structure(structure, min, cur_ne_id, ne_ids_mapping):

                    new_structure = []
                    new_structure.append(structure[0])
                    actual_structure = structure[1]
                    event_structure = []
                    for arg in actual_structure:
                        role = arg[0]
                        arg_id = arg[1]
                        if arg_id.startswith("TR"):
                            new_arg_id = arg_id
                        elif arg_id.startswith("T"):
                            if arg_id in ne_ids_mapping:#use the previously set one
                                new_arg_id = ne_ids_mapping[arg_id]
                            else:
                                try:
                                    ent_num = int(arg_id[1:])
                                except:
                                    print("debug change structure")
                                if ent_num >= min: # this should be changed
                                    cur_ne_id += 1
                                    ent_num = cur_ne_id
                                    new_arg_id = "T" + str(ent_num)
                                    ne_ids_mapping[arg_id] = new_arg_id
                                else:
                                    new_arg_id = arg_id
                        else: # includes Event and the rest
                            new_arg_id = arg_id
                        new_arg = (role, new_arg_id)
                        event_structure.append(new_arg)
                    new_structure.append(event_structure)
                    return new_structure, cur_ne_id


                values = file_id_trigger_ids[file_id]

                max = values[0]
                min = values[1]

                cur_ne_id = max
                new_events = dict()
                ne_ids_mapping = dict()
                for event_id, structure in events.items():
                    new_structure, cur_ne_id = _change_structure(structure, min, cur_ne_id, ne_ids_mapping)
                    new_events[event_id] = new_structure
                return new_events, ne_ids_mapping

            file_id_trigger_ids = dict()
            for i  in range(len(readable)):
                file_id = readable[i][0]
                triggers =readable[i][2]
                max, min = _get_max_min_id(triggers)
                if file_id in file_id_trigger_ids:
                    old_values = file_id_trigger_ids[file_id]
                    old_max = old_values[0]
                    old_min = old_values[1]
                    if max > old_max:
                        new_max = max
                    else:
                        new_max = old_max
                    if min < old_min:
                        new_min = min
                    else:
                        new_min = old_min
                    new_values = [new_max, new_min]
                    file_id_trigger_ids[file_id] = new_values
                else:
                    file_id_trigger_ids[file_id] = [max, min]

            more_readable = []
            for i in range(len(readable)):
                file_id = readable[i][0]
                if file_id == "PMC-3245220-13-Materials_and_Methods":
                    print("debug")
                entities = readable[i][1]
                triggers = readable[i][2]
                events = readable[i][3]
                new_events, ne_ids_mapping = _change_entity_ids(events, file_id, file_id_trigger_ids)
                more_readable.append((file_id, entities, triggers, new_events, ne_ids_mapping))

            return more_readable

        def _fix_ne_id(readable, SPECIAL_ENTITIES):
            def _get_max_trigger_id(triggers):
                max_id = 0
                for trig_id, content in triggers.items():
                    id = int(trig_id[2:])
                    if id > max_id:
                        max_id = id
                return max_id

            def _renumber_ne(ne, entities, max_trigger_id, SPECIAL_ENTITIES):

                #collect the special entities
                ne_dict = dict()
                for id, value in entities.items():
                    type = value[0]
                    if type in SPECIAL_ENTITIES:
                        offset_from = value[1]
                        offset_to = value[2]
                        mention_words = value[3]
                        # line = id + "\t" + type + " " + str(offset_from) + " " + str(offset_to) + "\t" + mention_words.rstrip()
                        # file.write(line + "\n")
                        ne_dict[id] = [type, str(offset_from), str(offset_to), mention_words.rstrip()]


                #get max id in ne
                max_id = max_trigger_id
                if ne:
                    for id, value in ne.items():
                        val = int(value[1:])
                        if val > max_id:
                            max_id = val

                new_ne_dict = dict()
                new_ne_id_mapping = dict()
                for id, defn in ne_dict.items():
                    old_id = id
                    if id in ne:
                        id = ne[id]
                    else:
                        id = max_id + 1
                        max_id = id
                        id = "T"+str(id)
                    new_ne_dict[id] = defn
                    new_ne_id_mapping[old_id] = id

                # print("temp")
                # # replace with the new max_trigger_id + 1
                # cur_id = max_trigger_id +  1
                # ne_id_mapping = dict() #stores ne old id and new id mapping
                # new_ne_dict = dict()
                # for old_id, defn in ne_dict.items():
                #     #set id to above trigger number which here is cur_id
                #     id = "T"+str(cur_id)
                #     # update cur id to next number
                #     cur_id += 1
                #     #retrieve defn
                #     # defn = entities[old]
                #     type = defn[0]
                #     offset_from = defn[1]
                #     offset_to = defn[2]
                #     mention_words = defn[3]
                #     #set new ne with the new id and its defn
                #     new_ne_dict[id] = [type, str(offset_from), str(offset_to), mention_words.rstrip()]
                #     #store old and new id mapping for events
                #     # if old_id in ne:
                #     ne_id_mapping[old_id] = id
                #
                # #adjust with respect to ne
                # temp_mapping = dict()
                # for id, value in ne.items():
                #     temp = ne_id_mapping[id]
                #     temp_mapping[value] = temp
                #
                # new_ne_id_mapping = dict()
                # for id, value in ne_id_mapping.items():
                #     if value in temp_mapping:
                #         value = temp_mapping[value]
                #     if id in ne:
                #         value = ne[id]
                #     new_ne_id_mapping[id] = value
                # #update ne_dict
                # for id, value in temp_mapping.items():
                #     temp = new_ne_dict[id]
                #     temp2 = new_ne_dict[value]
                #     new_ne_dict[value] = temp
                #     new_ne_dict[id] = temp2

                return new_ne_dict, new_ne_id_mapping

            def _change_event_arg_ids(events, ne_id_mapping):
                new_events = dict()
                for event_id, structure in events.items():
                    trig_id = structure[0]
                    new_structure = [trig_id]
                    for relation in structure[1]:
                        arg_role = relation[0]
                        arg_type = relation[1]
                        if arg_type in ne_id_mapping:
                            arg_type = ne_id_mapping[arg_type]
                        new_relation = (arg_role, arg_type)
                        new_structure.append(new_relation)
                    new_events[event_id] = new_structure
                return new_events

            by_file_id = dict()
            for i in range(len(readable)):
                file_id = readable[i][0]
                entities = readable[i][1]
                triggers = readable[i][2]
                events = readable[i][3]
                ne_mapping = readable[i][4]
                by_file_id[file_id] = [entities, triggers, events, ne_mapping]
            new_file_id = dict()
            for file_id, contents in by_file_id.items():
                if file_id == "PMC-3245220-13-Materials_and_Methods":
                    print("debug")
                entities = contents[0]
                triggers = contents[1]
                events = contents[2]
                ne = contents[3]
                max_trigger_id = _get_max_trigger_id(contents[1])
                #some nes are in the entities already but no in ne so no if here
                origne = ne
                ne, ne_id_mapping = _renumber_ne(ne, entities,max_trigger_id, SPECIAL_ENTITIES)
                if not ne:
                    origne = {v: k for k,v in origne.items()}
                    ne_id_mapping = origne
                events = _change_event_arg_ids(events, ne_id_mapping)
                new_contents = [entities, triggers, events, ne]
                new_file_id[file_id] = new_contents
            return new_file_id

        # id2role_type = {v: k for k, v in role_type2id.items()}

        file_id_events_mapping = _extract_event_structures_and_map_to_file_ids(predictions, instances)

        file_id_with_new_events = _convert_to_event_structures_and_assign_ids(file_id_events_mapping)

        # readable = _make_readable(file_id_with_new_events, id2role_type)

        more_readable = _adjust_ids(file_id_with_new_events)
        ready_for_writing = _fix_ne_id(more_readable, SPECIAL_ENTITIES)

        # ready_for_writing = dict()
        # for i in file_id_with_new_events:
        #     file_id = i[0]
        #     entities = i[1]
        #     triggers = i[2]
        #     events = i[3]
        #     new_events = dict()
        #     for id, st in events.items():
        #         trig = st[0]
        #         args = st[1]
        #         joined = [trig]
        #         for a in args:
        #             joined.append((a))
        #         new_events[id] =  joined
        #     ready_for_writing[file_id] = [entities, triggers, new_events]

        # events_count = Util.count_written_events(more_readable)
        # events_count = Util.count_written_events(ready_for_writing)
        # print("events_count:", events_count)

        _write_to_file(ready_for_writing, OUTPUT_DIR, all_file_ids, SPECIAL_ENTITIES)
        # print("test")