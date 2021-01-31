#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019 18:20
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""

import src.util.dataprocessor as dp
import src.util.goldprocessor as gdp

def exclude(instances, idx):
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

def load(dir, ner_dir, rel_dir, is_emo, excl, incl_mention, generalisation,
                 use_filter, partial_arg_matching, templates, gold=None):
    # load file
    instances, all_file_ids = dp.load(dir, ner_dir, rel_dir, is_emo)

    # exclude same instances - see yaml file
    if excl:
        instances = exclude(instances, excl)
        if gold:
            gold = exclude(gold, excl)

    # remove instances with no triggers
    trig_empty_indices = dp.get_indices_with_empty_triggers(instances)
    instances = dp.remove_instances_with_empty_triggers(instances, trig_empty_indices)
    if gold:
        gold = gdp.remove_instances_with_attributes(gold, trig_empty_indices)

    # generate candidate structures
    instances = dp.generate_candidate_structures(instances, templates, generalisation,
                                                 incl_mention, use_filter,
                                                 partial_arg_matching, dp)
    # remove instance with no events
    event_empty_indices = dp.get_indices_with_empty_events(instances, use_filter)
    instances = dp.remove_instances_with_empty_events(instances, event_empty_indices)
    if gold:
        gold = gdp.remove_instances_with_attributes(gold, event_empty_indices)

    return instances, gold, all_file_ids