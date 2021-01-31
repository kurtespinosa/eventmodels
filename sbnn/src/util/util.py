#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019 18:54
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""

# seed
import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)

# standard
import yaml
from collections import OrderedDict, defaultdict
import logging
log = logging.getLogger(__name__)
from shutil import copytree, ignore_patterns

from collections import OrderedDict, Callable
import datetime

# local modules
import src.util.constants as const

def log_params(args):
    for k,v in args.items():
        log.info("%s : %s", str(k), str(v))

def get_prefix():
    now = datetime.datetime.now()
    year = str(now.year)
    month = "0" + str(now.month) if len(str(now.month)) < 2 else str(now.month)
    day = "0" + str(now.day) if len(str(now.day)) < 2 else str(now.day)
    hour = "0" + str(now.hour) if len(str(now.hour)) < 2 else str(now.hour)
    minute = "0" + str(now.minute) if len(str(now.minute)) < 2 else str(now.minute)
    sec = "0" + str(now.second) if len(str(now.second)) < 2 else str(now.second)
    total_len = len(year) + len(month) + len(day) + len(hour) + len(minute) + len(sec)
    assert total_len == 14, 'Error: creating prefix'
    prefix = year + month + day + hour + minute + sec
    return prefix

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
        args = ordered_load(stream)
    args = args['EVENT']
    return args

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

def create_trig_event_map(events):
    trigger_event_map = defaultdict(list)
    for _, structure in events.items():
        trigger = structure[0].split(":")[1]
        trigger_event_map[trigger].append(structure)
    return trigger_event_map

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

def get_indices(dic, boundaries):
    '''
    Get indices of a list of sentence offsets.

    :param dic: dict containing sentence offsets (from, to)
    :param boundaries: list of sentence boundaries
    :return: dictionary of items group by indices
    '''
    lst_indices = defaultdict(dict)
    for k, v in dic.items():
        e_from = v[1]
        e_to = v[2]
        index = get_index(int(e_from), int(e_to), boundaries)
        lst_indices[index][k] = v
    return lst_indices

def get_index(e_from, e_to, boundaries):
    '''
    Get the index from the boundaries.

    :param e_from: element from value
    :param e_to: element to value
    :param boundaries: list of sentence boundaries
    :return: index of [from, to] in the boundaries
    '''
    ind = -1
    for i in range(len(boundaries)):
        fr = boundaries[i][0]
        to = boundaries[i][1]
        if e_from >= fr and e_to <= to:
            ind = i
            break
    return ind


def set_out_dir(output_root, output_folder, epoch_output_folder):
    OUTPUT_DIR = output_root + epoch_output_folder + output_folder + "/"

    TRAIN_OUTPUT_DIR = OUTPUT_DIR + "train_pred/"
    TEST_OUTPUT_DIR = OUTPUT_DIR + "test_pred/"

    if not os.path.exists(TRAIN_OUTPUT_DIR):
        os.makedirs(TRAIN_OUTPUT_DIR)

    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)

    PROCESSED_A2_FILES = OUTPUT_DIR + "pred/"
    if not os.path.exists(PROCESSED_A2_FILES):
        os.makedirs(PROCESSED_A2_FILES)

    #TODO: can be removed and just compose when needed
    PRED_DIR = OUTPUT_DIR + "pred/"
    PRED_NEWDIR = OUTPUT_DIR + "prednew/"

    # OUTPUT_LOG_FILE = OUTPUT_DIR + "result.log"
    OUTPUT_DATA_FILE = OUTPUT_DIR + "data.out"
    MODEL_OUTPUT_FILE = OUTPUT_DIR + "sbm.model"

    TRAIN_PREDICTION_FILE_PATH = OUTPUT_DIR + "train_prediction.out"
    TEST_PREDICTION_FILE_PATH = OUTPUT_DIR + "dev_prediction.out"
    TRAIN_PREDICTION_FILE = OUTPUT_DIR + "train_prediction.out"
    TEST_PREDICTION_FILE = OUTPUT_DIR + "dev_prediction.out"
    LEARNING_CURVE = OUTPUT_DIR + "learning_curve.png"

    return OUTPUT_DIR

def get_unique_types_level():
    l1 = set()
    l2 = set()
    for k,v in const.TYPE_GENERALISATION.items():
        l1.add(v[0])
        l2.add(v[1])
    return l1, l2

def extract_category(arg, level, TYPE_GENERALISATION):
    '''

    :param arg:
    :param level:
    :param TYPE_GENERALISATION:
    :return:
    '''
    new_arg = ''
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

def get_type_gen(dct, type2id, l1_types2id, l2_types2id):
    d = DefaultOrderedDict()
    for k, v in dct.items():
        typ = v[0]
        level1 = extract_category(typ, 1, const.TYPE_GENERALISATION)
        level2 = extract_category(typ, 2, const.TYPE_GENERALISATION)
        if level1 in l1_types2id:
            level1_id = l1_types2id[level1]
        else:
            level1_id = const.ID_NONE_LEVEL_1_TYPE
        if level2 in l2_types2id:
            level2_id = l2_types2id[level2]
        else:
            level2_id = const.ID_NONE_LEVEL_2_TYPE
        d[k] = (typ, level1_id, level2_id)
    return d



def add_type_generalisation(instance_ids, arg_type2id, l1_types2id, l2_types2id):
    for i in range(len(instance_ids)):
        entity_actual = instance_ids[i][const.IDS_ENTITIES_MERGED_IDX]
        trigger_actual = instance_ids[i][const.IDS_TRIGGERS_MERGED_IDX]
        ent_type_gen = get_type_gen(entity_actual, arg_type2id, l1_types2id, l2_types2id)
        trig_type_gen = get_type_gen(trigger_actual, arg_type2id, l1_types2id, l2_types2id)
        ent_type_gen[const.NONE_ARG_TYPE] = (const.NONE_ARG_TYPE, const.ID_NONE_LEVEL_1_TYPE, const.ID_NONE_LEVEL_2_TYPE)
        instance_ids[i].insert(const.IDS_TYPE_GENERALISATION, {**ent_type_gen, **trig_type_gen})
    return instance_ids

def merge(instances, instance_ids):
    new_instance_ids = []
    for i in range(len(instance_ids)):
        instance = instance_ids[i]
        entities = instances[i][const.IDS_ENTITIES_IDX]
        triggers = instances[i][const.IDS_TRIGGERS_IDX]
        sentence = instances[i][const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]
        instance.append(entities)
        instance.append(triggers)
        instance[0].append(sentence.split())
        assert len(instance[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_IDX]) == len(instance[const.IDS_SENTENCE_INFO_IDX][const.IDS_SENTENCE_WORD_INDICES]), log.exception("Error in merge.")
        new_instance_ids.append(instance)
    return new_instance_ids

def copy_output_dir(output_folder, root_output_copy_folder, epoch_output_folder, train_epoch):
    root_folder = root_output_copy_folder
    epoch_output_folder = root_folder + epoch_output_folder
    if not os.path.exists(epoch_output_folder):
        os.makedirs(epoch_output_folder)
    from_folder =  epoch_output_folder + output_folder
    log.debug("Copying - from: %s", from_folder)
    to_folder = epoch_output_folder + output_folder + "/best/"+ str(train_epoch)
    log.debug("Copying - to: %s", to_folder)
    copytree(from_folder, to_folder, ignore=ignore_patterns('best'))

def get_rel_indices(rels, ent_indices, trig_indices):

    rel_index = defaultdict(dict)
    invalid_count = 0
    total_count = 0
    for id, defn in rels.items():
        trig = defn[1]
        arg = defn[2]

        trig_ind = get_arg_index(trig, trig_indices)
        arg_ind = get_arg_index(arg, ent_indices)
        if arg_ind == -1:
            arg_ind = get_arg_index(arg, trig_indices)

        if trig_ind == arg_ind:
            rel_index[trig_ind][id] = defn
            total_count += 1
        else:
            log.debug("Not supported: intersentence relation: %s, %s, %s, %s", str(id),
                      defn, str(trig_ind), str(arg_ind))
            invalid_count += 1
    return rel_index, invalid_count, total_count

def get_arg_index(trig, indices):
    ind = -1
    for k, v in indices.items():
        if trig in v:
            ind = k
            break
    return ind

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

