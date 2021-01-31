import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)


#TODO: remove this file

# from util.util import Util
import string
import argparse
import yaml
from collections import OrderedDict


# def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
#   class OrderedLoader(Loader):
#       pass
#
#   def construct_mapping(loader, node):
#       loader.flatten_mapping(node)
#       return object_pairs_hook(loader.construct_pairs(node))
#
#   OrderedLoader.add_constructor(
#       yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
#       construct_mapping)
#   return yaml.load(stream, OrderedLoader)


# def update_constants(randstr):
#     global params
#     global TEMP
#     global OUTPUT_DIR
#     global TRAIN_OUTPUT_DIR
#     global TEST_OUTPUT_DIR
#     global TRAIN_PREDICTION_FILE_PATH
#     global TEST_PREDICTION_FILE_PATH
#     global TRAIN_PREDICTION_FILE
#     global TEST_PREDICTION_FILE
#     global LEARNING_CURVE
#     global PROCESSED_A2_FILES
#     global PRED_DIR
#     global PRED_NEWDIR
#     global OUTPUT_LOG_FILE
#     global OUTPUT_DATA_FILE
#     global MODEL_OUTPUT_FILE
#     TEMP = params['OUTPUT_FOLDER'] + "_" + randstr
#
#     print("TEMP:", TEMP)
#
#     OUTPUT_DIR = "../model/evaluation/" + TEMP + "/"
#
#     TRAIN_OUTPUT_DIR = OUTPUT_DIR + "train_pred/"
#     TEST_OUTPUT_DIR = OUTPUT_DIR + "test_pred/"
#
#     if not os.path.exists(TRAIN_OUTPUT_DIR):
#         os.makedirs(TRAIN_OUTPUT_DIR)
#
#     if not os.path.exists(TEST_OUTPUT_DIR):
#         os.makedirs(TEST_OUTPUT_DIR)
#
#     TRAIN_PREDICTION_FILE_PATH = OUTPUT_DIR + "train_prediction.out"
#     TEST_PREDICTION_FILE_PATH = OUTPUT_DIR + "dev_prediction.out"
#     TRAIN_PREDICTION_FILE = OUTPUT_DIR + "train_prediction.out"
#     TEST_PREDICTION_FILE = OUTPUT_DIR + "dev_prediction.out"
#     # EVALUATION_SCRIPT = "../model/evaluation/evaluation-CG-modified.py"
#     LEARNING_CURVE = OUTPUT_DIR + "learning_curve.png"
#
#     PROCESSED_A2_FILES = OUTPUT_DIR + "pred/"
#     if not os.path.exists(PROCESSED_A2_FILES):
#         os.makedirs(PROCESSED_A2_FILES)
#
#     PRED_DIR = OUTPUT_DIR + "pred/"
#     PRED_NEWDIR = OUTPUT_DIR + "prednew/"
#
#     OUTPUT_LOG_FILE = OUTPUT_DIR + "result.log"
#
#     OUTPUT_DATA_FILE = OUTPUT_DIR + "data.out"
#
#     MODEL_OUTPUT_FILE = OUTPUT_DIR + "sbm.model"


# str = random.choice(string.ascii_letters)+random.choice(string.ascii_letters)+random.choice(string.ascii_letters)+random.choice(string.ascii_letters)+random.choice(string.ascii_letters)

# parser = argparse.ArgumentParser()
# parser.add_argument('--yaml', type=str, required=True, help='yaml file')
# parser.add_argument('--train', action='store_true', help='training mode - uses train and dev set; gold is provided')
# parser.add_argument('--test', action='store_true', help='testing mode - predicts only; needs a model to load')

# parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
# parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=9)
# parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=243)
# parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=10)
# parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=4)
# parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
# parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
# parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.')
# parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.')
# parser.add_argument('--task_id', type=str, help='Task ID')
# parser.add_argument('--yaml', type=str, required=True, help='yaml file')

# args = parser.parse_args()
#
# with open(args.yaml, 'r') as stream:
#     # params = Util.ordered_load(stream)
#     params = ordered_load(stream)
# params = params['EVENT']

#set output folder from the yaml file
# temp1 = args.yaml.split("./")
# output_folder = temp1[1].split(".yaml")[0]
# output_folder = output_folder.split("yaml/")[1]
# params['output_folder'] = output_folder

class Constants(object):

    TEMP = output_folder

    OUTPUT_DIR = "../../" + epoch_output_folder + TEMP +"/"

    TRAIN_OUTPUT_DIR = OUTPUT_DIR + "train_pred/"
    TEST_OUTPUT_DIR = OUTPUT_DIR + "test_pred/"

    if not os.path.exists(TRAIN_OUTPUT_DIR):
        os.makedirs(TRAIN_OUTPUT_DIR)

    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)


    TRAIN_PREDICTION_FILE_PATH = OUTPUT_DIR + "train_prediction.out"
    TEST_PREDICTION_FILE_PATH = OUTPUT_DIR + "dev_prediction.out"
    TRAIN_PREDICTION_FILE = OUTPUT_DIR + "train_prediction.out"
    TEST_PREDICTION_FILE = OUTPUT_DIR + "dev_prediction.out"
    # EVALUATION_SCRIPT = "../model/evaluation/evaluation-CG-modified.py"
    LEARNING_CURVE = OUTPUT_DIR + "learning_curve.png"


    OUTPUT_EXTENSION = ".a2"

    PROCESSED_A2_FILES = OUTPUT_DIR + "pred/"
    if not os.path.exists(PROCESSED_A2_FILES):
        os.makedirs(PROCESSED_A2_FILES)

    PRED_DIR = OUTPUT_DIR + "pred/"
    PRED_NEWDIR = OUTPUT_DIR + "prednew/"

    #DIRECTORIES
    GOLD_TXT_OUTPUT_EXT = ".split.txt"
    GOLD_MAIN_TXT_EXT = ".txt"
    GOLD_ANN_OUTPUT_EXT = ".split.ann"
    PRED_NER_EXT = ".split.ner.ann"
    PRED_REL_EXT = ".split.rel.ann"

    OUTPUT_LOG_FILE = OUTPUT_DIR  + "result.log"

    OUTPUT_DATA_FILE = OUTPUT_DIR  + "data.out"

    MODEL_OUTPUT_FILE = OUTPUT_DIR + "sbm.model"



    MAX_INT_SIZE_FOR_IDS = 1000000


    # ON GOLD INSTANCES
    GOLD_SENTENCE_INFO_IDX = 0
    GOLD_SENTENCE_IDX = 0
    GOLD_SENTENCE_FILE_ID_IDX = 1
    GOLD_ENTITIES_IDX = 1
    GOLD_TRIGGERS_IDX = 2
    GOLD_EVENTS_IDX = 3

    #ON PREDS INSTANCES
    PRED_SENTENCE_INFO_IDX = 0
    PRED_SENTENCE_IDX = 0
    PRED_SENTENCE_FILE_ID_IDX = 1
    PRED_ENTITIES_IDX = 1
    PRED_TRIGGERS_IDX = 2
    PRED_RELATIONS_IDX = 3
    PRED_RELATION_TRIG_IDX = 1
    PRED_RELATION_ROLE_IDX = 0
    PRED_RELATION_ARG_IDX = 2
    PRED_NODES_IDX = 4
    PRED_CAND_STRUCTURES_IDX = 5
    PRED_FILTERED_STRUCTURES_IDX = 6

    # ON TRANSFORMED INSTANCES
    IDS_SENTENCE_INFO_IDX = 0
    IDS_SENTENCE_IDX = 0
    IDS_SENTENCE_FILE_ID_IDX = 1
    IDS_SENTENCE_ARG_INDICES = 2

    IDS_ENTITIES_IDX = 1
    IDS_TRIGGERS_IDX = 2

    IDS_ENTITIES_MERGED_IDX = 6
    IDS_TRIGGERS_MERGED_IDX = 7


    IDS_RELATIONS_IDX = 3
    IDS_EVENT_IDX = 4
    IDS_ARG_TYPE = 0
    IDS_ARG_MENTION = 3
    IDS_SIMPLE_EVENTS_IDX = 0
    IDS_COMPLEX_EVENTS_IDX = 1
    IDS_INSTANCE_ID = 1
    IDS_INSTANCE_DEFN = 0

    CONSTANT_WORDS = ['<UNK>', '<IN>', '<OUT>', '<NONE>']
    UNK_TOKEN = '<UNK>'

    IS_EVENT = 1
    IS_NON_EVENT = 0
    IN_EDGE = 1
    OUT_EDGE = 0
    EMPTY_STRUCTURE = [()]
    NONE_ROLE_TYPE = "NONE"
    NONE_ENTITY_TYPE = "NONE"
    NONE_TRIGGER_TYPE = "NONE"
    NONE_WORD_TYPE = "NONE"
    ACTION_NONE = -1
    ACTION_IGNORE = 0
    ACTION_ADD = 1
    ACTION_ADDFIX = 2
    ACTION_ADDFIXREMOVE = 3
    ACTION_LIST = [ACTION_IGNORE, ACTION_ADD, ACTION_ADDFIX]

#
# TRIGGER_LIST = [
#             'Development',
#             'Blood_vessel_development',
#             'Growth',
#             'Death',
#             'Cell_death',
#             'Breakdown',
#             'Cell_proliferation',
#             'Cell_division',
#             'Cell_differentiation',
#             'Remodeling',
#             'Reproduction',
#             'Mutation',
#             'Carcinogenesis',
#             'Cell_transformation',
#             'Metastasis',
#             'Infection',
#             'Metabolism',
#             'Synthesis',
#             'Catabolism',
#             'Amino_acid_catabolism',
#             'Glycolysis',
#             'Gene_expression',
#             'Transcription',
#             'Translation',
#             'Protein_processing',
#             'Phosphorylation',
#             'Dephosphorylation',
#             'DNA_methylation',
#             'DNA_demethylation',
#             'Pathway',
#             'Binding',
#             'Dissociation',
#             'Localization',
#             'Regulation',
#             'Positive_regulation',
#             'Negative_regulation',
#             'Planned_process',
#             'Acetylation',
#             'Glycosylation',
#             'Ubiquitination',
# ]
#
# ENTITY_LIST = [
#             'Organism',
#             'Organism_subdivision',
#             'Anatomical_system',
#             'Organ',
#             'Multi-tissue_structure',
#             'Tissue',
#             'Developing_anatomical_structure',
#             'Cell',
#             'Cellular_component',
#             'Organism_substance',
#             'Immaterial_anatomical_entity',
#             'Gene_or_gene_product',
#             'Simple_chemical',
#             'Protein_domain_or_region',
#             'Amino_acid',
#             'DNA_domain_or_region',
#             'Pathological_formation',
#             'Cancer'
# ]


# For CG
# Source: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-16-S10-S2
# and here: http://2013.bionlp-st.org/tasks/cancer-genetics
# CG_TYPES = {
#     'Event':
#         {
#             'Anatomical':
#                 [
#                     'Development',
#                         'Blood_vessel_development',
#                     'Growth',
#                     'Death',
#                         'Cell_death',
#                     'Breakdown',
#                     'Cell_proliferation',
#                     'Cell_division',
#                     'Cell_differentiation',
#                     'Remodeling',
#                     'Reproduction'
#                 ],
#             'Pathological':
#                 [
#                     'Mutation',
#                     'Carcinogenesis',
#                     'Cell_transformation',
#                     'Metastasis',
#                     'Infection'
#                 ],
#             'Molecular':
#                 [
#                     'Metabolism',
#                         'Synthesis',
#                         'Catabolism',
#                             'Amino_acid_catabolism',
#                             'Glycolysis',
#                         'Gene_expression',
#                             'Transcription',
#                             'Translation',
#                             'Protein_processing',
#                     'Phosphorylation',
#                     'Dephosphorylation',
#                     'DNA_methylation',
#                     'DNA_demethylation',
#                     'Pathway'
#                 ],
#             'General':
#                 [
#                     'Binding',
#                     'Dissociation',
#                     'Localization',
#                     'Regulation',
#                         'Positive_regulation',
#                         'Negative_regulation',
#                     'Planned_process'
#                 ]
#         },
#     'Entity':
#         {
#             'Anatomy':
#                 [
#                     'Organism',
#                     'Organism_subdivision',
#                     'Anatomical_system',
#                     'Organ',
#                     'Multi-tissue_structure',
#                     'Tissue',
#                     'Developing_anatomical_structure',
#                     'Cell',
#                     'Cellular_component',
#                     'Organism_substance',
#                     'Immaterial_anatomical_entity',
#                 ],
#             'Molecule':
#                 [
#                     'Gene_or_gene_product',
#                     'Simple_chemical',
#                     'Protein_domain_or_region',
#                     'Amino_acid',
#                     'DNA_domain_or_region',
#                     'Pathological_formation',
#                     'Cancer'
#                 ]
#         }
# }

'''
Generates the generalisations from CG types above
'''
# import collections
# mapping = collections.defaultdict(list)
# for topkey, toplevel in CG_TYPES.items():
#     for midkey, midlevel in toplevel.items():
#         for actual in midlevel:
#             mapping[actual].append(midkey)
#             mapping[actual].append(topkey)

# TYPE_GENERALISATION = {'Amino_acid': ['Molecule', 'Entity'],
#              'Amino_acid_catabolism': ['Molecular', 'Event'],
#              'Anatomical_system': ['Anatomy', 'Entity'],
#              'Binding': ['General', 'Event'],
#              'Blood_vessel_development': ['Anatomical', 'Event'],
#              'Breakdown': ['Anatomical', 'Event'],
#              'Cancer': ['Molecule', 'Entity'],
#              'Carcinogenesis': ['Pathological', 'Event'],
#              'Catabolism': ['Molecular', 'Event'],
#              'Cell': ['Anatomy', 'Entity'],
#              'Cell_death': ['Anatomical', 'Event'],
#              'Cell_differentiation': ['Anatomical', 'Event'],
#              'Cell_division': ['Anatomical', 'Event'],
#              'Cell_proliferation': ['Anatomical', 'Event'],
#              'Cell_transformation': ['Pathological', 'Event'],
#              'Cellular_component': ['Anatomy', 'Entity'],
#              'DNA_demethylation': ['Molecular', 'Event'],
#              'DNA_domain_or_region': ['Molecule', 'Entity'],
#              'DNA_methylation': ['Molecular', 'Event'],
#              'Death': ['Anatomical', 'Event'],
#              'Dephosphorylation': ['Molecular', 'Event'],
#              'Developing_anatomical_structure': ['Anatomy', 'Entity'],
#              'Development': ['Anatomical', 'Event'],
#              'Dissociation': ['General', 'Event'],
#              'Gene_expression': ['Molecular', 'Event'],
#              'Gene_or_gene_product': ['Molecule', 'Entity'],
#              'Glycolysis': ['Molecular', 'Event'],
#              'Growth': ['Anatomical', 'Event'],
#              'Immaterial_anatomical_entity': ['Anatomy', 'Entity'],
#              'Infection': ['Pathological', 'Event'],
#              'Localization': ['General', 'Event'],
#              'Metabolism': ['Molecular', 'Event'],
#              'Metastasis': ['Pathological', 'Event'],
#              'Multi-tissue_structure': ['Anatomy', 'Entity'],
#              'Mutation': ['Pathological', 'Event'],
#              'Negative_regulation': ['General', 'Event'],
#              'Organ': ['Anatomy', 'Entity'],
#              'Organism': ['Anatomy', 'Entity'],
#              'Organism_subdivision': ['Anatomy', 'Entity'],
#              'Organism_substance': ['Anatomy', 'Entity'],
#              'Pathological_formation': ['Molecule', 'Entity'],
#              'Pathway': ['Molecular', 'Event'],
#              'Phosphorylation': ['Molecular', 'Event'],
#              'Planned_process': ['General', 'Event'],
#              'Positive_regulation': ['General', 'Event'],
#              'Protein_domain_or_region': ['Molecule', 'Entity'],
#              'Protein_processing': ['Molecular', 'Event'],
#              'Regulation': ['General', 'Event'],
#              'Remodeling': ['Anatomical', 'Event'],
#              'Reproduction': ['Anatomical', 'Event'],
#              'Simple_chemical': ['Molecule', 'Entity'],
#              'Synthesis': ['Molecular', 'Event'],
#              'Tissue': ['Anatomy', 'Entity'],
#              'Transcription': ['Molecular', 'Event'],
#              'Translation': ['Molecular', 'Event'],
#             'Acetylation': ['General', 'Event'],
#             'Glycosylation': ['General', 'Event'],
#             'Ubiquitination': ['General', 'Event'],
# }
# '''
# GENERATING TAR.GZ FOR ONLINE SUBMISSION
# 1. go to pred directory
# 2. execute: tar -zcvf preds.tar.gz *
# '''