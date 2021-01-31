#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019 19:21
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""

TRAIN_PRED_DIR = "train_pred/"
TEST_OUTPUT_DIR = "test_pred/"
TRAIN_PREDICTION_FILE_PATH = "train_prediction.out"
TEST_PREDICTION_FILE_PATH = "dev_prediction.out"
TRAIN_PREDICTION_FILE = "train_prediction.out"
TEST_PREDICTION_FILE = "dev_prediction.out"
LEARNING_CURVE = "learning_curve.png"


OUTPUT_EXTENSION = ".a2"

PROCESSED_A2_FILES = "pred/"


PRED_DIR = "pred/"
PRED_NEWDIR = "prednew/"

OUTPUT_LOG_FILE = "result.log"

OUTPUT_DATA_FILE = "data.out"

MODEL_FILE = "sbm.model"

# DIRECTORIES
GOLD_TXT_OUTPUT_EXT = ".split.txt"
GOLD_MAIN_TXT_EXT = ".txt"
GOLD_ANN_OUTPUT_EXT = ".split.ann"
PRED_NER_EXT = ".split.ner.ann"
PRED_REL_EXT = ".split.rel.ann"

MAX_INT_SIZE_FOR_IDS = 1000000

# ON GOLD INSTANCES
GOLD_SENTENCE_INFO_IDX = 0
GOLD_SENTENCE_IDX = 0
GOLD_SENTENCE_FILE_ID_IDX = 1
GOLD_SENTENCE_POS_IDX = 2
GOLD_ENTITIES_IDX = 1
GOLD_TRIGGERS_IDX = 2
GOLD_EVENTS_IDX = 3

# ON PREDS INSTANCES
PRED_SENTENCE_INFO_IDX = 0
PRED_SENTENCE_IDX = 0
PRED_SENTENCE_FILE_ID_IDX = 1
PRED_SENTENCE_POS_IDX = 2
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
IDS_SENTENCE_POS_IDX = 2
IDS_SENTENCE_ARG_INDICES = 3
IDS_SENTENCE_WORD_INDICES = 4

IDS_ENTITIES_IDX = 1
IDS_TRIGGERS_IDX = 2

IDS_RELATIONS_IDX = 3
IDS_EVENT_IDX = 4

IDS_SIMPLE_EVENTS_IDX = 0
IDS_COMPLEX_EVENTS_IDX = 1

IDS_ENTITIES_MERGED_IDX = 6
IDS_TRIGGERS_MERGED_IDX = 7

IDS_TYPE_GENERALISATION = 8

IDS_ARG_TYPE = 0
IDS_ARG_MENTION = 3

IDS_INSTANCE_DEFN = 0
IDS_INSTANCE_ID = 1

CONSTANT_WORDS = ['<UNK>', '<IN>', '<OUT>']
UNK_TOKEN = '<UNK>'

IS_EVENT = 1
IS_NON_EVENT = 0
IN_EDGE = 1
OUT_EDGE = 0
EMPTY_STRUCTURE = [()]
NONE_ROLE_TYPE = "<NONE>"
NONE_ARG_TYPE = "<NONE>"
# NONE_TRIGGER_TYPE = "NONE"
NONE_WORD_TYPE = "<NONE>"

ID_UNK_ROLE_TYPE = 1
# for the NONE relation
ID_NONE_WORD_TYPE = 0
ID_NONE_ROLE_TYPE = 0
ID_NONE_ARG_TYPE = 0

NONE_LEVEL_1_TYPE = "<NONE>"
NONE_LEVEL_2_TYPE = "<NONE>"
ID_NONE_LEVEL_1_TYPE = 0
ID_NONE_LEVEL_2_TYPE = 0

# has dependencies in generating gold actions so can't put ACTION_NONE in index 0
INITIAL_ACTION = ''
ACTION_NONE = -1
ACTION_IGNORE = 0
ACTION_ADD = 1
ACTION_ADDFIX = 2
ACTION_LIST = [ACTION_IGNORE, ACTION_ADD, ACTION_ADDFIX]

TYPE_GENERALISATION = {'Amino_acid': ['Molecule', 'Entity'],
             'Amino_acid_catabolism': ['Molecular', 'Event'],
             'Anatomical_system': ['Anatomy', 'Entity'],
             'Binding': ['General', 'Event'],
             'Blood_vessel_development': ['Anatomical', 'Event'],
             'Breakdown': ['Anatomical', 'Event'],
             'Cancer': ['Molecule', 'Entity'],
             'Carcinogenesis': ['Pathological', 'Event'],
             'Catabolism': ['Molecular', 'Event'],
             'Cell': ['Anatomy', 'Entity'],
             'Cell_death': ['Anatomical', 'Event'],
             'Cell_differentiation': ['Anatomical', 'Event'],
             'Cell_division': ['Anatomical', 'Event'],
             'Cell_proliferation': ['Anatomical', 'Event'],
             'Cell_transformation': ['Pathological', 'Event'],
             'Cellular_component': ['Anatomy', 'Entity'],
             'DNA_demethylation': ['Molecular', 'Event'],
             'DNA_domain_or_region': ['Molecule', 'Entity'],
             'DNA_methylation': ['Molecular', 'Event'],
             'Death': ['Anatomical', 'Event'],
             'Dephosphorylation': ['Molecular', 'Event'],
             'Developing_anatomical_structure': ['Anatomy', 'Entity'],
             'Development': ['Anatomical', 'Event'],
             'Dissociation': ['General', 'Event'],
             'Gene_expression': ['Molecular', 'Event'],
             'Gene_or_gene_product': ['Molecule', 'Entity'],
             'Glycolysis': ['Molecular', 'Event'],
             'Growth': ['Anatomical', 'Event'],
             'Immaterial_anatomical_entity': ['Anatomy', 'Entity'],
             'Infection': ['Pathological', 'Event'],
             'Localization': ['General', 'Event'],
             'Metabolism': ['Molecular', 'Event'],
             'Metastasis': ['Pathological', 'Event'],
             'Multi-tissue_structure': ['Anatomy', 'Entity'],
             'Mutation': ['Pathological', 'Event'],
             'Negative_regulation': ['General', 'Event'],
             'Organ': ['Anatomy', 'Entity'],
             'Organism': ['Anatomy', 'Entity'],
             'Organism_subdivision': ['Anatomy', 'Entity'],
             'Organism_substance': ['Anatomy', 'Entity'],
             'Pathological_formation': ['Molecule', 'Entity'],
             'Pathway': ['Molecular', 'Event'],
             'Phosphorylation': ['Molecular', 'Event'],
             'Planned_process': ['General', 'Event'],
             'Positive_regulation': ['General', 'Event'],
             'Protein_domain_or_region': ['Molecule', 'Entity'],
             'Protein_processing': ['Molecular', 'Event'],
             'Regulation': ['General', 'Event'],
             'Remodeling': ['Anatomical', 'Event'],
             'Reproduction': ['Anatomical', 'Event'],
             'Simple_chemical': ['Molecule', 'Entity'],
             'Synthesis': ['Molecular', 'Event'],
             'Tissue': ['Anatomy', 'Entity'],
             'Transcription': ['Molecular', 'Event'],
             'Translation': ['Molecular', 'Event'],
            'Acetylation': ['General', 'Event'],
            'Glycosylation': ['General', 'Event'],
            'Ubiquitination': ['General', 'Event'],
}