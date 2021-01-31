#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019 15:12
 
Author: Kurt Espinosa
Email: kpespinosa@gmail.com
"""

# seed for deterministic output
import os
os.environ["CHAINER_SEED"]="0"
import random
random.seed(0)
import numpy as np
np.random.seed(0)

# standard library
from datetime import datetime
import argparse
import logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# 3rd party
import chainer
from chainer import iterators, optimizers
from chainer import serializers
from chainer.backends import cuda

# local modules
from src.util import goldprocessor as gdp
from src.util import dataprocessor as dp
from src.util import util
from src.util import loader
from src.util import constants as const
from src.util import postprocess
from src.util import stats
from src.model.SBNN import SBNN
from src.model.SBM  import SBM
from src.model.SBM2  import SBM2


# logging settings
log = logging.getLogger(__name__)

def trainer(train_ids, model, optimizer, args, arg, output_dir, train_file_ids,
          output_folder, dev_ids, dev_file_ids, templates):
    '''
    :param train_ids:
    :param model:
    :param optimizer:
    :param args:
    :param arg:
    :param output_dir:
    :param train_file_ids:
    :param output_folder:
    :param dev_ids:
    :param dev_file_ids:
    :param templates:
    :return:

    TODO: Align logging for the scores.
    '''

    log.info("Training with number of instances: %s", len(train_ids))

    # iterators
    train_iter = iterators.SerialIterator(train_ids, batch_size=args['batch_size'], shuffle=True,
                                          repeat=False)

    entries = ['Epoch', 'Loss', 'Precision', 'Recall', 'Fscore', 'Runtime', 'Precision', 'Recall', 'Fscore', 'Runtime']
    header = ' '.join(a for a in ["{:<10}".format(str(a[:10])) for a in entries])
    log.info(header)
    print(header)

    train_predictions = []
    train_fscores = []
    test_fscores = []
    train_epoch = 0
    prev_fscore = 0.0
    prev_recall = 0.0
    prev_train_fscore = 0.0
    training_patience = 0
    epoch_loss = 0.0
    batch_count = 0
    best_test_score = 0.0
    new_train_instance_ids = []
    start_time = datetime.now()
    while train_epoch < args['max_epoch']:
        train_batch = train_iter.next()
        train_prediction, batch_loss, _ = model(train_batch)
        epoch_loss += batch_loss
        batch_count += 1
        train_predictions.extend(train_prediction)
        new_train_instance_ids.extend(train_batch)
        log.debug("computing gradients..")
        model.cleargrads()
        batch_loss.backward()
        # optimizer.update()
        log.debug("batch count: %s", batch_count)
        if train_iter.is_new_epoch:
            train_epoch += 1
            if arg.merged_train:
                log.info("Epoch: %s", str(train_epoch))
                if train_epoch >= args['starting_epoch_for_output']:
                    # save the model here
                    serializers.save_npz(output_dir + const.MODEL_FILE + "_" + str(train_epoch), model)
                    log.debug("Model saved: %s", output_dir + const.MODEL_FILE + "_" + str(train_epoch))
                    # copy to a directory
                    if args['copy_dir']:
                        util.copy_output_dir(output_folder, args['root_output_copy_folder'],
                                             args['epoch_output_folder'], train_epoch)
            else: # compute the train scores only when tuning
                ave_loss = epoch_loss / batch_count
                batch_count = 0
                epoch_loss = 0
                dp.generate_prediction_file_for_sbm(new_train_instance_ids, train_predictions,
                                                    templates, output_dir+const.TRAIN_PRED_DIR, train_file_ids,
                                                    args['special_entities'], args['post_filter'] )
                log.debug("computing fscore..")
                train_fscore, train_recall, train_precision = postprocess.compute_f_score(
                                                                output_dir + const.TRAIN_PRED_DIR,
                                                                args['train_dir'],
                                                                args['dev_dir'],
                                                                args['evaluation_script'],
                                                                args['interpreter'],
                                                                args['dataset'],
                                                                True,
                                                                output_dir)
                train_fscores.append(train_fscore)

                end = datetime.now()
                train_running_time = end - start_time
                # log.info("%3.2d\t%2.4f\t%2.4f\t\t%2.4f\t%2.4f\t%s", train_epoch,ave_loss.data.tolist(),
                #          train_precision,train_recall,train_fscore, running_time)

                train_predictions = []

                # evaluate model on dev
                with chainer.using_config('train', False):
                    test_precision, test_recall, test_fscore, test_running_time = evaluate(dev_ids, model, dev_file_ids, output_dir, False, args, templates)

                    entries = [str(train_epoch), str(ave_loss.data.tolist()), str(train_precision), str(train_recall), str(train_fscore), str(train_running_time), str(test_precision), str(test_recall), str(test_fscore), str(test_running_time)]
                    header = ' '.join(a for a in ["{:<10}".format(str(a[:10])) for a in entries])
                    log.info(header)
                    print(header)


                if test_fscore > best_test_score:
                    training_patience = 0
                    best_test_score = test_fscore
                    if train_epoch >= args['starting_epoch_for_output']:
                        # save the model here
                        serializers.save_npz(output_dir + "model_" + str(train_epoch), model)
                        log.debug("Model saved: %s", output_dir + "model_" + str(train_epoch))
                        # copy to a directory
                        if args['copy_dir']:
                            util.copy_output_dir(output_folder, args['root_output_copy_folder'],
                                                 args['epoch_output_folder'], train_epoch)
                else:
                    training_patience += 1
                if training_patience == args['patience']:
                    break



            train_iter = iterators.SerialIterator(train_ids, batch_size=args['batch_size'], shuffle=True,
                                                  repeat=False)
            new_train_instance_ids = []
            start_time = datetime.now()

        # call after the model has been applied to the dev
        optimizer.update()

def evaluate(dev_ids, model, dev_file_ids, output_dir, is_train, args, templates):
    start_time = datetime.now()
    test_predictions = []

    log.debug("Evaluating with number of instances: %s", len(dev_ids))
    test_iter = iterators.SerialIterator(dev_ids, batch_size=args['batch_size'], shuffle=True, repeat=False)
    new_test_instance_ids = []

    test_fscore = 0.0
    test_recall = 0.0
    test_ave_loss = 0.0
    test_batch_count = 0
    test_epoch_loss = 0.0

    while True:
        test_batch = test_iter.next()
        test_prediction, batch_loss, _ = model(test_batch)
        test_predictions.extend(test_prediction)
        new_test_instance_ids.extend(test_batch)

        test_epoch_loss += batch_loss
        test_batch_count += 1
        log.debug("batch count: %s", test_batch_count)

        if test_iter.is_new_epoch:
            test_ave_loss = test_epoch_loss / test_batch_count
            dp.generate_prediction_file_for_sbm(new_test_instance_ids, test_predictions, templates,
                                                output_dir + const.TEST_OUTPUT_DIR,
                                                dev_file_ids, args['special_entities'], args['post_filter'])
            log.debug("computing fscore..")
            test_fscore, test_recall, test_precision = postprocess.compute_f_score(
                                                            output_dir + const.TEST_OUTPUT_DIR,
                                                            args['train_dir'],
                                                            args['dev_dir'],
                                                            args['evaluation_script'],
                                                            args['interpreter'],
                                                            args['dataset'], is_train,
                                                            output_dir)

            end = datetime.now()
            running_time = end - start_time
            break
    return test_precision, test_recall, test_fscore, running_time

def predict(model, modelfile, test_ids, test_file_ids, is_train, output_dir, args, templates):

    #set chainer config train global variable
    chainer.config.train = False

    serializers.load_npz(modelfile, model)
    log.info("Model loaded: %s", modelfile)

    test_iter = iterators.SerialIterator(test_ids, batch_size=args['batch_size'], shuffle=True, repeat=False)

    test_fscore = 0.0
    test_predictions = []
    new_test_instance_ids = []
    log.info("Generating predictions on test set.")

    totalcount = 0

    while True:
        test_batch = test_iter.next()
        test_prediction, _, count = model(test_batch)
        test_predictions.extend(test_prediction)
        new_test_instance_ids.extend(test_batch)

        totalcount += count

        if test_iter.is_new_epoch:
            dp.generate_prediction_file_for_sbm(new_test_instance_ids, test_predictions, templates,
                                                output_dir + const.TEST_OUTPUT_DIR,
                                                test_file_ids, args['special_entities'], args['post_filter'])
            if args['scenario'] == 1:
                log.info("scenario 1")
                test_fscore, test_recall, test_precision = postprocess.compute_f_score(output_dir + const.TEST_OUTPUT_DIR,
                                                                                       args['train_dir'],
                                                                                       args['test_dir'],
                                                                                       args['evaluation_script'],
                                                                                       args['interpreter'],
                                                                                       args['dataset'], is_train,
                                                                                       output_dir)
                log.info("\t%.4f\t%.4f\t%.4f", test_precision, test_recall, test_fscore)
            elif args['scenario'] == 0:
                postprocess.convert_tr_to_t(output_dir + const.TEST_OUTPUT_DIR, output_dir)
            break
    log.info("Finished.")
    log.info("Total Classifications: %s", totalcount)

def main():
    # require yaml input
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True, help='yaml file')
    parser.add_argument('--merged_train', type=bool, default=False, help='train using merged train+dev')
    parser.add_argument('--test', type=bool, default=False, help='test')
    arg = parser.parse_args()

    # read params
    args = util.parse_yaml(arg.yaml)
    output_folder = util.extract_yaml_name(arg.yaml) +  "_" + util.get_prefix()


    # create output dir
    output_dir = util.set_out_dir(args['root_output_copy_folder'], output_folder,
                                args['epoch_output_folder'])

    # set logging output path
    if args['log_level'] == 'debug':
        logging.basicConfig(filename=str(output_dir + "result.log"),
                            format='%(asctime)s - %(levelname)s: %(message)s',
                            level=logging.DEBUG, filemode='w')
    elif args['log_level'] == 'info':
        logging.basicConfig(filename=str(output_dir+"result.log"),
                            format='%(asctime)s - %(levelname)s: %(message)s',
                            level=logging.INFO, filemode='w')
    log.info('Parameters loaded: %s', arg.yaml)
    util.log_params(args)

    # ------------ GOLD TRAIN AND DEV PROCESSING ----- #
    # load train, dev data
    train_gold, train_file_id_events = gdp.load(args['train_dir'], args['is_emo'])
    train_gold = gdp.denumber_roles(train_gold)
    dev_gold, dev_file_id_events = gdp.load(args['dev_dir'], args['is_emo'])
    dev_gold = gdp.denumber_roles(dev_gold)

    # extract templates from gold incl intersentence events
    if arg.merged_train:
        # templates = gdp.extract_templates({**train_file_id_events, **dev_file_id_events})
        # train_gold = train_gold+dev_gold
        log.error("Not implemented.")
    else:
        templates = gdp.extract_templates(train_file_id_events)

    # ------------ PREDICTED TRAIN AND DEV PROCESSING ----- #
    # load train data
    train, train_gold, train_file_ids = loader.load(args['train_dir'], args['train_ner'],
                                    args['train_rel'],args['is_emo'], args['excl_train'],
                                    args['incl_mention'], args['generalisation'], args['use_filter'],
                                    args['partial_arg_matching'], templates, train_gold)
    # load dev data
    dev, dev_gold, dev_file_ids = loader.load(args['dev_dir'], args['dev_ner'],
                                args['dev_rel'], args['is_emo'], args['excl_dev'],
                                args['incl_mention'], args['generalisation'], args['use_filter'],
                                args['partial_arg_matching'], templates, dev_gold)

    if arg.merged_train:
        train = train + dev
        train_gold = train_gold + dev_gold
        train_file_ids = train_file_ids + dev_file_ids

    # generate vocabulary
    word_type2id, role_type2id, arg_type2id, singleton, l1_types2id, l2_types2id = dp.create_vocabulary(
        train,args['unk_min_freq'], args['unk_assign_prob'])

    # train: generate ids, gold actions and merge
    train_ids = dp.instances_to_ids(train, word_type2id, role_type2id,
                                    arg_type2id, args['use_filter'], 'train')
    assert len(train_ids) == len(train_gold), log.error("Error: %s != %s", str(len(train_ids)), str(len(train_gold)))
    train_ids = gdp.generate_gold_actions(train_ids, train_gold)
    train_ids = util.merge(train, train_ids)
    train_ids = util.add_type_generalisation(train_ids, arg_type2id, l1_types2id, l2_types2id)


    # dev: generate ids, gold actions and merge
    dev_ids = dp.instances_to_ids(dev, word_type2id, role_type2id, arg_type2id, args['use_filter'], 'dev')
    assert len(dev_ids) == len(dev_gold), log.error("Error: %s != %s", str(len(dev_ids)), str(len(dev_gold)))
    dev_ids = gdp.generate_gold_actions(dev_ids, dev_gold)
    dev_ids = util.merge(dev, dev_ids)
    dev_ids = util.add_type_generalisation(dev_ids, arg_type2id, l1_types2id, l2_types2id)

    if arg.merged_train:
        train_ids = train_ids + dev_ids

    # ------------ TEST PROCESSING ----- #
    if arg.test:
        args['is_emo'] = True  # since predictions are based on IS_EMO, this is False for EM and True for TEES
        test, _, test_file_ids = loader.load(args['test_dir'], args['test_ner'],
                                        args['test_rel'], args['is_emo'], args['excl_test'],
                                        args['incl_mention'], args['generalisation'], args['use_filter'],
                                        args['partial_arg_matching'], templates)
        test_ids = dp.instances_to_ids(test, word_type2id, role_type2id,
                                       arg_type2id, args['use_filter'], 'test')
        test_ids = util.merge(test, test_ids)
        test_ids = util.add_type_generalisation(test_ids, arg_type2id, l1_types2id, l2_types2id)


    # ----------- STATS ---------------- #
    if args['event_arg_counts']:
        stats.generate_heatmaps(train_ids,dev_ids, args['use_filter'], train_gold, dev_gold, output_dir)

    if args['level_rec_analysis']:
        gold_per_level, pred_per_level = stats.level_recall_analysis(train_gold, train, dev_gold, dev)
        log.info("Gold per level: %s", gold_per_level)
        log.info("Pred per level: %s", pred_per_level)

    if args['max_nbest']:
        max_nbest_train = stats.get_max_nbest(train_ids, args['use_filter'])
        max_nbest_dev = stats.get_max_nbest(dev_ids, args['use_filter'])
        log.info("Max N-best needed for train = %s, dev = %s", max_nbest_train, max_nbest_dev)

    if args['argcnt_eventctr']:
        d_train = stats.arg_cnt_event_ctr(train_ids, args['use_filter'])
        d_dev = stats.arg_cnt_event_ctr(dev_ids, args['use_filter'])
        log.info("Count of event with certain argument count in train: %s", d_train)
        log.info("Count of event with certain argument count in dev: %s", d_dev)

    if args['only_stats']:
        log.info("Finished.")
        sys.exit()


    # compute nestedness in train and test
    # train_nestedness = Util.compute_nestedness(train_instances, train_new_gold, USE_FILTER)
    # if TASK in ['TUNE'] :
    #     test_nestedness = Util.compute_nestedness(test_instances, test_new_gold,  USE_FILTER)


    #NESTEDNESS STATS
    '''
    if TASK == "TUNE":
        print("Train nestedness statistics:")
        print("Level : Events")
        for level, count in train_nestedness.items():
            print(str(level), ":", str(count))
        print("Test nestedness statistics:")
        print("Level : Events")
        for level, count in test_nestedness.items():
            print(str(level), ":", str(count))

        result_file.write("Train nestedness statistics:")
        result_file.write("Level : Events")
        for level, count in train_nestedness.items():
            result_file.write(str(level) + ":" + str(count))
        result_file.write("Test nestedness statistics:")
        result_file.write("Level : Events")
        for level, count in test_nestedness.items():
            result_file.write(str(level) +  ":" + str(count))
    '''

    # train_num_events_ids = Util.count_gold_events_in_ids(train_instances_ids, USE_FILTER)
    # if TASK in ['TUNE'] :
    #     test_num_events_ids = Util.count_gold_events_in_ids(test_instances_ids, USE_FILTER)

    # if TASK in ['TUNE'] :
    #     train_max_num_gold_actions_d, train_max_num_gold_actions = Util.max_num_gold_actions(train_instances_ids)
    #     test_max_num_gold_actions_d, test_max_num_gold_actions = Util.max_num_gold_actions(test_instances_ids)
    #     least_beam_size = max(train_max_num_gold_actions, test_max_num_gold_actions)
    #     print("Least beam size:", least_beam_size)

    # print("TRAIN set: ", train_num_events_ids)
    # if TASK in ['TUNE'] :
    #     print("TEST set: ", test_num_events_ids)


    #UNCOMMENT FOR TESTING
    # start_id = 0
    # end_id = 2
    # train = train[start_id:end_id]
    # train_ids = train_ids[start_id:end_id]
    # dev = dev[start_id:end_id]
    # dev_ids = dev_ids[start_id:end_id]
    # dev_gold = dev_gold[start_id:end_id]


    #DEBUG
    # for i in train_instances_ids:
    #     for j in i:
    #         data_file.write(str(j)+"\n")
    # for i in test_instances_ids:
    #     for j in i:
    #         data_file.write(str(j)+"\n")
    # data_file.close()


    # model prep
    # model = SBNN(len(word_type2id), len(arg_type2id), len(role_type2id), args)
    model = SBM(len(word_type2id), len(arg_type2id), len(role_type2id), len(arg_type2id), args)
    # model = SBM2(len(word_type2id), len(arg_type2id), len(role_type2id), len(arg_type2id), args)
    if args['embed_path'] is not None:
        model.load_pret_embed(args['embed_path'], word_type2id)
    if args['gpu'] >= 0:
        chainer.backends.cuda.get_device_from_id(args['gpu']).use()
        model.to_gpu(args['gpu'])


    optimizer = None
    if args['optimiser'] == 'Adam':
        optimizer = optimizers.Adam(amsgrad=True, weight_decay_rate=args['weight_decay_rate'])
    elif args['optimiser'] == 'SGD':
        optimizer = optimizers.SGD(lr=float(args['LR']))
    optimizer.setup(model)

    # test
    if arg.test:
        predict(model, args['model'], test_ids, test_file_ids, False, output_dir, args, templates)
    else:
        # train and tune
        trainer(train_ids, model, optimizer, args, arg, output_dir, train_file_ids,
                output_folder, dev_ids, dev_file_ids, templates)

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    log.info("Running time: %s", str(end - start))