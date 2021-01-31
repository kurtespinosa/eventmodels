import os
import constants as const
import subprocess
# import matplotlib.pyplot as plt

def convert_tr_to_t(INPUT_DIR):
    files = os.listdir(INPUT_DIR)

    pred_dir = os.listdir(const.PROCESSED_A2_FILES)
    for f in pred_dir:
        os.remove(os.path.join(const.PROCESSED_A2_FILES, f))

    for file in files:
        new_file = const.PROCESSED_A2_FILES + file
        file_write = open(new_file, 'w')
        if file.endswith(".a2"):
            lines = open(INPUT_DIR+file, 'r').readlines()
            for line in lines:
                new_line = ''
                if line.startswith("E"):
                    tokens = line.split()
                    type_and_trig_id = tokens[1]
                    event_type, trigger_id = type_and_trig_id.split(":")
                    new_trigger_id = "T"+trigger_id[2:]
                    the_rest = ''
                    if len(tokens) > 2:
                        for token in tokens[2:]:
                            new_token = token.split(":")
                            new_role = new_token[0]
                            try:
                                new_arg = new_token[1]
                            except:
                                print("here")
                            if new_arg.startswith("TR"):
                                new_arg = "T" + new_token[1][2:]
                            combined = new_role + ":" + new_arg
                            the_rest += combined + " "
                    new_line = tokens[0]+"\t"+event_type+":"+new_trigger_id+" "+the_rest.rstrip()
                else:
                    tokens = line.split()
                    trig_id = tokens[0]
                    if trig_id.startswith("TR"):
                        trig_id = "T" + trig_id[2:]
                    new_line = trig_id+"\t"+tokens[1]+" "+tokens[2]+" "+tokens[3]+"\t"+' '.join(tokens[4:])
                file_write.write(new_line+"\n")
        file_write.close()



def compute_f_score(INPUT_DIR, TRAIN_DIR, TEST_DIR, EVALUATION_SCRIPT, INTERPRETER, DATASET, IS_TRAIN):
    convert_tr_to_t(INPUT_DIR)

    if IS_TRAIN:
        ref = TRAIN_DIR
        pred = const.TRAIN_PREDICTION_FILE
        path = const.TRAIN_PREDICTION_FILE_PATH
    else:
        ref = TEST_DIR
        pred = const.TEST_PREDICTION_FILE
        path = const.TEST_PREDICTION_FILE_PATH

    # OPTION 1
    # command = "python " + const.EVALUATION_SCRIPT + " -r " + ref
    # result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    # output = open(const.PREDICTION_FILE, 'w')
    # output.write((result.stdout).decode("utf-8"))
    # output.close()

    try:
        command = None
        if DATASET == 'GE11':
            #needs normalisation as it says here when executing perl a2-evaluate.pl
            try:
                pred_dir = os.listdir(const.PRED_NEWDIR)
                for f in pred_dir:
                    os.remove(os.path.join(const.PRED_NEWDIR, f))
            except Exception as e:
                print("Error removing files", e)
            command = INTERPRETER + EVALUATION_SCRIPT + "a2-normalize.pl " + " -g " + ref + " -o " + const.PRED_NEWDIR +" " + const.PRED_DIR + "*.a2 > " + pred
            # print(command)
            os.system(command)

            # command = INTERPRETER + EVALUATION_SCRIPT + " -g " + ref + " " + const.PRED_DIR + "*.a2 > " + pred
            command = INTERPRETER + EVALUATION_SCRIPT + "a2-evaluate.pl " + " -g " + ref + " " + const.PRED_NEWDIR + "*.a2 > " + pred
            # print(command)
        elif DATASET in ['PC13', 'CG13']:
            command = INTERPRETER + EVALUATION_SCRIPT + " -r " + ref + " -d " + const.PRED_DIR +  " > " + pred
            # print(command)
        elif DATASET in ['ID11', 'EPI11']:
            command = INTERPRETER + EVALUATION_SCRIPT + ref + " " + const.PRED_DIR + "*.a2 > " + pred
            # print(command)
        elif DATASET == 'GE13':
            try:
                pred_dir = os.listdir(const.PRED_NEWDIR)
                for f in pred_dir:
                    os.remove(os.path.join(const.PRED_NEWDIR, f))
            except Exception as e:
                print("Error removing files", e)
            command = INTERPRETER + EVALUATION_SCRIPT + "a2-normalize.pl " + " -g " + ref + " -o " + const.PRED_NEWDIR +" " + const.PRED_DIR + "*.a2 > " + pred
            os.system(command)
            command = INTERPRETER + EVALUATION_SCRIPT + "a2-evaluate.pl " + " -g " + ref + " " + const.PRED_NEWDIR + "*.a2 > " + pred
        elif DATASET == 'GE09':
            try:
                pred_dir = os.listdir(const.PRED_NEWDIR)
                for f in pred_dir:
                    os.remove(os.path.join(const.PRED_NEWDIR, f))
            except Exception as e:
                print("Error removing files", e)
            command = "export PATH=$PATH:" + EVALUATION_SCRIPT
            # print(command)
            os.system(command)
            command = INTERPRETER + EVALUATION_SCRIPT + "generate-task-specific-a2-file.pl -t 1 " + ref +"*.a2"
            # print(command)
            os.system(command)
            command = INTERPRETER + EVALUATION_SCRIPT + "generate-task-specific-a2-file.pl -t 1 " + const.PRED_DIR + "*.a2"
            # print(command)
            os.system(command)
            command = INTERPRETER + EVALUATION_SCRIPT + "prepare-eval.pl -g " + ref + " " + const.PRED_DIR +" " + const.PRED_NEWDIR
            # print(command)
            os.system(command)
            command = INTERPRETER + EVALUATION_SCRIPT + "a2-evaluate.pl " + " -g "+ ref + " " + const.PRED_NEWDIR + "*.t1 >" + pred
            # print(command)
        os.system(command)
        fscore, recall, precision = extract_fscore(path)
    except:
        fscore = recall = precision = 0.0

    return fscore, recall, precision

def extract_fscore(path):
    file = open(path, 'r')
    lines = file.readlines()
    fscore = ''
    for line in lines:
        if line.split()[0] == '===[SUB-TOTAL]===' or line.split()[0] == '====[TOTAL]====' or line.split()[0] == '==[ALL-TOTAL]==':
            tokens = line.split()
            recall = tokens[-3]
            precision = tokens[-2]
            fscore = tokens[-1]
            break
    return float(fscore.strip()), float(recall.strip()), float(precision.strip())



# def plot(array1, array2, array1_label, array2_label, title, filename):
#     x_axis = range(1, len(array1) + 1)
#     lines = ["-", "--", "-.", ":", "^", ",", "_", "+"]
#
#     plt.plot(x_axis, array1, label=array1_label, linestyle=lines[1], color='black')
#     plt.plot(x_axis, array2, label=array2_label, linestyle=lines[0], color='black')
#
#     plt.rcParams["figure.figsize"] = (10, 6)
#
#     plt.legend()
#     plt.ylabel('Fscore')
#     plt.xlabel('Epoch')
#     plt.title(title)
#     plt.grid()
#     plt.savefig(const.LEARNING_CURVE)
    # plt.show()
# compute_f_score(const.TRAIN_OUTPUT_DIR, IS_TRAIN=True)
