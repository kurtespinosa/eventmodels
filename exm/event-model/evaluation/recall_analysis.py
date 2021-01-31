import os
import collections

TRIGGER_LIST = [
            'Development',
            'Blood_vessel_development',
            'Growth',
            'Death',
            'Cell_death',
            'Breakdown',
            'Cell_proliferation',
            'Cell_division',
            'Cell_differentiation',
            'Remodeling',
            'Reproduction',
            'Mutation',
            'Carcinogenesis',
            'Cell_transformation',
            'Metastasis',
            'Infection',
            'Metabolism',
            'Synthesis',
            'Catabolism',
            'Amino_acid_catabolism',
            'Glycolysis',
            'Gene_expression',
            'Transcription',
            'Translation',
            'Protein_processing',
            'Phosphorylation',
            'Dephosphorylation',
            'DNA_methylation',
            'DNA_demethylation',
            'Pathway',
            'Binding',
            'Dissociation',
            'Localization',
            'Regulation',
            'Positive_regulation',
            'Negative_regulation',
            'Planned_process',
            'Acetylation',
            'Glycosylation',
            'Ubiquitination',
]

gold_dir = "/Users/kurt/Desktop/exm/dataset/emo/dev/"
pred_dir = "/Users/kurt/Desktop/exm/event-model/evaluation/EPOCH_OUTPUT_FOLDER/exm_emo_10_39/pred/"

gold_files = os.listdir(gold_dir)
pred_files = os.listdir(pred_dir)

event_types = ['Pathway', 'Binding']

rel_ext = ".split.rel.ann"

def create_dict_for_special_ents(file, reverse=False):
    SPECIAL_ENTITIES = ['Protein_domain_or_region', 'DNA_domain_or_region']
    file = open(file, 'r')
    lines = file.readlines()
    special_ent_dict = dict()
    for line in lines:
        if line.startswith("T"):
            tokens = line.split()
            if tokens[1] in SPECIAL_ENTITIES:
                offset_str = str(tokens[2]+tokens[3])
                if reverse:
                    special_ent_dict[tokens[0]] = offset_str
                else:
                    special_ent_dict[offset_str] = tokens[0]
    return special_ent_dict

def get_events(file):
    # create dict for all events: event_id: trigger_id
    file = open(file, 'r')
    lines = file.readlines()
    events_dict = dict()
    for line in lines:
        if line.startswith("E"):
            tokens = line.split()
            events_dict[tokens[0]] = tokens[1].split(":")[1]
    return events_dict

def create_set(file, gold_ne=None, pred_ne=None):
    events_dict = get_events(file)

    events = []
    file = open(file, 'r')
    lines = file.readlines()
    for line in lines:
        if line.startswith("E"):
            tokens = line.split()

            #create event set
            event_set = set()
            event_set.add(tokens[1])

            # if an arg is an event, replace it with the trigger_id
            if len(tokens) > 2:
                for arg in tokens[2:]:
                    role, arg_id = arg.split(":")
                    if arg_id.startswith("E"):
                        arg_id = events_dict[arg_id]
                    if pred_ne is not None:
                        if arg_id in pred_ne:
                            try:
                                offset = pred_ne[arg_id]
                                if offset in gold_ne:
                                    arg_id = gold_ne[offset]
                            except:
                                print("error")
                    new_arg = role+":"+arg_id
                    event_set.add(new_arg)

            events.append(event_set)
    return events

def extract_trigger_args(event):
    trigger = None
    args = []
    for s in event:
        arg1, arg2 = s.split(":")
        if arg1 in TRIGGER_LIST:
            trigger = arg2
            # break
        else:
            args.append(s)
    return trigger, args

def create_rel_dict(rel_file):
    lines = open(rel_file, 'r').readlines()
    rel_d = collections.defaultdict(list)
    for line in lines:
        _, role, arg1, arg2 = line.split()
        arg = role+":"+arg2
        rel_d[arg1].append(arg)
    return rel_d

def which_args_not_in_rel(args, relations):
    def convert_arg_to_tr(a):
        role, arg = a.split(":")
        new_arg = role+":"+"TR"+arg[1:]
        return new_arg
    missing_args = []
    for a in args:
        if a not in relations:
            new_a = convert_arg_to_tr(a)
            if new_a not in relations:
                missing_args.append(a)
    return missing_args

# open gold file and its equivalent pred file, create a set an expanded set representation for each event
gold_events_d = collections.defaultdict()
pred_events_d = collections.defaultdict()
rel_dict = collections.defaultdict()
for file in gold_files:
    if file.endswith(".a2"):
        file_id = file.split(".a2")[0]
        # if file_id == 'PMID-1675427':
        #     print("debug")
        gold_file = gold_dir+file
        pred_file = pred_dir+file

        gold_special_ents = create_dict_for_special_ents(gold_file)
        pred_special_ents = create_dict_for_special_ents(pred_file, reverse=True)
        gold_event_set = create_set(gold_file)
        pred_event_set = create_set(pred_file, gold_special_ents, pred_special_ents)
        gold_events_d[file_id] = gold_event_set
        pred_events_d[file_id] = pred_event_set

        rel_file = gold_dir + file_id + rel_ext
        rel_d = create_rel_dict(rel_file)
        rel_dict[file_id] = rel_d

# given some event types display the difference in predictions if there are

num_missing_events = 0
num_with_missed_relations = 0
num_with_missing_triggers = 0
for file_id, file_events in gold_events_d.items():
    # if file_id == 'PMID-1675427':
    #     print("debug")
    pred_events = pred_events_d[file_id]
    relations_d = rel_dict[file_id]
    print("\n", file_id)
    print("\tMissed events:")
    for event in file_events:
        if event not in pred_events:
            trigger, args = extract_trigger_args(event)
            trig_mod = "TR"+str(trigger[1:])
            relations = relations_d[trig_mod]
            args_not_in_rel = which_args_not_in_rel(args, relations)
            if not relations:
                # print("\t\t", event)
                # print("\t\t\tTrigger missing in relations.")
                num_with_missing_triggers += 1
            elif not args_not_in_rel:
                print("\t\t", event)
                print("\t\t\tALL relations present.")
                #check for other reasons such as templates, if sub-events cannot be formed because of missing relations/trigger
            else:
                # print("\t\t", event)
                # print("\t\t\tMissing relations:", args_not_in_rel)
                num_with_missed_relations += 1
            num_missing_events += 1

    # print("\tWrong predictions:")
    # for event in pred_events:
    #     if event not in file_events:
    #         print("\t\t",event)

print("Num Missing Events:", num_missing_events)
print("Events with missing triggers in relations:", num_with_missing_triggers, num_with_missing_triggers/num_missing_events*100)
print("Events with missing relations:", num_with_missed_relations, num_with_missed_relations/num_missing_events*100)
print("Events missed but has relations:", num_missing_events-num_with_missed_relations-num_with_missing_triggers)