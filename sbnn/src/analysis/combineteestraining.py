gold = "/Users/kurt/Desktop/dataset/tees/GE13/train/"
pred = "/Users/kurt/Desktop/dataset/tees/trainrelationpreds/ge13/"
output = "/Users/kurt/Desktop/dataset/tees/GE13/traingoldpred/"

import os
from collections import OrderedDict


separator = "=#=#"
def read_entitiestriggers(file):
    triggers = OrderedDict()
    entities = OrderedDict()
    for line in file:
        if line.startswith("TR"):
            tokens = line.split()
            triggers[tokens[0]] = separator.join(tokens[1:])
        else:
            tokens = line.split()
            entities[tokens[0]] = separator.join(tokens[1:])

    return entities,triggers

def read_relations(file):
    relations = OrderedDict()
    for line in file:
        tokens = line.split()
        relations[tokens[0]] = separator.join(tokens[1:])
    return relations

def change_trigids(predrelations, trigger_mapping):
    new_relations = OrderedDict()
    for id, defn in predrelations.items():
        tokens = defn.split(separator)
        arg1 = tokens[1]
        arg2 = tokens[2]
        if arg1 in trigger_mapping:
            arg1 = trigger_mapping[arg1]
        if arg2 in trigger_mapping:
            arg2 = trigger_mapping[arg2]
        new_defn = tokens[0]+separator+arg1+separator+arg2
        new_relations[id] = new_defn
    return new_relations

def merge_relations(invgold, invpred, gold):
    #find the relations in pred that are not in gold
    rel2add = []
    for defn, id in invpred.items():
        if defn not in invgold:
            rel2add.append(defn)

    #add the gold relations and determine max id
    new_relations = OrderedDict()
    max_id = 0
    for k, v in gold.items():
        new_relations[k] = v
        rel_id = int(k[1:])
        if rel_id > max_id:
            max_id = rel_id

    ctr = 1
    for defn in rel2add:
        new_id = "R" + str(max_id + ctr)
        new_relations[new_id] = defn
        ctr += 1

    return new_relations

def merge_triggers(invgold, invpred, gold, pred):
    # search for ids in pred that are not in gold
    id2add = []
    for defn, id in invpred.items():
        if defn not in invgold:
            id2add.append(id)

    # add the gold trigges and get max trig id
    new_trigs = OrderedDict()
    max_id = 0
    for k, v in gold.items():
        new_trigs[k] = v
        trig_id = int(k[2:])
        if trig_id > max_id:
            max_id = trig_id

    # add predicted triggers with new ids using max id and create mapping of ids
    ctr = 1
    trigger_mapping = OrderedDict()
    for id in id2add:
        defn = pred[id]
        new_id = "TR" + str(max_id + ctr)
        new_trigs[new_id] = defn
        ctr += 1
        trigger_mapping[id] = new_id
    return new_trigs, trigger_mapping


gold_dir = os.listdir(gold)
for file in gold_dir:
    if file.endswith(".split.ner.ann"):
        file_id = file[:-14]
        nergoldfile = open(gold + file_id + ".split.ner.ann", 'r')
        nerpredfile = open(pred + file_id + ".split.ner.ann", 'r')
        goldentities, goldtriggers = read_entitiestriggers(nergoldfile)
        _, predtriggers = read_entitiestriggers(nerpredfile)

        #invert mapping
        inv_gold_trigs = {v:k for k, v in goldtriggers.items()}
        inv_pred_trigs = {v:k for k, v in predtriggers.items()}

        #merge pred and gold triggers
        final_triggers, trigger_mapping = merge_triggers(inv_gold_trigs, inv_pred_trigs, goldtriggers, predtriggers)

        #change the trigger ids in relation preds with the new ids
        relgoldfile = open(gold + file_id + ".split.rel.ann", 'r')
        relpredfile = open(pred + file_id + ".split.rel.ann", 'r')
        predrelations = read_relations(relpredfile)
        goldrelations = read_relations(relgoldfile)

        new_predrelations = change_trigids(predrelations, trigger_mapping)

        #invert relations
        inv_goldrelations = {v:k for k,v in goldrelations.items()}
        inv_predrelations = {v:k for k,v in new_predrelations.items()}

        #merge pred and gold relations
        final_relations = merge_relations(inv_goldrelations, inv_predrelations, goldrelations)

        #write to final ner and rel files
        final_nerfile = open(output+file_id+".split.ner.ann", 'w')
        final_relfile = open(output + file_id + ".split.rel.ann", 'w')

        for id, defn in goldentities.items():
            tokens = defn.split(separator)
            final_nerfile.write(id+"\t"+tokens[0]+" "+tokens[1]+" "+tokens[2]+"\t"+' '.join(tokens[3:])+"\n")

        for id, defn in final_triggers.items():
            tokens = defn.split(separator)
            final_nerfile.write(
                id + "\t" + tokens[0] + " " + tokens[1] + " " + tokens[2] + "\t" + ' '.join(tokens[3:]) + "\n")
        final_nerfile.close()
        for id, defn in final_relations.items():
            tokens = defn.split(separator)
            final_relfile.write(
                id + "\t" + tokens[0] + " " + tokens[1] + " " + tokens[2] + "\n")
        final_relfile.close()
print("Finished.")