import os

dir = "/Users/kurt/GoogleDrive/02N/04PROFDEV/PHD/phd/02process/experiments/deepevent/event-structure-detection/codeversions/event-util-system/model/evaluation/scenario2_T.5/"
table = dir + "dev_prediction.out"



file = open(table, 'r')
lines = file.readlines()
triplet = []
for line in lines:
    line = line.lstrip()
    if not (line.startswith("Event") or line.startswith("=") or line.startswith("--") or line.startswith("INTERNAL")):
        vals = line.split()
        triplet.append((vals[0], vals[1], vals[-3]))


output = open(dir + "file.csv", 'w')
output.write("Type"+","+"Gold Count"+","+"Recall"+"\n")
for t in triplet:
    output.write(t[0]+","+t[1]+","+t[2]+"\n")

output.close()