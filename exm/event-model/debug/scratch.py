import collections

def get_info(instances):
    d = dict()
    max = 0
    dl = collections.defaultdict(list)
    for i in range(len(instances)):
        try:
            groups = instances[i][6]
        except:
            print("debug")
        for level in groups:
            for trig, s in level.items():
                l = len(s)
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
                dl[l].append(i)
                if l > max:
                    max = l
    return d, max, dl

def printinfo(instances):
    d, max, dl = get_info(instances)
    print("max:"+ str(max))
    print("combi-count:")
    for k, v in d.items():
        print(k, "\t", v)
    print("combi-indices:")
    for k, v in dl.items():
        if k > 500:
            print(k, "\t", v)