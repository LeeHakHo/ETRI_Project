with open('/home/ohh/dataset/merge/gt_merge_valid.txt') as f:
    with open('/home/ohh/dataset/merge/gt_merge_valid_1.txt' + '', 'w', encoding='utf-8') as t:
        datafile = f.readlines()
        for line in datafile:
            line = line.rstrip()
            t.write(line+'\n')