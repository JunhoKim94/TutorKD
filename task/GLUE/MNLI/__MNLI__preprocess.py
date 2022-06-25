import os
import csv
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='dev_matched.tsv', help="insert file name", type=str)
parser.add_argument("--load_dir", default='./task/GLUE/MNLI/MNLI/', help="insert load directory name", type=str)
parser.add_argument("--save_dir", default='./task/GLUE/MNLI/dataset/', help="insert save dir name", type=str)
parser.add_argument("--save_filename", default="val", help="insert save file name", type=str)

args = parser.parse_args()

with open(os.path.join(args.load_dir, args.dataset), 'r') as tsv:
    file = [line.strip().split('\t') for line in tsv]

print(len(file))
dataset = []
for example in file:
    datapoint = {}
    if example[-1] == 'neutral':
        datapoint['label'] = 0
    elif example[-1] == 'entailment':
        datapoint['label'] = 1
    else:
        datapoint['label'] = 2
    datapoint['sen1'] = example[8].lower()
    datapoint['sen2'] = example[9].lower()
    dataset.append(datapoint)

print(datapoint)
print(len(dataset))
with open(os.path.join(args.save_dir,args.save_filename), 'wb') as f:
    pickle.dump(dataset, f)


