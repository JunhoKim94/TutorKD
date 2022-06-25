import os
import csv
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='msr_paraphrase_test.txt', help="insert file name", type=str)
parser.add_argument("--load_dir", default='./task/GLUE/MSRParaphraseCorpus/', help="insert load directory name", type=str)
parser.add_argument("--save_dir", default='./task/GLUE/MSRParaphraseCorpus/dataset/', help="insert save dir name", type=str)
parser.add_argument("--save_filename", default="test", help="insert save file name", type=str)

args = parser.parse_args()

with open(os.path.join(args.load_dir, args.dataset), 'r', encoding='utf-8') as tsv:
    file = [line.strip().split('\t') for line in tsv]

print(len(file))
dataset = []
print(file[0])
for example in file[1:]:
    datapoint = {}
    datapoint['label'] = int(example[0])
    datapoint['sen1'] = example[3].lower()
    datapoint['sen2'] = example[4].lower()
    dataset.append(datapoint)
print(datapoint)
print(len(dataset))
with open(os.path.join(args.save_dir,args.save_filename), 'wb') as f:
    pickle.dump(dataset, f)


