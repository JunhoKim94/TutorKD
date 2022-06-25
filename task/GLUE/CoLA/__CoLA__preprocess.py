import os
import csv
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='dev.tsv', help="insert file name", type=str)
parser.add_argument("--load_dir", default='./task/GLUE/CoLA/', help="insert load directory name", type=str)
parser.add_argument("--save_dir", default='./task/GLUE/CoLA/dataset/', help="insert save dir name", type=str)
parser.add_argument("--save_filename", default="val", help="insert save file name", type=str)

args = parser.parse_args()

with open(os.path.join(args.load_dir, args.dataset), 'r') as tsv:
    file = [line.strip().split('\t') for line in tsv]

print(len(file))
dataset = []
print(file[0])
for example in file:
    datapoint = {}
    datapoint['label'] = int(example[1])
    datapoint['pure_txt'] = example[3].lower()
    dataset.append(datapoint)
print(datapoint)
print(len(dataset))
with open(os.path.join(args.save_dir,args.save_filename), 'wb') as f:
    pickle.dump(dataset, f)


