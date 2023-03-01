import pickle
from transformers import BertTokenizer
import argparse
import os

def preprocess(args):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    path = args.data_path
    save_path = args.save_data_path
    for file in os.listdir(path):
        p = path + str(file)
        with open(p, "r") as f:
            x = f.readlines()

        ret = []
        for s in x:
            ret_dict = dict()
            encoded = tokenizer.encode(s, add_special_tokens=False)
            ret_dict["encoded_txt"] = encoded
            ret.append(ret_dict)

        save_file = file.split(".")[0] + ".pickle"
        with open(save_path + save_file, "wb") as f:
            pickle.dump(ret, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_data_path", default = "./raw_data/", type = str)
    parser.add_argument("--data_path", default = "./data/", type = str)
    args = parser.parse_args()

    preprocess(args)