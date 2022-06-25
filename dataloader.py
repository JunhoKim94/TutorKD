from torch.utils.data import Dataset
import pickle
from transformers import *
import random
import copy
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class create_dataset(Dataset):
    def __init__(self, mode, tokenizer, path):
        print("create_dataset..." + mode)
        with open(path + mode, "rb") as f:
            self.datas = pickle.load(f)
        self.tokenizer = tokenizer
    def __len__(self):
        # 데이터 전체의 사이즈 반환
        return len(self.datas)
        
    def random_masking(self, cpt_masked_sentence, adjusting_mask_prob, mask_count):
        masked_sentence = []
        label_mask_ = []
        lm_position = []
        for id_position, id in enumerate(cpt_masked_sentence):
            if id not in [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                          self.tokenizer.mask_token_id]:
                if random.random() <= adjusting_mask_prob:
                    lm_position.append(id_position+1)
                    mask_count += 1
                    label_mask_.append(False)  # masking 할거면 false
                    if random.random() >= 0.2:
                        masked_sentence.append(self.tokenizer.mask_token_id)
                    elif random.random() <= 0.5:
                        masked_sentence.append(random.randint(1, 30521))
                    else:
                        masked_sentence.append(id)
                else:
                    label_mask_.append(True)
                    masked_sentence.append(id)
            else:
                if id == self.tokenizer.mask_token_id:
                    label_mask_.append(False)  # 이미 mask token이면 false
                else:
                    label_mask_.append(True)
                masked_sentence.append(id)

        return masked_sentence, torch.BoolTensor(label_mask_), lm_position, mask_count

    def __getitem__(self, idx):
        #datapoint = self.datas[idx]
        # 학습할 corpus에 있는 concept 찾고
        datapoint = self.datas[idx]
        mask_count = 0
        masked_sentence, label_mask_, lm_position, mask_count = self.random_masking(datapoint['encoded_txt'], 0.15, mask_count)
        datapoint['masking_txt'] = masked_sentence
        datapoint['label_mask'] = label_mask_.tolist()
        datapoint['lm_position'] = lm_position
        return datapoint

def padded_sequence(samples):
        
    masked_LM = []
    LM_label = []
    label_mask_ = []
    label_position = []
    LM_max = 0
    for sample in samples:
        masked_LM.append(sample['masking_txt'])
        LM_label.append(sample['encoded_txt'])
        label_mask_.append(sample['label_mask'])
        label_position.append(sample['lm_position'])
        if len(sample['masking_txt']) > LM_max:
            LM_max = len(sample['masking_txt'])

    # print("q_max, a_max:", q_max, a_max)
    masked_lm_batch = []
    lm_label_batch = []
    label_mask_batch = []
    if LM_max > 128:
        LM_max = 128
    for i, LM_example in enumerate(masked_LM):
        if len(LM_example) <= LM_max:
            masked_lm_batch.append([tokenizer.cls_token_id]+LM_example+[tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (LM_max-len(LM_example)))
            lm_label_batch.append([tokenizer.cls_token_id]+LM_label[i]+[tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (LM_max-len(LM_label[i])))
            label_mask_batch.append([True]+label_mask_[i]+[True]+[True]*(LM_max-len(label_mask_[i])))
        else:
            masked_lm_batch.append(
                [tokenizer.cls_token_id] + LM_example[:LM_max] + [tokenizer.sep_token_id])
            lm_label_batch.append([tokenizer.cls_token_id]+ LM_label[i][:LM_max] + [tokenizer.sep_token_id])
            label_mask_batch.append([True] + label_mask_[i][:LM_max] + [True])

    return masked_lm_batch, lm_label_batch, label_mask_batch, label_position
