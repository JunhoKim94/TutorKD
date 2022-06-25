import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import random
import argparse
import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.getcwd())
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
#from model.bert_layers import BertModel, BertForMaskedLM
from model.__main__module import *
from datetime import datetime
import torch
from dataloader import *
# from parallel
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
from model.bert import Small_Bert
from model.bert_layers import Bert_For_Att_output, Bert_For_Att_output_MLM
from config import *
from sampling import *

parser = argparse.ArgumentParser()

parser.add_argument("--data_parallel", default='False', help="use data parallel", type=str)
parser.add_argument("--gpu_num", default='0', help="choose gpu number: 0, 1, 2, 3", type=int)
parser.add_argument("--model", default='tiny-bert', help="choose model architecture from: tiny-bert, mobile-bert, minilm ", type=str)
parser.add_argument("--pretrained", default='bert-base_uncased',
                    help="choose model pretrained weight from: bert-base-uncased, bert-large-uncased, roberta-base, roberta-large",
                    type=str)
#parser.add_argument("--size", default='312', help="choose model size from: 768, 1024", type=int)
parser.add_argument("--lr", default=5e-4, help="insert learning rate", type=float)
parser.add_argument("--weight_decay", default=0.01, help="insert weight decay", type=float)
parser.add_argument("--epochs", default=1000, help="insert epochs", type=int)
parser.add_argument("--batch_size", default=128, help="insert batch size", type=int)
parser.add_argument("--step_batch_size", default=128, help="insert step batch size", type=int)
parser.add_argument("--random_seed", default=16, help="insert step batch size", type=int)
args = parser.parse_args()

summary = SummaryWriter(comment = 'runs/Distillation_%s_%s'%(str(args.pretrained), str(args.random_seed)))

#device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0")
torch.cuda.set_device(device)  # change allocation of current GPU
print('Current cuda device ', torch.cuda.current_device())  # check


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#configuration = Bert_Small_Head_Config
#configuration = Bert_Tiny_Config
#configuration = Mini_LM_2_layer
configuration = Bert_Small_Head_6_Config
#configuration = Bert_Small_Head_4_Config
#configuration = Bert_Small_Head_2_Config
#configuration = Bert_6_layer
#configuration = Bert_Small_Head_Hidden_Config

def get_model(args, configuration):
    if args.model.lower() == "minilm":
        base_model = Bert_For_Att_output(configuration, True, None)
        prediction_model = Bert_For_Att_output_MLM.from_pretrained("bert-base-uncased").to(device)
        model = MINILM_Only(base_model, configuration).cuda()

    elif args.model.lower() == "soft-distill":
        base_model = Bert_For_Att_output(configuration, True, None)
        prediction_model = Bert_For_Att_output_MLM.from_pretrained("bert-base-uncased").to(device)
        model = Soft_Distill_Model_All(base_model, configuration).cuda()

    return model, prediction_model

model, prediction_model = get_model(args, configuration)


#model_save_path = "save_model/baseline/2_layer_500000_tiny-bert.pt"
#model_save_path = "save_model/baseline/4_layer_400000_tiny-bert.pt"
#model.load_state_dict(torch.load(model_save_path, map_location = device))
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

print(args.random_seed)
scaler = torch.cuda.amp.GradScaler()

step = 1
iters = 1

# %load_ext tensorboard
# %tensorboard --logdir runs --port=8088
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
#optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98), eps=1e-6, lr=args.lr)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-6, lr=args.lr)
optimizer.zero_grad()

norm = transforms.Normalize(mean=torch.zeros(args.batch_size), std=torch.ones(args.batch_size))
prediction_model.requires_grad_(False)
softmax = nn.Softmax(-1)

path = "/home/user10/RTPP/data/origin/"


temp = []
for epoch in range(args.epochs):
    Loss = 0
    Distill_Loss = 0
    PP_Loss = 0
    Loss_len = 0
    print("now %s epoch..." % str(epoch + 1))
    for file in os.listdir(path):
        #train_dataset = create_dataset_RTPP(str(file), tokenizer)

        #train_dataset = create_dataset_RTPP_Allmask(str(file), tokenizer)
        train_dataset= create_dataset_Electra(str(file), tokenizer, path)

        #train_dataset = Origin_MLM_loader(str(file), tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=args.step_batch_size, shuffle=True,
                                      collate_fn=padded_sequence, drop_last=True, num_workers=5)

        for batch in tqdm(train_dataloader, ncols = 100):
            lm_embed, lm_label_embed, label_mask_, label_position = batch
            for i in range(int(args.step_batch_size / args.batch_size)):

                sub_lm_embed = lm_embed[i * args.batch_size:(i + 1) * args.batch_size]
                sub_lm_label = lm_label_embed[i * args.batch_size:(i + 1) * args.batch_size]
                sub_label_mask = label_mask_[i * args.batch_size:(i + 1) * args.batch_size]
                sub_label_position = label_position[i * args.batch_size:(i + 1) * args.batch_size]

                sub_batch = torch.LongTensor(sub_lm_embed).cuda()
                sub_lm_label = torch.LongTensor(sub_lm_label).cuda()
                sub_label_mask = torch.BoolTensor(sub_label_mask)

                attention_mask_ = (sub_batch == tokenizer.pad_token_id)
                zero_pad = torch.zeros(attention_mask_.size()).cuda()
                _attention_mask_ = zero_pad.masked_fill(attention_mask_, 1)

                with torch.cuda.amp.autocast():
            
                    outputs = prediction_model(sub_batch, attention_mask=_attention_mask_, output_hidden_states = True, output_attentions = True, return_dict = False)
                    t_logit = outputs[0]
                    t_hidden, t_att, t_value = outputs[-3:]

                    if args.model.lower() == "minilm":
                        distil_loss = model(sub_batch, _attention_mask_, t_hidden, t_att, t_value)

                        #distil_loss = torch.mean(distil_loss)

                        loss = distil_loss / (args.step_batch_size / args.batch_size)
                        Distill_Loss += distil_loss.item() / (args.step_batch_size / args.batch_size)

                    elif args.model.lower() == "tiny-bert":

                        t_logit = torch.softmax(t_logit[sub_label_mask == False], dim = -1)

                        distil_loss, mlm_loss = model(sub_batch, t_logit, sub_label_mask, _attention_mask_, t_hidden, t_att)
                        loss = (0.5 * distil_loss + 0.5 * mlm_loss) / (args.step_batch_size / args.batch_size)
                        Distill_Loss += distil_loss.item() / (args.step_batch_size / args.batch_size)

                    elif args.model.lower() == "soft-distill":
                        #soft_labels, sub_label_position = get_soft_label(outputs, sub_label_position, top_k = None)
                        #soft_labels, ret_indices, sub_label_position = get_soft_label(outputs, sub_label_position, top_k = 50)
                        soft_labels, ret_indices, sub_label_position = get_soft_label(outputs, sub_label_position, top_k = None)
                        #distil_loss = model(sub_batch, soft_labels, ret_indices, _attention_mask_)
                        distil_loss = model(sub_batch, soft_labels, _attention_mask_)
                        loss = distil_loss / (args.step_batch_size / args.batch_size)
                        Distill_Loss += distil_loss.item() / (args.step_batch_size / args.batch_size)


                    loss = loss.mean()
                    Loss += loss.item()
                    scaler.scale(loss).backward()

            optimizer, lr = lr_scheduler(args.lr, optimizer, step, warmup_step=10000, max_step=1000000)
            #optimizer, lr = lr_scheduler(args.lr, optimizer, step, warmup_step=4000, max_step = 400000)

            iters += 1
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            Loss_len += 1

            if iters % 500 == 0:
                summary.add_scalar('loss/loss_a', float(Loss / Loss_len), step)
                summary.add_scalar("loss/disc_loss", float(PP_Loss/ Loss_len), step)
                summary.add_scalar("loss/distill_loss", float(Distill_Loss / Loss_len), step)
                summary.add_scalar("hyp_para/lr", float(lr), step)
                Loss = 0
                Loss_len = 0
                Distill_Loss = 0
                PP_Loss = 0

            if iters % 100000 == 0:
                PATH = './save_model/baseline/%s_%s_%s.pt' % (
                    str(args.pretrained), str(step), str(args.model))
                print("save the model")
                torch.save(model.state_dict(), PATH)


summary.close()