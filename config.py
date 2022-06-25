from transformers import BertConfig, ElectraConfig
import argparse

Bert_6_layer = BertConfig(vocab_size = 30522,
                            num_hidden_layers = 6)

Bert_Small_Head_2_Config = BertConfig(vocab_size = 30522,
                                hidden_size = 264,
                                num_hidden_layers = 2,
                                num_attention_heads = 12,
                                intermediate_size = 1056,
                                hidden_act = "gelu",
                                hidden_dropout_prob = 0.1,
                                attention_probs_dropout_prob = 0.1,
                                max_position_embeddings = 512,
                                initializer_range = 0.02,
                                layer_norm_eps = 1e-12,
                                pad_token_id = 0,
                                gradient_checkpointing = False)


Bert_Small_Head_4_Config = BertConfig(vocab_size = 30522,
                                hidden_size = 264,
                                num_hidden_layers = 4,
                                num_attention_heads = 12,
                                intermediate_size = 1056,
                                hidden_act = "gelu",
                                hidden_dropout_prob = 0.1,
                                attention_probs_dropout_prob = 0.1,
                                max_position_embeddings = 512,
                                initializer_range = 0.02,
                                layer_norm_eps = 1e-12,
                                pad_token_id = 0,
                                gradient_checkpointing = False)


Bert_Small_Head_6_Config = BertConfig(vocab_size = 30522,
                                hidden_size = 264,
                                num_hidden_layers = 6,
                                num_attention_heads = 12,
                                intermediate_size = 1056,
                                hidden_act = "gelu",
                                hidden_dropout_prob = 0.1,
                                attention_probs_dropout_prob = 0.1,
                                max_position_embeddings = 512,
                                initializer_range = 0.02,
                                layer_norm_eps = 1e-12,
                                pad_token_id = 0,
                                gradient_checkpointing = False)


Bert_Small_Head_Config = BertConfig(vocab_size = 30522,
                                hidden_size = 264,
                                num_hidden_layers = 12,
                                num_attention_heads = 12,
                                intermediate_size = 1056,
                                hidden_act = "gelu",
                                hidden_dropout_prob = 0.1,
                                attention_probs_dropout_prob = 0.1,
                                max_position_embeddings = 512,
                                initializer_range = 0.02,
                                layer_norm_eps = 1e-12,
                                pad_token_id = 0,
                                gradient_checkpointing = False)

def ret_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--gpu_num", default='3', help="choose gpu number: 0, 1, 2, 3", type=int)
    parser.add_argument("--model", default='bert', help="choose model architecture from: bert, roberta, ", type=str)
    parser.add_argument("--pretrained", default='bert-base-uncased', help="choose model pretrained weight from: bert-base-uncased, bert-large-uncased, roberta-base, roberta-large", type=str)
    parser.add_argument("--size", default='768', help="choose model size from: 768, 1024", type=int)
    parser.add_argument("--lr", default=1e-5, help="insert learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.01, help="insert weight decay", type=float)
    parser.add_argument("--layer_wise", default= 0.8, help = "layer wise dacay", type = float)
    parser.add_argument("--epochs", default=3, help="insert epochs", type=int)
    parser.add_argument("--batch_size", default=32, help="insert batch size", type=int)
    parser.add_argument("--step_batch_size", default=32, help="insert step batch size", type=int)
    parser.add_argument("--random_seed", default=23, help="insert step batch size", type=int)

    args = parser.parse_args()

    return args

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

if __name__ == "__main__":
    from model.bert_layers import Bert_For_Att_output, Bert_For_Att_output_MLM
    from torchsummary import summary as summary_

    #configuration = Bert_Small_Config
    #configuration = Bert_Small_Head_Config
    configuration = Bert_Small_Head_6_Config

    base_model = Bert_For_Att_output(configuration, True, None)
    print(get_n_params(base_model) / 1e6)