import os
from gpu_mem_track import MemTracker

import torch.nn as nn
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
# from sklearn.preprocessing import normalize
from transformers import AutoConfig, AutoModel, AutoTokenizer

import data_utils_NEW as data_utils
import features
from eval_utils import evaluate, apply_heuristics

START_TAG = 7
STOP_TAG = 8

# gpu_tracker = MemTracker()

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BERT_BERT_MLP(nn.Module):

    def __init__(self, bert_model_name, chunk_hidden_dim, max_chunk_len, max_seq_len,
                 feat_sz, batch_size, output_dim, use_features=False, bert_freeze=0, templates=None, class_weights=None):
        super(BERT_BERT_MLP, self).__init__()
        self.chunk_hidden_dim = chunk_hidden_dim
        self.max_chunk_len = max_chunk_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.bert_model_name = bert_model_name

        bert_config = AutoConfig.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)

        print("initialing")
        chunk_bert = AutoModel.from_pretrained(bert_model_name)
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer
        self.chunk_encoder = chunk_bert.encoder
        self.chunk_bert = chunk_bert
        self.fc = nn.Linear(768, vocab_size)
        self.crossEntropyLoss = nn.CrossEntropyLoss()

        self.projection = nn.Linear(768+4053, 768)

        self.templates = templates if templates is not None else []

        if bert_freeze > 0:
            # We freeze here the embeddings of the model
            for param in self.bert_model.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[:bert_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.use_features = use_features

    def _get_bert_features(self, x, x_feats, x_len, x_chunk_len):
        self.bert_batch = 8
        # print("_get_bert_features()")
        #print("x", x.shape)

        input_ids = x[:,0,:,:]
        token_ids = x[:,1,:,:]
        attn_mask = x[:,2,:,:]

        max_seq_len = max(x_len)
        #print("x_len", x_len.shape, max_seq_len)
        #print("x_chunk_len", x_chunk_len.shape, x_chunk_len)

        # tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float().cuda()
        tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float()

        idx = 0
        for inp, tok, att, seq_length, chunk_lengths in zip(input_ids, token_ids, attn_mask, x_len, x_chunk_len):
            curr_max = max(chunk_lengths)

            inp = inp[:seq_length, :curr_max]
            tok = tok[:seq_length, :curr_max]
            att = att[:seq_length, :curr_max]
            #print("inp", inp.shape)

            # Run bert over this
            outputs = self.bert_model(inp, attention_mask=att, token_type_ids=tok,
                                      position_ids=None, head_mask=None)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            tensor_seq[idx, :seq_length] = pooled_output
            del outputs

            #print("output", pooled_output.shape)
        #print("tensor_seq.shape", tensor_seq.shape)

        ## concate features
        if self.use_features:
            x_feats = x_feats[:, :max_seq_len, :]
            tensor_seq = torch.cat((tensor_seq, x_feats), 2)

        ## projection
        tensor_seq = self.projection(tensor_seq)

        return tensor_seq

    def _get_batch_list(self, data, batch_size):
        max_len = data.shape[0]
        batch_num = max_len // batch_size
        batch_list = []
        start = 0
        end = 0
        for batch_id in range(int(batch_num)):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            new_data = data[start:end, :, :]
            batch_list.append(new_data)

        if end < max_len:
            new_data = data[end:max_len, :, :]
            batch_list.append(new_data)

        return batch_list

    def _get_chunk_features(self, feats, templates):
        # print("_get_chunk_features()")
        # feats [1, len, 768]

        # preds = torch.zeros((0,self.tokenizer.vocab_size)).cuda()
        preds = torch.zeros((feats.shape[1],self.tokenizer.vocab_size))

        for template in templates:
            # pred = torch.zeros((0,self.tokenizer.vocab_size)).cuda()
            pred = torch.zeros((0,self.tokenizer.vocab_size))
            encode_template = self.tokenizer(template,return_tensors='pt')
            t_list = encode_template['input_ids'].squeeze().tolist()
            tid = t_list.__len__() - t_list.index(103) ## [MASK]
            uid = t_list.__len__() - t_list.index(100) ## [UNK]
            # print(tid)
            # encode_template = encode_template.to('cuda')
            output = self.bert_model(**encode_template)
            h_temp = output.last_hidden_state # h_temp [1, template_len, 768]
            del output

            #### window ####
            # window_size = 33 : 16 + 1 + 16
            # print('with windows')
            l = feats.shape[1]
            for i in range(l):
                left = 0 if i < 16 else i-16
                right = i+16 if i+16 < l-1 else l-1
                x = feats[:, left:right+1, :]
                ## construct template
                h_temp[:,-uid,:] = feats[:, i, :]
                # x = torch.cat((x, feats[:, i, :].unsqueeze(0)), 1)
                x = torch.cat((x, h_temp[:, 1:, :]), 1)
                x =self.chunk_bert(inputs_embeds=x)

                x = self.fc(x[0])
                pred = torch.cat((pred, x[:, -tid, :]), 0)
            preds += pred
        return preds / len(templates)

        #     #### iterate ####
        #     for i in range(feats.shape[1]):
        #         x = torch.cat((feats, feats[:, i, :].unsqueeze(0)), 1)
        #         x = torch.cat((x, h_t[:, 1:, :]), 1)
        #         x = self.chunk_encoder(x)
        #         x = self.fc(x[0])
        #         pred = torch.cat((pred, x[:, -2, :]), 0)
        # return pred

        #     #### batch ####
        #     new_data = torch.zeros((feats.shape[1], feats.shape[1]+h_t.shape[1], 768)).float().cuda()
        #     for i in range(feats.shape[1]):
        #         tmp = torch.cat((feats, feats[:, i, :].unsqueeze(0)), 1)
        #         tmp = torch.cat((tmp, h_t[:, 1:, :]), 1)
        #         new_data[i, :, :] = tmp
        #
        #     batch_size = 4
        #     data_list = self._get_batch_list(new_data, batch_size)
        #
        #     pred = torch.zeros((0,self.tokenizer.vocab_size)).cuda()
        #     for data in data_list:
        #         x = self.chunk_encoder(data)
        #         x = x[0]
        #         logit = self.fc(x)
        #         pred = torch.cat((pred, logit[:, -2, :]), 0)
        #     chunk_feat = pred
        #
        # return chunk_feat

    def _get_chunk_features_soft(self, feats, templates):

        pred = torch.zeros((0,self.tokenizer.vocab_size))
        l = feats.shape[1]
        for i in range(l):
            left = 0 if i < 16 else i-16
            right = i+16 if i+16 < l-1 else l-1
            x = feats[:, left:right+1, :]
            ## construct template
            templates[:,0,:] = feats[:, i, :]
            # x = torch.cat((x, feats[:, i, :].unsqueeze(0)), 1)
            x = torch.cat((x, templates), 1)
            x =self.chunk_bert(inputs_embeds=x)

            x = self.fc(x[0])

            pred = torch.cat((pred, x[:,-9:,:].mean(dim=1)), 0)
        return pred

    def tag2word(self, tag):
        ## label to template words
        # {'B-TRIGGER': 0, 'B-ACTION': 1, 'O': 2, 'B-TRANSITION': 3, 'B-TIMER': 4, 'B-ERROR': 5, 'B-VARIABLE': 6}
        labelwords = ['trigger', 'action', 'other', 'transition', 'time', 'error', 'variable']

        y_word = [labelwords[i] for i in tag ]
        y_out = self.tokenizer(y_word)
        y_ = [x[1] for x in y_out['input_ids']]

        return torch.tensor(y_)

    def word2tag(self, output):
        labelwords = ['trigger', 'action', 'other', 'transition', 'time', 'error', 'variable']
        labelword_ids = [self.tokenizer.encode(x)[1] for x in labelwords]

        pred = output[:,labelword_ids]
        pred = pred.argmax(-1)
        return pred

    def neg_log_likelihood(self, x, x_feats, x_len, x_chunk_len, y, is_soft):
        # print("neg_log_likelihood")

        feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len)
        if is_soft:
            output = self._get_chunk_features_soft(feats, self.templates)
        else:
            output = self._get_chunk_features(feats, self.templates)

        tag = y[0][:x_len[0]]
        tag_word = self.tag2word(tag)

        # loss = self.crossEntropyLoss(output, tag_word.cuda())
        loss = self.crossEntropyLoss(output, tag_word)

        return loss

    def predict_sequence(self, x, x_feats, x_len, x_chunk_len):
        # Get the emission scores from the BiLSTM
        #print("x.shape", x.shape)
        lstm_feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len)
        outputs = []
        for x_len_i, sequence in zip(x_len, lstm_feats):
            #print("sequence.shape", sequence[:x_len_i].shape)
            # Find the best path, given the features.
            score, tag_seq = self._viterbi_decode(sequence[:x_len_i])
            outputs.append(tag_seq)

        return outputs

    def forward(self, x, x_feats, x_len, x_chunk_len):  # dont confuse this with _forward_alg above
        output = self._get_bert_features(x, x_feats, x_len, x_chunk_len)
        return output

def evaluate_bert(model, test_dataloader):
    model.eval()
    total_loss_dev = 0
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        # x = x.cuda()
        # x_feats = x_feats.cuda()
        # x_len = x_len.cuda()
        # x_chunk_len = x_chunk_len.cuda()
        # y = y.cuda()

        model.zero_grad()
        feats = model._get_bert_features(x, x_feats, x_len, x_chunk_len)
        output = model._get_chunk_features(feats, model.templates)

        tag = y[0][:x_len[0]]
        tag_word = model.tag2word(tag)

        # loss = model.crossEntropyLoss(output, tag_word.cuda())
        loss = model.crossEntropyLoss(output, tag_word)
        total_loss_dev += loss.item()

        preds += model.word2tag(output).cpu().data.numpy().tolist()
        labels += tag.cpu().data.numpy().tolist()

    return total_loss_dev / len(test_dataloader), labels, preds

def evaluate_sequences(model, test_dataloader):
    model.eval()
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        # x = x.cuda()
        # x_feats = x_feats.cuda()
        # x_len = x_len.cuda()
        # x_chunk_len = x_chunk_len.cuda()
        # y = y.cuda()

        model.zero_grad()
        feats = model._get_bert_features(x, x_feats, x_len, x_chunk_len)
        output = model._get_chunk_features(feats, model.templates)

        tag = y[0][:x_len[0]]

        preds.append(model.word2tag(output).cpu().data.numpy().tolist())
        labels.append(tag.cpu().data.numpy().tolist())

    return labels, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs_emissions', action='store_true')
    parser.add_argument('--use_transition_priors', action='store_true')
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    parser.add_argument('--printout', default=False, action='store_true')
    parser.add_argument('--features', default=False, action='store_true')
    parser.add_argument('--token_level', default=False, action='store_true', help='perform prediction at token level')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--word_embed_path', type=str)
    parser.add_argument('--word_embed_size', type=int, default=100)
    parser.add_argument('--token_hidden_dim', type=int, default=50)
    parser.add_argument('--chunk_hidden_dim', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--write_results', default=False, action='store_true')
    parser.add_argument('--heuristics', default=False, action='store_true')
    parser.add_argument('--bert_model', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--soft_template', default=False, action='store_true')
    parser.add_argument('--multi_template', default=False, action='store_true')
    parser.add_argument('--template_num', type=int, default=1)
    parser.add_argument('--template_id', type=int, default=0)

    # I am not sure about what this is anymore
    parser.add_argument('--partition_sentence', default=False, action='store_true')
    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    if args.protocol not in protocols:
        print("Specify a valid protocol")
        exit(-1)

    # gpu_tracker.track()
    # if args.cuda_device >= 0:
    #     is_cuda_avail = torch.cuda.is_available()
    #     if not is_cuda_avail:
    #         print("ERROR: There is no CUDA device available, you need a GPU to train this model.")
    #         exit(-1)
    #     elif args.cuda_device >= torch.cuda.device_count():
    #         print("ERROR: Please specify a valid cuda device, you have {} devices".format(torch.cuda.device_count()))
    #         exit(-1)
    #     torch.cuda.set_device('cuda:{}'.format(args.cuda_device))
    #     torch.backends.cudnn.benchmark=True
    # else:
    #     print("ERROR: You need a GPU to train this model. Please specify a valid cuda device, you have {} devices".format(torch.cuda.device_count()))
    #     exit(-1)

    args.savedir_fold = os.path.join(args.savedir, "networking_bert_rfcs_only/checkpoint_{}.pt".format(args.protocol))

    word2id = {}; tag2id = {}; pos2id = {}; id2cap = {}; stem2id = {}; id2word = {}
    transition_counts = {}
    # Get variable and state definitions
    def_vars = set(); def_states = set(); def_events = set(); def_events_constrained = set()
    data_utils.get_definitions(def_vars, def_states, def_events_constrained, def_events)

    together_path_list = [p for p in protocols if p != args.protocol]
    args.train = ["rfcs-bio/{}_phrases_train.txt".format(p) for p in together_path_list]
    args.test = ["rfcs-bio/{}_phrases.txt".format(args.protocol)]

    X_train_data_orig, y_train, level_h, level_d = data_utils.get_data(args.train, word2id, tag2id, pos2id, id2cap, stem2id, id2word, transition_counts, partition_sentence=args.partition_sentence)
    X_test_data_orig, y_test, level_h, level_d = data_utils.get_data(args.test, word2id, tag2id, pos2id, id2cap, stem2id, id2word, partition_sentence=args.partition_sentence)


    def_var_ids = [word2id[x.lower()] for x in def_vars if x.lower() in word2id]
    def_state_ids = [word2id[x.lower()] for x in def_states if x.lower() in word2id]
    def_event_ids = [word2id[x.lower()] for x in def_events if x.lower() in word2id]

    max_chunks, max_tokens = data_utils.max_lengths(X_train_data_orig, y_train)
    max_chunks_test, max_tokens_test = data_utils.max_lengths(X_test_data_orig, y_test)

    max_chunks = max(max_chunks, max_chunks_test)
    max_tokens = max(max_tokens, max_tokens_test)

    print(max_chunks, max_tokens)
    #exit()

    id2tag = {v: k for k, v in tag2id.items()}

    vocab_size = len(stem2id)
    pos_size = len(pos2id)
    X_train_feats = features.transform_features(X_train_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, True)
    X_test_feats = features.transform_features(X_test_data_orig, vocab_size, pos_size, def_var_ids, def_state_ids, def_event_ids, id2cap, id2word, word2id, True)


    X_train_data, y_train, x_len, x_chunk_len = \
        data_utils.bert_sequences(X_train_data_orig, y_train, max_chunks, max_tokens, id2word, args.bert_model)
    X_test_data, y_test, x_len_test, x_chunk_len_test = \
        data_utils.bert_sequences(X_test_data_orig, y_test, max_chunks, max_tokens, id2word, args.bert_model)

    X_train_feats = data_utils.pad_features(X_train_feats, max_chunks)
    X_test_feats = data_utils.pad_features(X_test_feats, max_chunks)

    # Subsample a development set (10% of the data)
    n_dev = int(X_train_data.shape[0] * 0.1)
    dev_excerpt = random.sample(range(0, X_train_data.shape[0]), n_dev)
    train_excerpt = [i for i in range(0, X_train_data.shape[0]) if i not in dev_excerpt]

    X_dev_data = X_train_data[dev_excerpt]
    y_dev = y_train[dev_excerpt]
    x_len_dev = x_len[dev_excerpt]
    X_dev_feats = X_train_feats[dev_excerpt]
    x_chunk_len_dev = x_chunk_len[dev_excerpt]

    X_train_data = X_train_data[train_excerpt]
    y_train = y_train[train_excerpt]
    x_len = x_len[train_excerpt]
    X_train_feats = X_train_feats[train_excerpt]
    x_chunk_len = x_chunk_len[train_excerpt]

    #print(X_train_data.shape, y_train.shape, x_len.shape, x_chunk_len.shape)
    #print(X_dev_data.shape, y_dev.shape, x_len_dev.shape, x_chunk_len_dev.shape)
    print(X_train_data.shape, X_train_feats.shape, y_train.shape, x_len.shape, x_chunk_len.shape)
    print(X_dev_data.shape, X_dev_feats.shape, y_dev.shape, x_len_dev.shape, x_chunk_len_dev.shape)
    feat_sz = X_train_feats.shape[2]
    #print(x_chunk_len)
    #exit()

    print(y_train.shape)

    y_train_ints = list(map(int, y_train.flatten()))
    y_train_ints = [y for y in y_train_ints if y >= 0]

    classes = list(set(y_train_ints))
    print(classes, tag2id)
    class_weights = list(compute_class_weight('balanced', classes=classes, y=y_train_ints))
    # class_weights += [0.0, 0.0] #新加的两个是control_start 和 control_end
    # class_weights = torch.FloatTensor(class_weights).cuda()
    class_weights = torch.FloatTensor(class_weights)

    train_dataset = data_utils.ChunkDataset(X_train_data, X_train_feats, x_len, x_chunk_len, y_train)
    dev_dataset = data_utils.ChunkDataset(X_dev_data, X_dev_feats, x_len_dev, x_chunk_len_dev, y_dev)
    test_dataset = data_utils.ChunkDataset(X_test_data, X_test_feats, x_len_test, x_chunk_len_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # templates
    if args.soft_template:
        templates = torch.empty(1,10,768)
        nn.init.normal_(templates)
    else:
        templates_list = ["[UNK] is a [MASK]",
                          "The type of [UNK] is [MASK]",
                          "[UNK] belongs to [MASK] category",
                          "[UNK] should be tagged as [MASK]"
                          ]
        if args.multi_template:
            templates = templates_list[:args.template_num]
        else:
            templates = [templates_list[args.template_id]]
        for t in templates:
            print(t)

    # gpu_tracker.track()
    # Create model
    model = BERT_BERT_MLP(args.bert_model,
                          args.chunk_hidden_dim,
                          max_chunks, max_tokens, feat_sz, args.batch_size, output_dim=len(tag2id),
                          use_features=args.features, bert_freeze=10, templates=templates, class_weights=class_weights)
    # model.cuda()

    # gpu_tracker.track()

    if args.do_train:

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Training loop
        patience = 0; best_f1 = 0; epoch = 0; best_loss = 10000
        training_statics = ""

        print("TRAINING!!!!")
        while True:
            pbar = tqdm(total=len(train_dataloader))
            model.train()

            total_loss = 0

            for x, x_feats, x_len, x_chunk_len, y in train_dataloader:
                # gpu_tracker.track()
                # x = x.cuda()
                # x_feats = x_feats.cuda()
                # x_len = x_len.cuda()
                # x_chunk_len = x_chunk_len.cuda()
                # y = y.cuda()
                # print(f'chunk_size : {x_len}')

                model.zero_grad()

                loss = model.neg_log_likelihood(x, x_feats, x_len, x_chunk_len, y, args.soft_template)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # torch.cuda.empty_cache()

                pbar.update(1)


            dev_loss, dev_labels, dev_preds = evaluate_bert(model, dev_dataloader)
            test_loss, test_labels, test_preds = evaluate_bert(model, test_dataloader)
            macro_f1 = f1_score(dev_labels, dev_preds, average='macro')
            test_macro_f1 = f1_score(test_labels, test_preds, average='macro')
            if macro_f1 > best_f1:
                #if dev_loss < best_loss:
                # Save model
                #print("Saving model...")
                torch.save(model.state_dict(), args.savedir_fold)
                best_f1 = macro_f1
                best_loss = dev_loss
                patience = 0
            else:
                patience += 1

            training_str = "epoch {} loss {} dev_loss {} dev_macro_f1 {} test_macro_f1 {}".format(
                epoch,
                total_loss / len(train_dataloader),
                dev_loss,
                macro_f1,
                test_macro_f1)
            print(training_str)
            with open('training_each_epochs.txt', 'a') as fp:
                fp.write(training_str)
            training_statics += training_str + '\n'

            epoch += 1
            if patience >= args.patience:
                break

        with open('training_statics.txt', 'w') as fp:
            fp.write(training_statics)

    if args.do_eval:
        # Load model
        model.load_state_dict(torch.load(args.savedir_fold, map_location=lambda storage, loc: storage))

        y_test, y_pred = evaluate_sequences(model, test_dataloader)

        y_test_trans = data_utils.translate(y_test, id2tag)
        y_pred_trans = data_utils.translate(y_pred, id2tag)

        # Do it in a way that preserves the original chunk-level segmentation
        _, y_test_trans_alt, _, _ = data_utils.alternative_expand(X_test_data_orig, y_test_trans, level_h, level_d, id2word, debug=False)
        X_test_data_alt, y_pred_trans_alt, level_h_alt, level_d_alt = data_utils.alternative_expand(X_test_data_orig, y_pred_trans, level_h, level_d, id2word, debug=True)

        # Do it in a way that flattens the chunk-level segmentation for evaluation
        X_test_data_old = X_test_data_orig[:]
        _, y_test_trans_eval = data_utils.expand(X_test_data_orig, y_test_trans, id2word, debug=False)
        X_test_data_eval, y_pred_trans_eval = data_utils.expand(X_test_data_orig, y_pred_trans, id2word, debug=True)


        evaluate(y_test_trans_eval, y_pred_trans_eval)

        def_states_protocol = {}; def_events_protocol = {}; def_events_constrained_protocol = {}; def_variables_protocol = {}
        data_utils.get_protocol_definitions(args.protocol, def_states_protocol, def_events_constrained_protocol, def_events_protocol, def_variables_protocol)

        y_pred_trans_alt = \
            apply_heuristics(X_test_data_alt, y_test_trans_alt, y_pred_trans_alt,
                             level_h_alt, level_d_alt,
                             id2word, def_states_protocol, def_events_protocol, def_variables_protocol,
                             transitions=args.heuristics, outside=args.heuristics, actions=args.heuristics,
                             consecutive_trans=True)

        X_test_data_orig, y_pred_trans, level_h_trans, level_d_trans = \
            data_utils.alternative_join(
                X_test_data_alt, y_pred_trans_alt,
                level_h_alt, level_d_alt,
                id2word, debug=True)

        if args.heuristics:
            _, y_test_trans_eval = data_utils.expand(X_test_data_old, y_test_trans, id2word, debug=False)
            evaluate(y_test_trans_eval, y_pred_trans)


        if args.write_results:
            output_xml = os.path.join(args.outdir, "{}.xml".format(args.protocol))
            results = data_utils.write_results(X_test_data_orig, y_test_trans, y_pred_trans, level_h_trans, level_d_trans,
                                               id2word, def_states_protocol, def_events_protocol, def_events_constrained_protocol,
                                               args.protocol, cuda_device=args.cuda_device)
            with open(output_xml, "w") as fp:
                fp.write(results)

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(4321)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(4321)
    random.seed(4321)

    main()
