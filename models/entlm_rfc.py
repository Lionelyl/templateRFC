import os

import torch.nn as nn
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
# from sklearn.preprocessing import normalize
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM

import data_utils_NEW as data_utils
import features
from eval_utils import evaluate, apply_heuristics


class BERT_BERT_MLP(nn.Module):

    def __init__(self, bert_model_name, chunk_hidden_dim, max_chunk_len, max_seq_len, feat_sz,
                 batch_size, output_dim, use_features=False, bert_freeze=0, templates=None, device='cpu'):
        super(BERT_BERT_MLP, self).__init__()
        self.chunk_hidden_dim = chunk_hidden_dim
        self.max_chunk_len = max_chunk_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.bert_model_name = bert_model_name
        self.device = device

        bert_config = AutoConfig.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)

        print("initialing")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.chunk_bert = AutoModelForMaskedLM.from_pretrained(bert_model_name)
        # self.fc = nn.Linear(768, vocab_size)
        self.crossEntropyLoss = nn.CrossEntropyLoss()

        self.projection = nn.Linear(768+feat_sz, 768)

        self.templates = templates if templates else []

        if bert_freeze > 0:
            # We freeze here the embeddings of the model
            for param in self.bert_model.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert_model.encoder.layer[:bert_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.use_features = use_features

    def _get_bert_features(self, x, x_feats, x_len, x_chunk_len, device):
        self.bert_batch = 8
        # print("_get_bert_features()")
        #print("x", x.shape)

        input_ids = x[:,0,:,:]
        token_ids = x[:,1,:,:]
        attn_mask = x[:,2,:,:]

        max_seq_len = max(x_len)
        #print("x_len", x_len.shape, max_seq_len)
        #print("x_chunk_len", x_chunk_len.shape, x_chunk_len)

        tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float().to(device)
        # tensor_seq = torch.zeros((len(x_len), max_seq_len, 768)).float()

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

    def _get_chunk_features(self, feats, templates, device):
        # print("_get_chunk_features()")
        # chunk_feat = torch.zeros((0,self.tokenizer.vocab_size)).to(device)
        preds = torch.zeros((feats.shape[1],self.tokenizer.vocab_size)).to(device)

        for template in templates:
            pred = torch.zeros((0,self.tokenizer.vocab_size)).to(device)
            encode_template = self.tokenizer(template,return_tensors='pt')
            t_list = encode_template['input_ids'].squeeze().tolist()
            tid = t_list.__len__() - t_list.index(103) ## [MASK]
            uid = t_list.__len__() - t_list.index(100) ## [UNK]
            # print(tid)
            encode_template = encode_template.to(device)
            output = self.bert_model(**encode_template)
            h_temp = output.last_hidden_state
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

    def neg_log_likelihood(self, x, x_feats, x_len, x_chunk_len, y, device):
        # print("neg_log_likelihood")

        feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len, device)
        output = self._get_chunk_features(feats, self.templates, device)

        tag = y[0][:x_len[0]]
        tag_word = self.tag2word(tag)

        loss = self.crossEntropyLoss(output, tag_word.to(device))

        return loss

    def forword_entlm(self, x, x_feats, x_len, x_chunk_len, y, device):

        feats = self._get_bert_features(x, x_feats, x_len, x_chunk_len, device)
        y = y[0][:x_len[0]]
        output = self.chunk_bert(inputs_embeds=feats, labels=y)

        return output.loss, output.logits

    def forward(self, x, x_feats, x_len, x_chunk_len,):  # dont confuse this with _forward_alg above
        output = self._get_bert_features(x, x_feats, x_len, x_chunk_len, self.device)
        return output

def evaluate_entlm(model, test_dataloader, device, tag2index):
    label_indexs = [v for k, v in tag2index.items()]
    model.eval()
    total_loss_dev = 0
    preds = []; labels = []
    for x, x_feats, x_len, x_chunk_len, y in test_dataloader:
        x = x.to(device)
        x_feats = x_feats.to(device)
        x_len = x_len.to(device)
        x_chunk_len = x_chunk_len.to(device)
        y = y.to(device)

        model.zero_grad()
        loss, logits = model.forword_entlm(x, x_feats, x_len, x_chunk_len, y, device)

        tag = y[0][:x_len[0]]

        total_loss_dev += loss.item()
        label_logits = logits.squeeze(0).cpu()
        label_logits = label_logits[:, label_indexs]
        pred = label_logits.argmax(-1)
        # pred = [tag2index['O'] if ll[pred[i]]<0.5 else label_indexs[pred[i].item()] for i, ll in enumerate(label_logits)]
        pred = [label_indexs[p.item()] for p in pred]


        # preds += pred
        # labels += tag.cpu().data.numpy().tolist()
        preds.append(pred)
        labels.append(tag.cpu().data.numpy().tolist())

    return total_loss_dev / len(test_dataloader), labels, preds

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
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--multi_template', default=False, action='store_true')
    parser.add_argument('--template_num', type=int, default=1)
    parser.add_argument('--template_id', type=int, default=0)
    parser.add_argument('--label_map_path', type=str)
    parser.add_argument('--handmaded_label_map_path', type=str)
    parser.add_argument('--warmup', default=False, action='store_true')

    # I am not sure about what this is anymore
    parser.add_argument('--partition_sentence', default=False, action='store_true')

    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    if args.protocol not in protocols:
        print("Specify a valid protocol")
        exit(-1)

    device = 'cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu'

    args.savedir_fold = os.path.join(args.savedir, "output/checkpoint_EntLM_{}_{}_{}.pt".format("handmade+data_search" if args.handmaded_label_map_path else "data_search",
                                                                                                "warmup" if args.warmup else "",
                                                                                                args.protocol))

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
    feat_sz = X_train_feats[0].shape[1]

    # templates
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
    # Create model
    model = BERT_BERT_MLP(args.bert_model,
                          args.chunk_hidden_dim,
                          max_chunks, max_tokens, feat_sz, args.batch_size, output_dim=len(tag2id),
                          use_features=args.features, bert_freeze=10, templates=templates, device=device)
    model.to(device)

    ## label to template words
    # {'B-TRIGGER': 0, 'B-ACTION': 1, 'O': 2, 'B-TRANSITION': 3, 'B-TIMER': 4, 'B-ERROR': 5, 'B-VARIABLE': 6}
    labelwords = ['trigger', 'action', 'other', 'transition', 'timer', 'error', 'variable']
    labelwords = [ 'B-'+lw.upper() for lw in labelwords]

    tag2index = {}

    tokenizer = model.tokenizer
    data_utils.add_label_token_bert(model, tokenizer, args.protocol, args.label_map_path, args.handmaded_label_map_path, tag2index)
    print(tag2index)

    X_train_data, y_train, x_len, x_chunk_len = \
        data_utils.bert_sequences(X_train_data_orig, y_train, max_chunks, max_tokens, id2word, tokenizer, tag2index, id2tag)
    X_test_data, y_test, x_len_test, x_chunk_len_test = \
        data_utils.bert_sequences(X_test_data_orig, y_test, max_chunks, max_tokens, id2word, tokenizer, tag2index, id2tag)

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

    print(X_train_data.shape, X_train_feats.shape, y_train.shape, x_len.shape, x_chunk_len.shape)
    print(X_dev_data.shape, X_dev_feats.shape, y_dev.shape, x_len_dev.shape, x_chunk_len_dev.shape)

    #print(x_chunk_len)
    #exit()

    print(y_train.shape)

    y_train_ints = list(map(int, y_train.flatten()))
    y_train_ints = [y for y in y_train_ints if y >= 0]

    classes = list(set(y_train_ints))
    print(classes, tag2id)

    train_dataset = data_utils.ChunkDataset(X_train_data, X_train_feats, x_len, x_chunk_len, y_train)
    dev_dataset = data_utils.ChunkDataset(X_dev_data, X_dev_feats, x_len_dev, x_chunk_len_dev, y_dev)
    test_dataset = data_utils.ChunkDataset(X_test_data, X_test_feats, x_len_test, x_chunk_len_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.do_train:
        n_epoch = 100
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-5,
            div_factor=2,
            final_div_factor=10,
            epochs=n_epoch,
            steps_per_epoch=1,
            pct_start=0.2)

        # Training loop
        patience = 0; best_f1 = 0; epoch = 0; best_loss = 10000
        training_statics = ""


        print("TRAINING!!!!")
        while epoch < n_epoch:
            pbar = tqdm(total=len(train_dataloader))
            model.train()

            total_loss = 0

            for x, x_feats, x_len, x_chunk_len, y in train_dataloader:
                x = x.to(device)
                x_feats = x_feats.to(device)
                x_len = x_len.to(device)
                x_chunk_len = x_chunk_len.to(device)
                y = y.to(device)
                # print(f'chunk_size : {x_len}')

                model.zero_grad()

                loss, _ = model.forword_entlm(x, x_feats, x_len, x_chunk_len, y, device)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                torch.cuda.empty_cache()

                pbar.update(1)

            if args.warmup and scheduler:
                print(f'lr: {optimizer.param_groups[0]["lr"]}')
                scheduler.step()

            dev_loss, dev_labels, dev_preds = evaluate_entlm(model, dev_dataloader, device, tag2index)
            test_loss, test_labels, test_preds = evaluate_entlm(model, test_dataloader, device, tag2index)
            dev_labels = data_utils.flatten(dev_labels)
            dev_preds = data_utils.flatten(dev_preds)
            test_labels = data_utils.flatten(test_labels)
            test_preds = data_utils.flatten(test_preds)
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

            training_str = "\nepoch {} loss {} dev_loss {} dev_macro_f1 {} test_macro_f1 {}".format(
                epoch,
                total_loss / len(train_dataloader),
                dev_loss,
                macro_f1,
                test_macro_f1)
            print(training_str)
            training_statics += training_str

            epoch += 1
            if patience >= args.patience:
                break

    if args.do_eval:
        # Load model
        model.load_state_dict(torch.load(args.savedir_fold, map_location=lambda storage, loc: storage))

        _, y_test, y_pred = evaluate_entlm(model, test_dataloader, device, tag2index)

        y_test_trans = data_utils.translate_entlm(y_test, tag2index)
        y_pred_trans = data_utils.translate_entlm(y_pred, tag2index)

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
