import os
from seq2seq_model import Seq2SeqModel
import torch.nn as nn
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *
from sklearn.preprocessing import normalize
from transformers import AutoConfig, AutoModel, BertModel

import data_utils_NEW as data_utils
import features
from eval_utils import evaluate, apply_heuristics

START_TAG = 7
STOP_TAG = 8

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


def evaluate_bert(model, test_dataloader, classes):
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
        batch_p = model.predict_sequence(x, x_feats, x_len, x_chunk_len)
        batch_preds = []
        for p in batch_p:
            batch_preds += p

        batch_y = y.view(-1)
        # Focus on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        label = batch_y.to('cpu').numpy()
        #print(len(batch_preds), label.shape)
        #exit()

        # Accumulate predictions
        preds += list(batch_preds)
        labels += list(label)

        # Get loss
        loss = model.neg_log_likelihood(x, x_feats, x_len, x_chunk_len, y)
        #print(loss.item())
        total_loss_dev += loss.item()
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
        batch_p = model.predict_sequence(x, x_feats, x_len, x_chunk_len)
        batch_preds = []
        for p in batch_p:
            batch_preds += p

        batch_y = y.view(-1)
        # Focus on non-pad elemens
        idx = batch_y >= 0
        batch_y = batch_y[idx]
        label = batch_y.to('cpu').numpy()

        preds.append(list(batch_preds))
        labels.append(list(label))
    return labels, preds

def evaluate_emissions(model, test_dataloader, loss_fn, classes):
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

        logits = model(x, x_feats, x_len, x_chunk_len)
        logits = logits.view(-1, len(classes) + 2)
        batch_y = y.view(-1)

        # Focus the loss on non-pad elemens
        max_len = logits.shape[0]
        batch_y = batch_y[:max_len]

        # Get predictions
        _, batch_preds = torch.max(logits, 1)
        pred = batch_preds.detach().cpu().numpy()
        label = batch_y.to('cpu').numpy()
        # Accumulate predictions

        preds += list(pred)
        labels += list(label)

        # Get loss
        loss = loss_fn(logits, batch_y)
        #print(loss.item())
        total_loss_dev += loss.item()
    return total_loss_dev / len(test_dataloader), labels, preds


def hot_start_emissions(args, model, train_dataloader, dev_dataloader, classes, class_weights):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    patience = 0; best_f1 = 0; epoch = 0; best_loss = 10000
    while True:
        #pbar = tqdm(total=len(train_dataloader))
        model.train()

        total_loss = 0

        for x, x_feats, x_len, x_chunk_len, y in train_dataloader:
            # x = x.cuda()
            # x_feats = x_feats.cuda()
            # x_len = x_len.cuda()
            # x_chunk_len = x_chunk_len.cuda()
            # y = y.cuda()

            model.zero_grad()

            logits = model(x, x_feats, x_len, x_chunk_len)

            logits = logits.view(-1, len(classes) + 2)
            batch_y = y.view(-1)

            # Focus the loss on non-pad elemens
            max_len = logits.shape[0]
            batch_y = batch_y[:max_len]

            loss = loss_fn(logits, batch_y)
            #print(loss.item())
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            #pbar.update(1)

        dev_loss, dev_labels, dev_preds = evaluate_emissions(model, dev_dataloader, loss_fn, classes)
        macro_f1 = f1_score(dev_labels, dev_preds, average='macro')
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

        print("epoch {} loss {} dev_loss {} dev_macro_f1 {}".format(
            epoch,
            total_loss / len(train_dataloader),
            dev_loss,
            macro_f1))
        epoch += 1
        if patience >= args.patience:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hs_emissions', action='store_true')
    parser.add_argument('--use_transition_priors', action='store_true')
    parser.add_argument('--protocol', type=str,  help='protocol', required=True)
    # parser.add_argument('--printout', default=False, action='store_true')
    # parser.add_argument('--features', default=False, action='store_true')
    # parser.add_argument('--token_level', default=False, action='store_true', help='perform prediction at token level')
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--word_embed_path', type=str)
    # parser.add_argument('--word_embed_size', type=int, default=100)
    # parser.add_argument('--token_hidden_dim', type=int, default=50)
    # parser.add_argument('--chunk_hidden_dim', type=int, default=50)
    # parser.add_argument('--patience', type=int, default=5)
    # parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    # parser.add_argument('--outdir', type=str, required=True)
    # parser.add_argument('--write_results', default=False, action='store_true')
    # parser.add_argument('--heuristics', default=False, action='store_true')
    # parser.add_argument('--bert_model', type=str, required=True)
    # parser.add_argument('--learning_rate', type=float, default=1e-1)
    # parser.add_argument('--cuda_device', type=int, default=0)

    # I am not sure about what this is anymore
    parser.add_argument('--partition_sentence', default=False, action='store_true')
    args = parser.parse_args()

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    if args.protocol not in protocols:
        print("Specify a valid protocol")
        exit(-1)

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

    # args.savedir_fold = os.path.join(args.savedir, "networking_bert_rfcs_only/checkpoint_{}.pt".format(args.protocol))

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
    id2tag = {v: k for k, v in tag2id.items()}
    #print(id2tag)

    transition_priors = np.zeros((9, 9))
    for i in transition_counts:
        for j in transition_counts[i]:
            transition_priors[i][j] = transition_counts[i][j]

    transition_priors = normalize(transition_priors, axis=1, norm='l1')#状态转移矩阵先验


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


    X_train_data, y_train, x_len, x_chunk_len =\
        data_utils.bert_sequences(X_train_data_orig, y_train, max_chunks, max_tokens, id2word, args.bert_model)
    X_test_data, y_test, x_len_test, x_chunk_len_test =\
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
    class_weights += [0.0, 0.0] #新加的两个是control_start 和 control_end
    # class_weights = torch.FloatTensor(class_weights).cuda()
    class_weights = torch.FloatTensor(class_weights)

    train_dataset = data_utils.ChunkDataset(X_train_data, X_train_feats, x_len, x_chunk_len, y_train)
    dev_dataset = data_utils.ChunkDataset(X_dev_data, X_dev_feats, x_len_dev, x_chunk_len_dev, y_dev)
    test_dataset = data_utils.ChunkDataset(X_test_data, X_test_feats, x_len_test, x_chunk_len_test, y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # # Create model
    # model = BERT_BiLSTM_CRF(args.bert_model,
    #                         args.chunk_hidden_dim,
    #                         max_chunks, max_tokens, feat_sz, args.batch_size, output_dim=len(tag2id),
    #                         use_features=args.features, bert_freeze=10)

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 50,
        "train_batch_size": 100,
        "num_train_epochs": 20,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": 25,
        "manual_seed": 4,
        "save_steps": 11898,
        "gradient_accumulation_steps": 1,
        "output_dir": "./exp/template",
    }
    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        # encoder_decoder_name="./bart-large",
        args=model_args,
        use_cuda=False,
    )

    # model.cuda()


    if args.do_train:
        if args.hs_emissions:
            hot_start_emissions(args, model, train_dataloader, dev_dataloader, classes, class_weights)
            model.load_state_dict(torch.load(args.savedir_fold, map_location=lambda storage, loc: storage))

        if args.use_transition_priors:
            model.load_transition_priors(transition_priors)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Training loop
        patience = 0; best_f1 = 0; epoch = 0; best_loss = 10000
        #print("TRAINING!!!!")
        while True:
            #pbar = tqdm(total=len(train_dataloader))
            model.train()

            total_loss = 0

            for x, x_feats, x_len, x_chunk_len, y in train_dataloader:
                x = x.cuda()
                x_feats = x_feats.cuda()
                x_len = x_len.cuda()
                x_chunk_len = x_chunk_len.cuda()
                y = y.cuda()

                model.zero_grad()

                loss = model.neg_log_likelihood(x, x_feats, x_len, x_chunk_len, y)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                #pbar.update(1)

            dev_loss, dev_labels, dev_preds = evaluate_bert(model, dev_dataloader, classes)
            test_loss, test_labels, test_preds = evaluate_bert(model, test_dataloader, classes)
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

            print("epoch {} loss {} dev_loss {} dev_macro_f1 {} test_macro_f1 {}".format(
                epoch,
                total_loss / len(train_dataloader),
                dev_loss,
                macro_f1,
                test_macro_f1))
            epoch += 1
            if patience >= args.patience:
                break

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
