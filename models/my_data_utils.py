import nltk
from nltk import word_tokenize
import re
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils import data
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm, trange

class ChunkDataset(data.Dataset):
    def __init__(self, x, x_feats, x_len, x_chunk_len, labels=None):
        self.x = torch.from_numpy(x).long()
        self.x_feats = torch.from_numpy(x_feats).float()
        self.x_len = torch.from_numpy(x_len).long()
        self.x_chunk_len = torch.from_numpy(x_chunk_len).long()
        self.labels = torch.from_numpy(labels).long() if labels else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_i = self.x[index]
        x_feats_i = self.x_feats[index]
        x_len_i = self.x_len[index]
        label = self.labels[index] if self.labels else None
        x_chunk_len_i = self.x_chunk_len[index]
        if label:
            return x_i, x_feats_i, x_len_i, x_chunk_len_i, label
        else:
            return x_i, x_feats_i, x_len_i, x_chunk_len_i

def get_data(files, word2id, id2word, id2cap):
    controls = []
    control_str = ""
    for f in files:
        with open(f, 'r') as fp:
            for line in fp:
                # if line != '\n':
                if line != '======\n':
                    control_str += line + " "
                else:
                    controls.append(control_str)
                    control_str = ""
            if control_str:
                controls.append(control_str)

    next_id = len(word2id)
    if next_id == 0:
        word2id["[PAD]"] = 0
    next_id = 1
    pred_data =[]; pred_level_h = []; pred_level_d = []
    level_h = []; level_d =[]
    x_control = []; x_chunk = [[]]

    for control in controls:
        tokens = word_tokenize(control)
        for i, token in enumerate(tokens):
            if is_puntuaction(token): # end of chunk
                next_id = get_token(token, next_id, word2id, id2word, id2cap)
                x_chunk[0].append(word2id[token])
                # 默认设置为0
                level_h.append(0)
                level_d.append(0)

                x_control.append(x_chunk)
                x_chunk = [[]]
            else:
                next_id = get_token(token, next_id, word2id, id2word, id2cap)
                x_chunk[0].append(word2id[token])

        if len(x_control) > 0:
            pred_data.append(x_control)
            pred_level_h.append(level_h)
            pred_level_d.append(level_d)

            curr_contr = [x[0] for x in x_control]
            curr_contr_str = ""
            for chunk in curr_contr:
                curr_contr_str += " ".join([id2word[w] for w in chunk]) + " | "
            x_control = []
            level_h = []; level_d = []
    return pred_data, pred_level_h, pred_level_d

def get_token(token, next_id, word2id, id2word, id2cap):
    if token not in word2id:
        word2id[token] = next_id
        id2word[next_id] = token
        next_id += 1
    if token.lower() not in word2id:
        word2id[token.lower()] = next_id
        id2word[next_id] = token.lower()
        next_id += 1
    # 0 for lower or capitalized, 1 for all caps, 2 for capitalized variable, 3 for camel case, 4 other form of apha, 5 numbers, 6 symbols
    if token.islower() or re.match(r'^[A-Z]{1}[a-z_-]+$', token):
        id2cap[word2id[token]] = 0
    elif token.isupper():
        id2cap[word2id[token]] = 1
    elif re.match(r'^([A-Z]{1}[a-z_-]+)+$', token):
        id2cap[word2id[token]] = 2
    elif re.match(r'^[a-z]+([A-Z][a-z_-]+)+$', token):
        id2cap[word2id[token]] = 3
    elif re.match(r'[a-zA-Z]+[a-zA-Z_-]+', token):
        id2cap[word2id[token]] = 4
    elif re.match(r'[0-9]+',  token):
        id2cap[word2id[token]] = 5
    else:
        id2cap[word2id[token]] = 6
    return next_id

def is_puntuaction(token):
    if token in [',', '.', ';', '?', '!']:
        return True
    return False

def  max_lengths(X):
    # First calculate the max
    max_chunks = 0; max_tokens = 0
    for control in X:
        max_chunks = max(max_chunks, len(control))
        for chunk in control:
            max_tokens = max(max_tokens, len(chunk[0]))
    return max_chunks, max_tokens

def bert_sequences(X,  max_chunks, max_tokens, id2word, bert_model):
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    X_new = []
    x_len = []; x_chunk_len = []
    #print("max_chunks", max_chunks, "max_tokens", max_tokens)

    for x in X:
        chunks_new = []; curr_chunk_len = []
        input_ids = []; token_ids = []; attention_masks = []
        for elems in x:
            tokens = " ".join([id2word[i] for i in elems[0]])
            #print(tokens)
            encoded_input = tokenizer(tokens, padding='max_length', max_length=max_tokens, truncation=True)
            input_ids.append(encoded_input['input_ids'])
            token_ids.append(encoded_input['token_type_ids'])
            attention_masks.append(encoded_input['attention_mask'])
            #print(len(encoded_input['input_ids']), len(encoded_input['token_type_ids']), len(encoded_input['attention_mask']))
            curr_len = encoded_input['attention_mask'].count(1)
            curr_chunk_len.append(curr_len)
        x_chunk_len.append(curr_chunk_len + [0] * (max_chunks - len(curr_chunk_len)))
        #print(np.array(input_ids).shape, np.array(token_ids).shape, np.array(attention_masks).shape)
        chunks_new = np.array([input_ids, token_ids, attention_masks])
        #print(chunks_new.shape)
        n_chunks = len(input_ids)
        if n_chunks < max_chunks:
            padded_chunks = np.zeros((3, max_chunks - n_chunks, max_tokens))
            chunks_new = np.concatenate((chunks_new, padded_chunks), axis=1)

        x_len.append(n_chunks)
        X_new.append(chunks_new)

    X_new = np.array(X_new)
    x_len = np.array(x_len)
    x_chunk_len = np.array(x_chunk_len)

    return X_new, x_len, x_chunk_len

def write_definitions(def_states, def_events):
    ret_str = ""
    for state in def_states:
        ret_str += '<def_state id="{}">{}</def_state>\n'.format(def_states[state], state)
    for event in def_events:
        if event in ['acknowledgment', 'reset']:
            continue
        ret_str += '<def_event id="{}">{}</def_event>\n'.format(def_events[event], event)
    return ret_str

def same_span(tag, prev_tag):
    if tag and prev_tag and tag[2:] == prev_tag[2:] and tag[0] == 'I':
        return True
    return False

def tag_lookahead(index, x_trans, y_p, current_tag):
    j = index + 1; tag_str = x_trans[index] + " "
    if j < len(y_p):
        current_tag_lookahead = y_p[j]
        while (same_span(current_tag_lookahead, current_tag)):
            tag_str += x_trans[j] + " "
            j += 1
            if j >= len(y_p):
                break
            current_tag_lookahead = y_p[j]
    tag_str = tag_str.replace("&newline;", "")
    return tag_str

def find_acknowledgment(tag_str, type_ref):
    tag_split = tag_str.split()
    if 'acked' in tag_split:
        ack_word_index = tag_split.index('acked')
    elif 'acknowledged' in tag_split:
        ack_word_index = tag_split.index('acknowledged')
    else:
        # In case of acknowledgement of ... take the whole phrase
        ack_word_index = len(tag_split) - 1
    tag_split = tag_split[:ack_word_index+1]
    ack_tags = ['B-ACK'] + ['I-ACK'] * (len(tag_split) - 1)
    ack_type = type_ref
    #print(tag_split)
    return ack_tags, ack_type

def last_was_trigger(seq_tags, seq_words):
    rev_tags = seq_tags[::-1];
    rev_words = seq_words[::-1];
    i = 0
    while (i < len(rev_tags) and rev_tags[i] == 'O' and (rev_words[i] == "&newline;" or rev_words[i].isspace())):
        i += 1
    if i < len(rev_tags) and rev_tags[i].endswith('TRIGGER'):
        return True
    return False

def guess_recursive_controls(index, y_p, x_trans, current_tag, offset, offset_str, open_recursive_control, open_continued_control):
    ret_str = ""
    if current_tag[2:].lower() == "trigger": #and x_trans[index:][0] != "otherwise":
        num_triggers = len([p for p in y_p[index:] if p.endswith('TRIGGER')])
        if num_triggers != len(y_p[index:]):
            #print(num_trigers)
            #print([p for p in y_p[index:] if p.endswith('TRIGGER')])
            #print(y_p[index:])
            #print(x_trans[index:])
            #print(x_trans[index-1])
            #print(num_triggers, len(y_p[index:]))
            #if open_continued_control:
            #    ret_str += offset_str + "".join(["\t"] * offset) + "\n</control_GR>"
            #    open_continued_control = False
            if not ret_str.endswith('<control relevant="true">'):
                if not last_was_trigger(y_p[:index], x_trans[:index]):
                    offset_str = "".join(["\t"] * offset)
                    if open_continued_control:
                        ret_str += '\n' + offset_str + "</control>"
                    ret_str += '\n' + offset_str + '<control relevant="true">'
                    open_continued_control = True
                else:
                    offset += 1
                    offset_str = "".join(["\t"] * offset)
                    ret_str += '\n' + offset_str + '<control relevant="true">'
                    open_recursive_control = True

    return open_recursive_control, open_continued_control, offset_str, ret_str, offset

def srl_events(predictor, action_str, current_tag, def_states, protocol):
    tag_type = None; tags = []
    if not re.match('.*(syn-sent|syn-received).*', action_str):
        srl = predictor.predict(sentence=action_str)
        #print(srl['words'])
        #print('------')

        # don't break and keep the last observed verb (are they in order? make sure!)
        for verb in srl['verbs']:
            if verb['verb'].startswith('send') or verb['verb'].startswith('sent'):
                tags = verb['tags']
                tag_type = "send"
                break
            elif verb['verb'].startswith('receiv'):
                tags = verb['tags']
                tag_type = "receive"
                break
            elif verb['verb'].startswith('issu') or verb['verb'].startswith('form') or verb['verb'].startswith('generat'):
                ## WARNING -- cheating!
                #if "reset" in action_str:
                #    pass
                #else:
                tags = verb['tags']
                if "send" in action_str or "sent" in action_str and "receiv" not in action_str:
                    tag_type = "send"
                    print("send", action_str)
                elif "send" in action_str:
                    print("HERE", action_str)
                elif "receive" in action_str and "send" not in action_str and "receiv" not in action_str:
                    tag_type = "receive"
                    print("receive", action_str)
                else:
                    tag_type = "issue"
                #print(action_str)
                break
            #elif verb['verb'].startswith('ack'):
            #    tag_type = "receive"
            elif 'handshake' in srl['words']:
                tag_type = "send"
            #elif current_tag.endswith("ACTION") and any_defs(action_str.split(), def_states):
            #else:
            #    print("RECEIVE", action_str)
            #    print(action_str, current_tag)
            #    tag_type = "receive"
            #    #print(verb['verb'])
            #    pass
            '''
            elif "send" in action_str or "sent" in action_str and "receiv" not in action_str:
                print("SEND", action_str)
                tag_type = "send"
            elif "receiv" in action_str and "send" not in action_str and "sent" not in action_str:
                print("RECEIVE", action_str)
                tag_type = "receive"
            '''
    if tag_type is None and current_tag.endswith('ACTION') and "do not" not in action_str and "do n ` t" not in action_str:
        #print("\tRECEIVE", action_str)
        tag_type = "receive"

    return tags, tag_type

def srl_transitions(predictor, transition_str, def_states):
    tag_type = None; tags = []
    srl = predictor.predict(sentence=transition_str)
    #print(transition_str)
    #print(srl)
    #print("-------")
    for verb in srl['verbs']:
        if verb['verb'].startswith('mov') or \
                verb['verb'].startswith('enter') or \
                verb['verb'].startswith('transition') or \
                verb['verb'].startswith('go') or \
                verb['verb'].startswith('chang') or \
                verb['verb'].startswith('leav') or \
                verb['verb'].startswith('remain') or \
                verb['verb'].startswith('stay'):

            tags = verb['tags']

            if verb['verb'].startswith('leav'):
                print(srl)

            break

    transition_splits = transition_str.split()
    '''
    if transition_splits[0] in ["to", "from"] and transition_splits[1] in def_states and len(tags) == 0:
        tags = ['O'] * len(transition_splits)
        tags[0] = 'B-ARGM-DIR'; tags[1] = "I-ARGM-DIR"
    '''
    return tags, tag_type

def overlap(word, definitions):
    splits = word.split('-')
    if len(splits) == 2:
        word_sp = splits[1]
    else:
        word_sp = splits[0]

    for _def in definitions:
        def_splits = _def.split('-')
        if len(def_splits) == 2:
            def_sp = def_splits[1]
        else:
            def_sp = def_splits[0]

        if word == _def or word == def_sp or word_sp == _def or word_sp == def_sp:
            definitions[word] = definitions[_def]
            print(word, _def)
            return True
    return False

def write_results(X_test_data, y_test_trans, y_pred_trans, level_h_trans, level_d_trans, id2word, def_states, def_events, def_events_constrained, protocol, cuda_device):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")#, cuda_device=cuda_device)

    ret_str = "<p>"

    # Write down definitions
    #print(def_events_constrained)
    ret_str += write_definitions(def_states, def_events_constrained)

    # Iterate over sequences
    n_controls = len(y_pred_trans)
    pbar = tqdm(total=n_controls)

    for x, y_g, y_p, h, d in zip(X_test_data, y_test_trans, y_pred_trans, level_h_trans, level_d_trans):

        cur_str =  '\n<control relevant="true">'
        ret_str += '\n<control relevant="true">'

        x_trans = [id2word[i].lower() for i in x]

        #print("H", [(a,b,c) for (a,b,c) in zip(x_trans, h, y_p)])
        #print("////////////")
        #print("D", [(a,b,c) for (a,b,c) in zip(x_trans, d, y_p)])
        #exit()

        #open_recursive_control = False
        #open_continued_control = False
        prev_tag = None
        prev_h = 0; prev_d = 0; prev_offset = 1
        num_open_control = {1: 1}
        i = 0
        ack_tags = []; ack_type = None
        open_arg = False;
        explicit_source = False; explicit_target = False; explicit_intermediate = False
        explicit_state = False; explicit_type = False

        tag_type = None; tags = []

        while (i < len(x_trans)):
            offset = d[i];
            #print(x_trans[i], offset)

            word = x_trans[i]
            pred_tag = y_p[i]

            current_tag = pred_tag
            if prev_tag is not None and not same_span(current_tag, prev_tag) and prev_tag != 'O':
                if prev_tag[2:].lower() == 'action' and open_arg:
                    ret_str += "</arg> "; open_arg = False
                    cur_str += "</arg> "; open_arg = False
                    tag_type = None

                offset_str = "".join(['\t'] * offset)
                ret_str += offset_str + "</{}>".format(prev_tag[2:].lower())
                cur_str += offset_str + "</{}>".format(prev_tag[2:].lower())
                closed = True

            if current_tag is None or (not same_span(current_tag, prev_tag) and current_tag != 'O'):

                '''
                # Guess recursive controls
                open_recursive_control, open_continued_control, offset_str, new_ret_str, offset =\
                    guess_recursive_controls(i, y_p, x_trans, current_tag, offset, offset_str,
                                             open_recursive_control, open_continued_control)
                ret_str += new_ret_str
                '''
                # Use offsets to identify recursive controls
                if prev_h > 0 and prev_d > 0 and \
                        (h[i] > prev_h and prev_d == d[i]):
                    offset_str = "".join(['\t'] * (offset-1))
                    ret_str += "\n" + offset_str + '</control>'
                    cur_str += "\n" + offset_str + '</control>'
                    #print("------")
                    #print(cur_str)
                    #print("removing", offset, num_open_control)
                    num_open_control[offset] -= 1

                if prev_h > 0 and prev_d > 0 and \
                        (d[i] < prev_d):
                    if prev_h > 0 and h[i] > prev_h:
                        relevant_offsets = [index for index in num_open_control if num_open_control[index] > 0 and index >= offset]
                    else:
                        relevant_offsets = [index for index in num_open_control if num_open_control[index] > 0 and index > offset]
                    relevant_offsets.sort(reverse=True)
                    for open_control in relevant_offsets:
                        offset_str = "".join(['\t'] * (open_control-1))
                        ret_str += "\n" + offset_str + "</control>"
                        cur_str += "\n" + offset_str + "</control>"
                        num_open_control[open_control] -= 1

                if prev_h > 0 and h[i] > prev_h:
                    offset_str = "".join(['\t'] * (offset-1))
                    ret_str += "\n\n" + offset_str + '<control relevant="true">'
                    cur_str += "\n\n" + offset_str + '<control relevant="true">'
                    if offset not in num_open_control:
                        num_open_control[offset] = 0
                    num_open_control[offset] += 1
                    #print("adding", offset, num_open_control)
                tag_type = None; tags = []

                #if current_tag[2:].lower() in ["action", "trigger"]:
                # Do a look-ahead
                action_str = tag_lookahead(i, x_trans, y_p, current_tag)
                tags, tag_type = srl_events(predictor, action_str, current_tag, def_states, protocol)

                if current_tag[2:].lower() == "transition":
                    # Do a look-ahead
                    transition_str = tag_lookahead(i, x_trans, y_p, current_tag)
                    tags, tag_type = srl_transitions(predictor, transition_str, def_states)

                if tag_type is None or current_tag[2:].lower() != "action":
                    offset_str = "".join(['\t'] * offset)
                    ret_str += "\n" + offset_str + "<{}>".format(current_tag[2:].lower())
                    cur_str += "\n" + offset_str + "<{}>".format(current_tag[2:].lower())
                else:
                    offset_str = "".join(['\t'] * offset)
                    ret_str += "\n" + offset_str + '<{} type="{}">'.format(current_tag[2:].lower(), tag_type)
                    cur_str += "\n" + offset_str + '<{} type="{}">'.format(current_tag[2:].lower(), tag_type)
                closed = False

            word = word.replace("&newline;", "")
            word = word.replace("&", "&amp;")
            word = word.replace("<", "&lt;")
            word = word.replace(">", "&gt;")

            if word.endswith("type"):
                explicit_type = True
            if word.endswith("state"):
                explicit_state = True

            # Add opening tag for identified argument in case of actions
            if current_tag[2:].lower() == 'action' and len(tags) > 0 and tags[0] == 'B-ARG1':
                ret_str += "<arg> "
                cur_str += "<arg> "
                open_arg = True

            # Check explicit source/target in transition tags
            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and (tags[0] in ['B-ARGM-DIR', 'B-ARG2', 'B-ARG1', 'I-ARGM-DIR', 'I-ARG2', 'I-ARG1', 'B-ARGM-PRD', 'I-ARGM-PRD']) and word in ["to", "for"]:
                explicit_target = True
                #print("TO | ", transition_str, tags)

            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and (tags[0] in ['B-ARGM-DIR', 'B-ARG2', 'B-ARG1', 'I-ARGM-DIR', 'I-ARG2', 'I-ARG1', 'B-ARGM-PRD', 'I-ARGM-PRD']) and word == "from":
                explicit_source = True

            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and tags[0] in ['B-V', 'I-V'] and word.startswith('leav'):
                explicit_source = True

            if current_tag[2:].lower() == 'transition' and len(tags) > 0 and (tags[0] in ['B-ARGM-DIR', 'B-ARG2', 'B-ARG1', 'I-ARGM-DIR', 'I-ARG2', 'I-ARG1', 'B-ARGM-PRD', 'I-ARGM-PRD']) and word == "through":
                explicit_intermediate = True
                #print("FROM | ", transition_str, tags)

            #if current_tag[2:].lower() in ['trigger', 'transition'] and overlap(word, def_states):
            if current_tag[2:].lower() in ['trigger', 'transition'] and word in def_states and (not explicit_type):
                tagged_word = ""

                # If source/target explicit, open tag
                if explicit_target:
                    tagged_word += "<arg_target>"
                if explicit_source:
                    tagged_word += "<arg_source>"
                if explicit_intermediate:
                    tagged_word += "<arg_intermediate>"

                # Write the state
                tagged_word += '<ref_state id="{}">{}</ref_state>'.format(def_states[word], word)
                explicit_type = False

                # If source/target explicit, close the tag and turn off the flags
                if explicit_source:
                    tagged_word += "</arg_source>"
                if explicit_target:
                    tagged_word += "</arg_target>"
                if explicit_intermediate:
                    tagged_word += "</arg_intermediate>"
                explicit_source = False; explicit_target = False; explicit_intermediate = False;
                word = tagged_word
                #print(word)

            if word.endswith('msl') and 'timeout' in def_events:
                word = '<ref_event id="{}" type="compute">{}</ref_event>'.format(def_events['timeout'], word)

            #elif current_tag[2:].lower() in ['trigger', 'action'] and overlap(word, def_events) and word not in ["send", "receive"]:
            elif word in def_events and word not in ["send", "receive"]:

                # Do a lookahead to see if we have an acknowledgment phrase
                if len(ack_tags) > 0 and ack_tags[0] in ['B-ACK', 'I-ACK']:
                    pass
                else:
                    tag_str = tag_lookahead(i, x_trans, y_p, current_tag)
                    type_ref = tag_type

                    if type_ref == "issue":
                        type_ref = "send"
                    if current_tag[2:].lower() == 'trigger' and type_ref is None:
                        type_ref = "receive"
                    if re.match('[\s\w]*( acked| acknowledged|acknowledgment of|acknowledge).*', tag_str) and protocol == "TCP":
                        ack_tags, ack_type = find_acknowledgment(tag_str, type_ref)
                        print(tag_str, ack_tags, tags, ack_type)
                    else:
                        word = '<ref_event id="{}" type="{}">{}</ref_event>'.format(def_events[word], type_ref, word)
                        explicit_type = False

            if current_tag[2:].lower() in ['action', 'trigger'] and len(ack_tags) > 0 and ack_tags[0] == 'B-ACK':
                ret_str += '<ref_event id="{}" type="{}">'.format(def_events['ack'], ack_type, word)
                cur_str += '<ref_event id="{}" type="{}">'.format(def_events['ack'], ack_type, word)

            # Add the actual word
            ret_str += word + " "
            cur_str += word + " "

            if current_tag[2:].lower() in ['action', 'trigger'] and len(ack_tags) == 1 and ack_tags[0] == 'I-ACK':
                ret_str += '</ref_event>'
                cur_str += '</ref_event>'

            '''
                Deal with action cases below
                We need to identify all the cases in which we are closing the ARG 
            '''
            if current_tag[2:].lower() == 'action' and len(tags) == 1 and tags[0] == 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 1", tags[0], tags)
            elif current_tag[2:].lower() == 'action' and len(tags) == 2 and word == '``' and tags[0] == 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) == 3 and re.match(r'\w+(\=|\-|\/)\w+', word) and tags[0] == 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) > 1 and tags[0] == 'I-ARG1' and tags[1] != 'I-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 2", tags[0], tags)
            elif current_tag[2:].lower() == 'action' and len(tags) == 1 and tags[0] == 'B-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 3", tags[0], tags)
            elif current_tag[2:].lower() == 'action' and len(tags) == 2 and word == '``' and tags[0] == 'B-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) == 3 and re.match(r'\w+(\=|\-|\/)\w+', word) and tags[0] == 'B-ARG1':
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            elif current_tag[2:].lower() == 'action' and len(tags) > 1 and tags[0] == 'B-ARG1' and tags[1] != "I-ARG1":
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
                #print("Closing 4", tags[0], tags)


            # Move two positions to account for the fact that the SRL separates `` into two tokens
            # It also happens when we have a composed word with symbols in the middle
            if len(tags) > 0 and word == '``':
                tags = tags[2:]
            elif len(tags) > 0 and re.match(r'\w+(\=|\-|\/)\w+', word):
                tags = tags[3:]
            elif len(tags) > 0:
                tags = tags[1:]

            if len(ack_tags) > 0:
                ack_tags = ack_tags[1:]

            prev_tag = current_tag
            prev_h = h[i]
            if current_tag != 'O':
                prev_d = d[i]
                prev_offset = offset
            i += 1

        if not closed and prev_tag != 'O':
            if prev_tag[2:].lower() == 'action' and open_arg:
                ret_str += "</arg> "; open_arg = False
                cur_str += "</arg> "; open_arg = False
                #tag_type = None
            offset_str = "".join(['\t'] * offset)
            ret_str += offset_str + "</{}>".format(prev_tag[2:].lower())
            cur_str += offset_str + "</{}>".format(prev_tag[2:].lower())

        relevant_offsets = [index for index in num_open_control if num_open_control[index] > 0]
        relevant_offsets.sort(reverse=True)

        for open_control in relevant_offsets:
            offset_str = "".join(['\t'] * (open_control-1))
            ret_str += "\n" + offset_str + "</control>"
            cur_str += "\n" + offset_str + "</control>"
        num_open_control[open_control] -= 1

        ret_str += "\n"
        cur_str += "\n"
        pbar.update(1)
        #print(cur_str)
        #print("---------------")

        #exit()
        #print(num_recursive_controls, '----------')
    pbar.close()

    #if ret_str.endswith("<error>. </control>\n"):
    #    print(closed, prev_tag)
    #    exit()
    #print(ret_str)
    #exit()
    ret_str += "</p>"
    #ret_str = ret_str.replace("\f", "").replace("\t", "")
    return ret_str