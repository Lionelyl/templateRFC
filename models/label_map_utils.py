import collections
import os
import json

def count_entity_freq(data_path):

    entity_freq = collections.defaultdict(dict)
    label_map = collections.defaultdict(dict)
    for file in data_path:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if len(line) < 2 or "-DOCSTART-" in line or "END-OF" in line:
                continue
            line = line.strip().split(" ======= ")
            word = line[0]
            if line[1].startswith('ref'):
                label = 'O'
            else:
                label = line[1].upper()

            # if label != "O":
            #     label = label[2:]
            entity_freq[word][label] = entity_freq[word].get(label, 0) + 1
            label_map[label][word] = label_map[label].get(word, 0) + 1


    return entity_freq, label_map

def get_label_map(protocol, k=6, filter_ratio=0.6):

    protocols = ["TCP", "SCTP", "PPTP", "LTP", "DCCP", "BGPv4"]
    output_dir =  'data/'
    raw_data_file = ['rfcs-bio/{}_phrases_train.txt'.format(p) for p in protocols if p != protocol]
    entity_freq, data_label_token_map = count_entity_freq(raw_data_file)

    label_map = {}
    label_list = []
    # get the top k word after sorted
    for label_name, _ in data_label_token_map.items():

        cnt = 0

        token_frac_dict = data_label_token_map[label_name]
        sort_key = lambda x: x[1]

        for token, frac in sorted(token_frac_dict.items(), key = sort_key, reverse=True):

            if label_name not in label_map:
                label_map[label_name] = {}
            if len(token)>1 and token in entity_freq and entity_freq[token]: #and "##" not in token
                entity_label_ratio = entity_freq[token].get(label_name, 0) / sum(entity_freq[token].values())
                if entity_label_ratio > filter_ratio:

                    label_map[label_name][token] = (frac, entity_freq[token])
                    cnt+=1
            if cnt>=k:
                break
    label_map_output = {"B-"+label:list(tokens.keys()) for label, tokens in label_map.items()}

    multitoken_term = "_multitoken"
    file_name = f"{protocol}_label_map_data_ratio{filter_ratio}{multitoken_term}_top{k}.json"

    output_file = os.path.join(output_dir, file_name)
    print("Dumping label_map to ", output_file)
    with open(output_file, "w") as f:
        json.dump(label_map_output, f)
    return label_map_output
