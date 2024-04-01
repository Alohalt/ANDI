import json
import os
import pickle
import re
import codecs
from tqdm import tqdm
import pinyin
import unicodedata

import random
import logging

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score

from transformers import BertConfig
from transformers import BertTokenizer

from model import ClsBERT
from config import DATASET_PATH


puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
            'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                    'journal', 'science', 'international', 'key', 'sciences', 'research',
                    'academy', 'state', 'center']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these']


def load_json(*paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pickle(*paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data, *paths, ensure_ascii=False):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=4)


def save_pickle(data, *paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)



# 模型字典映射
MODEL_CLASSES = {
    'bert': (BertConfig, ClsBERT, BertTokenizer),
    'cls_bert1': (BertConfig, ClsBERT, BertTokenizer),
    'cls_bert2': (BertConfig, ClsBERT, BertTokenizer),
}
# 模型路径字典映射
MODEL_PATH_MAP = {
    # 'bert': './bert_finetune_cls1/resources/bert_base_uncased',
    'bert': './resources/uncased_L-2_H-128_A-2',
    'cls_bert1': './experiments/outputs/cls_bert1',
    'cls_bert2': './experiments/outputs/cls_bert2',
}

# 获得意图标签
def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]

# 加载分词模型
def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(MODEL_PATH_MAP['bert'])   # need modify

# 日志初始化
def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

# 固定种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# 打分
def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)

    results.update(intent_result)

    return results

#
def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()

    # TP = sum([1 for i in range(len(preds)) if preds[i] == labels[i] and preds[i] != 0])
    # FP = sum([1 for i in range(len(preds)) if preds[i] != labels[i] and preds[i] != 0])
    # FN = sum([1 for i in range(len(preds)) if preds[i] != labels[i] and preds[i] == 0])
    # pre = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # f1 = 2 * pre * recall / (pre + recall)

    pre = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(
        labels, preds
    )

    return {
        "acc": acc,
        "precision": pre,
        "recall": recall,
        "f1_score": f1,
    }


# 获得预测文本
def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]




def read_pubs(raw_data_root, dataset, mode):
    if dataset in ['Aminer-v3', 'Aminer-v2', 'Aminer-v1', 'CiteSeerX', 'dblp']:
        if mode == 'train':
            pubs = load_json(os.path.join(raw_data_root, "train_pub.json"))
        elif mode == 'valid' or mode == 'valid' or mode == 'valid_test':
            pubs = load_json(os.path.join(raw_data_root, "valid_pub.json"))
        elif mode == 'test':
            pubs = load_json(os.path.join(raw_data_root, "test_pub.json"))
        else:
            raise ValueError('choose right mode')
    elif dataset == 'Aminer-na':
        pubs = load_json(os.path.join(raw_data_root, 'train_pub.json'))
    else:
        raise Exception("data_version must be v1, v2, v3, na, x or dblp")
    return pubs


def read_raw_pubs(raw_data_root, dataset, mode):
    if dataset in ['Aminer-v3', 'Aminer-v2', 'Aminer-v1', 'CiteSeerX', 'dblp']:
        if mode == 'train':
            raw_pubs = load_json(os.path.join(raw_data_root, "train_author.json"))
        elif mode == 'valid' or mode == 'valid' or mode == 'valid_test':
            raw_pubs = load_json(os.path.join(raw_data_root, "valid_author.json"))
        elif mode == 'test':
            raw_pubs = load_json(os.path.join(raw_data_root, "test_author.json"))
        else:
            raise ValueError('choose right mode')
    elif dataset == 'Aminer-na':
        if mode == 'train':
            raw_pubs = load_json(os.path.join(raw_data_root, "trian_author.json"))
        elif mode == 'valid' or mode == 'valid' or mode == 'valid_test':
            raw_pubs = load_json(os.path.join(raw_data_root, "valid_author.json"))
    else:
        raise Exception("data_version must be v1, v2, v3 or na")
    return raw_pubs


def pairwise_evaluate(correct_labels, pred_labels):
    """Pairwise evaluation.

    Args:
        correct_labels: ground-truth labels (Numpy Array).
        pred_labels: predicted labels (Numpy Array).

    Returns:
        pairwise_precision (Float).
        pairwise_recall (Float).
        pairwise_f1 (Float).

    """

    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)

    return pairwise_precision, pairwise_recall, pairwise_f1


def unify_name_order(name):
    """
    unifying different orders of name.
    Args:
        name
    Returns:
        name and reversed name
    """
    token = name.split("_")
    name = token[0] + token[1]
    name_reverse = token[1] + token[0]
    if len(token) > 2:
        name = token[0] + token[1] + token[2]
        name_reverse = token[2] + token[0] + token[1]

    return name, name_reverse


names_wrong = [
    # find in train
    (['takahiro', 'toshiyuki', 'takeshi', 'toshiyuki', 'tomohiro', 'takamitsu', 'takahisa', 'takashi',
     'takahiko', 'takayuki'], 'ta(d|k)ashi'),
    (['akimasa', 'akio', 'akito'], 'akira'),
    (['kentarok'], 'kentaro'),
    (['xiaohuatony', 'tonyxiaohua'], 'xiaohua'),
    (['ulrich'], 'ulrike'),
    # find in valid
    (['naoto', 'naomi'], 'naoki'),
    (['junko'], 'junichi'),
    # find in test
    (['isaku'], 'isao')
]


def is_contains_chinese(strs):
    """
    Check if contains chinese characters.
    """
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def match_name(name, target_name):
    """
    Match different forms of names.
    """
    [first_name, last_name] = target_name.split('_')
    first_name = re.sub('-', '', first_name)
    # change Chinese name to pinyin
    if is_contains_chinese(name):
        name = re.sub('[^ \u4e00-\u9fa5]', '', name).strip()
        name = pinyin.get(name, format='strip')
        # remove blank space between characters
        name = re.sub(' ', '', name)
        target_name = last_name + first_name
        return name == target_name
    else:
        # unifying Pinyin characters with tones
        str_bytes = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore')
        name = str_bytes.decode('ascii')

        name = name.lower()
        name = re.sub('[^a-zA-Z]', ' ', name)
        tokens = name.split()

        if len(tokens) < 2:
            return False
        if len(tokens) == 3:
            # just ignore middle name
            if re.match(tokens[0], first_name) and re.match(tokens[-1], last_name):
                return True
            # ignore tail noise char
            if tokens[-1] == 'a' or tokens[-1] == 'c':
                tokens = tokens[:-1]

        if re.match(tokens[0], last_name):
            # unifying two-letter abbreviation of the Chinese name
            if len(tokens) == 2 and len(tokens[1]) == 2:
                if re.match(f'{tokens[1][0]}.*{tokens[1][1]}.*', first_name):
                    return True
            remain = '.*'.join(tokens[1:]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[1]) == 1 and len(tokens[2]) == 1:
                remain_reverse = f'{tokens[2]}.*{tokens[1]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        if re.match(tokens[-1], last_name):
            candidate = ''.join(tokens[:-1])
            find_remain = False
            for (wrong_list, right_one) in names_wrong:
                if candidate in wrong_list:
                    remain = right_one
                    find_remain = True
                    break
            if not find_remain:
                remain = '.*'.join(tokens[:-1]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[0]) == 1 and len(tokens[1]) == 1:
                remain_reverse = f'{tokens[1]}.*{tokens[0]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        return False


def dump_names_pub(raw_data_root, processed_data_root, dataset, mode):
    pubs = read_pubs(raw_data_root, dataset, mode)
    raw_pubs = read_raw_pubs(raw_data_root, dataset, mode)
    print('dumping names_pub...')
    for n, name in enumerate(tqdm(raw_pubs)):
        if mode in ['train', 'valid'] or dataset in ['Aminer-v2', 'Aminer-v1', 'CiteSeerX', 'dblp']:
            pid_list = []
            ilabel = 0
            labels = []
            for aid in raw_pubs[name]:
                pid_list.extend(raw_pubs[name][aid])
                labels.extend([ilabel] * len(raw_pubs[name][aid]))
                ilabel += 1
        elif mode == 'test':
            # test
            pid_list = raw_pubs[name]
        else:
            print("Invalid type!")
        name_pubs = {}
        for pid in pid_list:
            if pid in pubs:
                name_pubs[pid] = pubs[pid]
        os.makedirs(os.path.join(processed_data_root, 'names_pub', mode), exist_ok=True)
        save_json(name_pubs, os.path.join(processed_data_root, 'names_pub', mode, f'{name}.json'))


def dump_features_relations_to_file(data_version):
    """
    Generate paper features and relations by raw publication data and dump to files.
    Paper features consist of title, org, keywords. Paper relations consist of author_name, org, venue.
    """

    if data_version == "v3":
        dataset = 'Aminer-v3'
    elif data_version == "v2":
        dataset = 'Aminer-v2'
    elif data_version == "v1":
        dataset = 'Aminer-v1'
    elif data_version == "na":
        dataset = 'Aminer-na'
    elif data_version == "x":
        dataset = 'CiteSeerX'
    elif data_version == "dblp":
        dataset = 'dblp'
    else:
        raise Exception("data_version must be v1, v2, v3, na, x")
    
    raw_data_root = f"{DATASET_PATH}/{dataset}"
    processed_data_root = f"{DATASET_PATH}/data/gene/{dataset}"

    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

    for mode in ['train', 'valid', 'test']:  # 'train', 'valid',

        dump_names_pub(raw_data_root, processed_data_root, dataset, mode)

        print(f'processing {dataset}--{mode} relation data...')
        raw_pubs = read_raw_pubs(raw_data_root, dataset, mode)
        for n, name in enumerate(tqdm(raw_pubs)):

            file_path = os.path.join(processed_data_root, mode, name)
            os.makedirs(file_path, exist_ok=True)
            coa_file = open(os.path.join(file_path, 'paper_author.txt'), 'w', encoding='utf-8')
            cov_file = open(os.path.join(file_path, 'paper_venue.txt'), 'w', encoding='utf-8')
            cot_file = open(os.path.join(file_path, 'paper_title.txt'), 'w', encoding='utf-8')
            coo_file = open(os.path.join(file_path, 'paper_org.txt'), 'w', encoding='utf-8')

            authorname_dict = {}  # maintain a author-name-dict
            pubs_dict = load_json(os.path.join(processed_data_root, 'names_pub', mode, name + '.json'))

            ori_name = name
            name, name_reverse = unify_name_order(name)

            for i, pid in enumerate(pubs_dict):
                paper_features = []
                pub = pubs_dict[pid]

                # Save title (relations)
                title = pub["title"]
                pstr = title.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_check]
                for word in pstr:
                    cot_file.write(pid + '\t' + word + '\n')

                # Save keywords
                word_list = []
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        word_list.append(word)
                    pstr = " ".join(word_list)
                    pstr = re.sub(' +', ' ', pstr)
                keyword = pstr

                # Save org (relations)
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()
                    token = authorname.split(" ")

                    if len(token) == 2:

                        authorname = token[0] + token[1]
                        authorname_reverse = token[1] + token[0]
                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:

                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        coa_file.write(pid + '\t' + authorname + '\n')  # current name is a name of co-author
                    else:
                        if "org" in author:
                            org = author["org"]  # current name is a name for disambiguating
                            find_author = True

                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            org = author['org']
                            break

                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = set(pstr)
                for word in pstr:
                    coo_file.write(pid + '\t' + word + '\n')

                # Save venue (relations)
                if pub["venue"]:
                    pstr = pub["venue"].strip()
                    pstr = pstr.lower()
                    pstr = re.sub(puncs, ' ', pstr)
                    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                    pstr = pstr.split(' ')
                    pstr = [word for word in pstr if len(word) > 1]
                    pstr = [word for word in pstr if word not in stopwords]
                    pstr = [word for word in pstr if word not in stopwords_extend]
                    pstr = [word for word in pstr if word not in stopwords_check]
                    for word in pstr:
                        cov_file.write(pid + '\t' + word + '\n')
                    if len(pstr) == 0:
                        cov_file.write(pid + '\t' + 'null' + '\n')

            coa_file.close()
            cov_file.close()
            cot_file.close()
            coo_file.close()
        print(f'Finish {mode} data extracted.')
    print(f'All paper features extracted.')