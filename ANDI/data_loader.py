import csv
import os
import copy
import json
import logging
import random
import re

import torch
from torch.utils.data import TensorDataset
from tqdm.contrib import tenumerate
from itertools import combinations

from config import DATASET_PATH
from utils import get_intent_labels, puncs, load_pickle
# 日志对象初始化
logger = logging.getLogger(__name__)


class InputExample(object):
    """
    定义InputExample类数据
    """
    def __init__(self, guid=None, text_a=None, text_b=None, intent_label=None, ):
        # 每个样本的独特序号
        self.guid = guid
        # 句子1
        self.text_a = text_a
        # 句子2
        self.text_b = text_b
        # 意图标签
        self.intent_label = intent_label

    def __repr__(self):
        """
        这里重写我们的输入信息
        """
        return str(self.to_json_string())

    def to_dict(self):
        """
         将此实例序列化到Python字典中
        __dict__:类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的
        对象实例的__dict__中存储了一些self.xxx的一些东西
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """
        类的属性等信息(字典格式)dumps进入json string
        json.dumps()函数将python对象编码成JSON字符串
        indent=2.文件格式中加入了换行与缩进
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    定义输入到BERT模型的InputFeatures类数据
    """
    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id):
        # 输入样本序列在bert词表里的索引，可以直接喂给nn.embedding
        self.input_ids = input_ids
        # 注意力mask，padding的部分为0，其他为1
        self.attention_mask = attention_mask
        # 表示每个token属于句子1还是句子2
        self.token_type_ids = token_type_ids
        # 意图标签id
        self.intent_label_id = intent_label_id

    def __repr__(self):
        """
        这里重写我们的输入信息
        """
        return str(self.to_json_string())

    def to_dict(self):
        """
         将此实例序列化到Python字典中
        __dict__:类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的
        对象实例的__dict__中存储了一些self.xxx的一些东西
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """
        类的属性等信息(字典格式)dump进入json string
        json.dumps()函数将python对象编码成JSON字符串
        indent=2.文件格式中加入了换行与缩进
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

class Processor(object):
    """
    句子分类任务的数据处理器
    """

    def __init__(self, args):
        # 参数
        self.args = args
        # 获得意图标签索引
        self.intent_labels = get_intent_labels(args)
        # 输入文本文件
        self.input_file = f'{"" if args.name is None else args.name + "_"}{args.task}.txt'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """
        逐行读取输入文件
        :param input_file: 输入文件路径
        :param quotechar:
        :return: 句子列表
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """
        为训练集与验证集构建example
        :param lines: 句子列表
        :param set_type: 区分训练集与验证集
        :return: 处理后的InputExample类数据
        """
        examples = []
        for i, line in enumerate(lines):
            # 每个样本的独特序号
            guid = "%s-%s" % (set_type, i)
            # 去掉首尾空格
            line = line.strip()
            if not line:
                continue
            # 以“\t”进行分割
            line = line.split("\t")
            if len(line) == 2:
                # 获得每行文本的句子1与句子2
                text_a = line[1].strip()  # Some are spaced twice
                text_b = ''  # Some are spaced twice
            elif len(line) == 1:
                text_a = ''
                text_b = ''
            else:
                # 获得每行文本的句子1与句子2
                text_a = line[1].strip()
                text_b = line[2].strip()
            # 获得意图标签
            intent = line[0]
            # print(intent, type(intent), self.intent_labels)
            # 获得意图标签对应的索引
            intent_label = self.intent_labels.index(intent)
            # 构造InputExample类数据
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    intent_label=intent_label
                )
            )
        return examples

    def get_examples(self, mode):
        """
        获得样例数据
        Args:
            mode: train, valid, test
        """
        # 拼接数据路径
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        if mode == "valid_test":
            data_path = os.path.join(self.args.data_dir, self.args.task, "valid", "AND")
        if mode == "test":
            # data_path = os.path.join(self.args.data_dir, self.args.task, "valid", "AND")
            data_path = os.path.join(self.args.data_dir, self.args.task, "test", "AND")
        # 写入日志
        logger.info("LOOKING AT {}".format(data_path))
        # 构建InputExample类样例数据
        return self._create_examples(lines=self._read_file(os.path.join(data_path, self.input_file)),
                                     set_type=mode)


# 如果有多个数据集，则数据集的processor可以通过映射得到
processors = {
    "v3": Processor,
    "na": Processor,
    'all': Processor,
    'v1': Processor,
    'v2': Processor,
    'x': Processor,
    'dblp': Processor
}


def convert_examples_to_features(examples,
                                 max_seq_len,
                                 tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """
    将example数据转化为BERT模型需要的输入格式
    :param examples: 样例数据
    :param max_seq_len: 最大序列长度
    :param tokenizer: 分词模型
    :param cls_token_segment_id: 0
    :param pad_token_segment_id: 0
    :param sequence_a_segment_id:0
    :param sequence_b_segment_id:1
    :param mask_padding_with_zero:True
    :return:返回BERT模型的输入数据
    """
    # 基于当前分词模型的设置
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    # 循环遍历每一个样例数据
    for (ex_index, example) in enumerate(examples):
        # 每隔1000个数据，写入日志
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize whole sent
        tokens = []
        token_type_ids = []
        # 将句子1的单词进行分词
        text_a_tokens = tokenizer.tokenize(example.text_a)
        text_b_tokens = None
        # 如果存在句子2，将句子2的单词进行分词
        if example.text_b:
            text_b_tokens = tokenizer.tokenize(example.text_b)
        # 如果已经将句子2的单词进行分词
        if text_b_tokens:
            # Account for [CLS] and [SEP]
            special_tokens_count = 3
            # 如果句子1太长，则对句子1进行截断
            if len(text_a_tokens) > (max_seq_len - special_tokens_count) // 2:
                text_a_tokens = text_a_tokens[: (max_seq_len - special_tokens_count) // 2]
            # 如果句子2太长，则对句子2进行截断
            if len(text_b_tokens) > (max_seq_len - special_tokens_count) // 2:
                text_b_tokens = text_b_tokens[: (max_seq_len - special_tokens_count) // 2]
            # 用句子1的tokens来扩充tokens列表
            tokens.extend(text_a_tokens)
            token_type_ids += [sequence_a_segment_id] * len(text_a_tokens)
            # 增加sep_token
            tokens.append(sep_token)
            token_type_ids.append(sequence_a_segment_id)
            # 用句子2的tokens来扩充tokens列表
            tokens.extend(text_b_tokens)
            token_type_ids += [sequence_b_segment_id] * len(text_b_tokens)

        else:
        # 如果只有句子1,不存在句子2
            # [CLS]和[SEP]
            special_tokens_count = 2
            # 如果句子1过长，需要对句子进行截断
            if len(text_a_tokens) > int(max_seq_len - special_tokens_count) // 2:
                text_a_tokens = text_a_tokens[: int(max_seq_len - special_tokens_count) // 2]
            # 用句子1的tokens来扩充tokens列表
            tokens.extend(text_a_tokens)
            token_type_ids += [sequence_a_segment_id] * len(text_a_tokens)

        # 增加 [SEP] token
        tokens += [sep_token]
        # 如果有句子2
        if text_b_tokens:
            token_type_ids.append(sequence_b_segment_id)
        # 如果只有句子1，没有句子2
        else:
            token_type_ids.append(sequence_a_segment_id)

        # 增加 [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # 将tokens转化为bert词表中对应的id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 注意力mask，句子中存在的部分为1
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # 需要填充的序列长度
        padding_length = max_seq_len - len(input_ids)
        # 输入样本序列在bert词表里的索引
        input_ids = input_ids + ([pad_token_id] * padding_length)
        # 注意力mask，padding的部分为0，其他为1
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        # token_type_ids表示每个token属于句子1还是句子2
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # 验证长度是否填充至最长序列
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        # 意图标签取整
        intent_label_id = int(example.intent_label)
        # 如果是前5个数据，则记录日志
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
        # 构造InputFeatures类BERT模型输入数据
        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    """
    将数据转化为cache文件，方便下一次快速加载
    :param args: 参数
    :param tokenizer: 分词
    :param mode: 区分训练、验证、测试
    """
    # 加载数据处理器
    processor = processors[args.task](args)

    # 拼接cach文件路径
    cached_features_file = os.path.join(
        args.data_dir, 'cached',
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )
    if mode == "test" or mode == "valid_test":
        cached_features_file = os.path.join(
            args.data_dir, 'cached',
            'cached_{}_{}_{}_{}'.format(
                args.name,
                args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len
            )
        )
    # logger.info("Loading features from cached file %s", cached_features_file)

    # 如果路径存在，则加载cach文件
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    # 如果路径不存在
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # 区分数据集
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "valid":
            examples = processor.get_examples("valid")
        elif mode == "test":
            examples = processor.get_examples("test")
        elif mode == "valid_test":
            examples = processor.get_examples("valid_test")
        else:
            raise Exception("For mode, Only train, valid, test is available")

        # 在计算交叉熵损失时忽略的索引：-100
        pad_token_label_id = args.ignore_index
        # 将example数据转化为features数据
        features = convert_examples_to_features(examples,
                                                args.max_seq_len,
                                                tokenizer,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        # 将数据保存至cach路径中
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features],
        dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features],
        dtype=torch.long
    )
    all_intent_label_ids = torch.tensor(
        [f.intent_label_id for f in features],
        dtype=torch.long
    )
    # 构造dataset
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids)
    # 返回dataset
    return dataset


def load_json(dir_path, file_name):
    path = os.path.join(dir_path, file_name)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_aminer_data(data_version, data_type='train'):
    if data_version == "v3":
        dir_path = f"{DATASET_PATH}/Aminer-v3"
    elif data_version == "v2":
        dir_path = f"{DATASET_PATH}/Aminer-v2"
    elif data_version == "v1":
        dir_path = f"{DATASET_PATH}/Aminer-v1"
    elif data_version == "na":
        dir_path = f"{DATASET_PATH}/Aminer-na"
    elif data_version == "x":
        dir_path = f"{DATASET_PATH}/CiteSeerX"
    elif data_version == "dblp":
        dir_path = f"{DATASET_PATH}/dblp"
    else:
        raise Exception("data_version must be v1, v2, v3, na, x")
    train_paper_file = "train_pub.json"
    train_author_file = "train_author.json"
    valid_paper_file = "valid_pub.json"
    valid_author_file = "valid_author.json"
    test_paper_file = "test_pub.json"
    test_author_file = "test_author.json"

    labels = []
    pids = []
    
    if data_type == 'train':
        authors = load_json(dir_path, train_author_file)
        paper = load_json(dir_path, train_paper_file)
    elif data_type == 'valid':
        authors = load_json(dir_path, valid_author_file)
        paper = load_json(dir_path, valid_paper_file)
    elif data_type == 'test':
        authors = load_json(dir_path, test_author_file)
        paper = load_json(dir_path, test_paper_file)
    author_label = 0
    names = []
    for name in authors:
        names.append(name)
        author_pids = authors[name]
        pid_list = []
        if data_type == 'test' and data_version == 'v3':
            pid_list = author_pids
            labels.extend([author_label] * len(pid_list))
        else:
            for author in author_pids:
                author_pid_list = author_pids[author]
                pid_list.extend(author_pid_list)
                labels.extend([author_label] * len(author_pid_list))
                author_label += 1
        pids.append(pid_list)
    
    return paper, pids, labels, names


def get_paper_text(paper):
    sentence = ''
    if 'title' in paper:
        sentence += re.sub(r'\s{2,}', ' ', re.sub(puncs, ' ', paper['title'].strip().lower())) + ' '
    if 'keywords' in paper:
        s = ' '.join(paper['keywords'])
        sentence += re.sub(r'\s{2,}', ' ', re.sub(puncs, ' ', s.strip().lower())) + ' '
    if 'authors' in paper:
        for author in paper['authors']:
            if 'org' in author and author['org'] is not None:
                sentence += re.sub(r'\s{2,}', ' ', re.sub(puncs, ' ', author['org'].strip().lower())) + ' '
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.replace('\r', ' ')
    sentence = sentence.replace('\t', ' ')

    return sentence


def paper_to_sentence(data_version, data_type):
    """paper data --> sentence data"""
    # load paper data

    papers_raw, pids, labels, _ = load_aminer_data(data_version, data_type)
    pids = sum(pids, [])
    print("read paper data done!")
    sentence_path = f'{DATASET_PATH}/data/{data_version}/{data_type}/{data_type}_sentences.txt'
    os.makedirs(os.path.dirname(sentence_path), exist_ok=True)
    with open(sentence_path, 'w', encoding='utf-8') as f:
        print("save sentence to file!")
        print("----------------------------")
        for i, pid in tenumerate(pids):
            if data_version == 'na':
                pid = pid.split('-')[0]
            paper = papers_raw[pid]
            sentence = get_paper_text(paper)
            if len(sentence) == 0:
                print(f"{data_version}-{data_type}-{pid} has no sentence!")
            f.write(str(labels[i]) + '\t' + sentence + '\n')
    print("save sentence to file done!")


def sample_sentence(data_version, data_type):
    if data_version == "v3":
        dir_path = f"{DATASET_PATH}/data/v3"
    elif data_version == "v2":
        dir_path = f"{DATASET_PATH}/data/v2"
    elif data_version == "v1":
        dir_path = f"{DATASET_PATH}/data/v1"
    elif data_version == "na":
        dir_path = f"{DATASET_PATH}/data/na"
    elif data_version == "x":
        dir_path = f"{DATASET_PATH}/data/x"
    elif data_version == "dblp":
        dir_path = f"{DATASET_PATH}/data/dblp"
    else:
        raise Exception("data_version must be v1, v2, v3, na or x")
    labels = []
    labels_num = 0
    sentences = []
    with open(f'{dir_path}/{data_type}/{data_type}_sentences.txt', 'r', encoding='utf-8') as f:
        pre = -1
        for i, line in enumerate(f.readlines()):
            if pre != int(line.split('\t')[0].strip()):
                pre = int(line.split('\t')[0].strip())
                labels.append([])
                labels_num += 1
            labels[labels_num-1].append(int(line.split('\t')[0].strip()))
            sentences.append(line.split('\t')[1].strip())
    print("read sentence data done!")
    print("----------------------------")
    # sample 100w pair index from range(0, len(labels)) not repeat, each sample is a (index_i, index_j) pair

    pair_index = []
    pre_len = 0
    all_len = len(sentences)
    for _, i in tenumerate(labels):
        all_pair = list(combinations(range(pre_len, pre_len+len(i)), 2))
        random.shuffle(all_pair)
        pos_num = len(i) * 5 if data_type == 'train' else len(i) * 2
        if len(all_pair) < pos_num:
            pos_pair = all_pair
        else:
            pos_pair = all_pair[:pos_num]
        pair_index.extend(pos_pair)
        
        neg_num = 15 if data_type == 'train' else 6
        neg_pair = []
        neg_index = [j for j in range(0, all_len) if j not in range(pre_len, pre_len+len(i))]
        neg_pair = [(k, j) for k in range(pre_len, pre_len+len(i)) for j in random.sample(neg_index, neg_num)]
        pair_index.extend(neg_pair)
        pre_len += len(i)

    labels = sum(labels, [])
    with open(f'{dir_path}/{data_type}/{data_version}.txt', 'w', encoding='utf-8') as f:
        print("save pair to file!")
        print("----------------------------")
        for _, (i, j) in tenumerate(pair_index):
            if labels[i] == labels[j]:
                f.write('1' + '\t' + sentences[i] + '\t' + sentences[j] + '\n')
            else:
                f.write('0' + '\t' + sentences[i] + '\t' + sentences[j] + '\n')
    
    print("sample done!")
        

def gen_test_sample(data_version, data_type):
    if data_version == "v3":
        dir_path = f"{DATASET_PATH}/data/v3"
    elif data_version == "v2":
        dir_path = f"{DATASET_PATH}/data/v2"
    elif data_version == "v1":
        dir_path = f"{DATASET_PATH}/data/v1"
    elif data_version == "na":
        dir_path = f"{DATASET_PATH}/data/na"
    elif data_version == "x":
        dir_path = f"{DATASET_PATH}/data/x"
    elif data_version == "dblp":
        dir_path = f"{DATASET_PATH}/data/dblp"
    else:
        raise Exception("data_version must be v1, v2, v3 or na")
    _, pids, _, names = load_aminer_data(data_version, data_type)
    labels = []
    sentences = []
    with open(f'{dir_path}/{data_type}/{data_type}_sentences.txt', 'r', encoding='utf-8') as f:

        for i, line in enumerate(f.readlines()):
            labels.append(int(line.split('\t')[0].strip()))
            if len(line.split('\t')) == 2:
                sentences.append(line.split('\t')[1].strip())
            else:
                sentences.append('')
    os.makedirs(f'{dir_path}/{data_type}/AND/', exist_ok=True)
    pre_len = 0
    for i, pid_list in tenumerate(pids):
        # if names[i] != "ling_huang":
        #     continue
        pid_num = len(pid_list)
        pair_index = list(combinations(range(pre_len, pre_len+pid_num), 2))
        
        with open(f'{dir_path}/{data_type}/AND/{names[i]}_{data_version}.txt', 'w', encoding='utf-8') as f:
            print("save pair to file!")
            print("----------------------------")
            for _, (i, j) in tenumerate(pair_index):
                if labels[i] == labels[j]:
                    f.write('1' + '\t' + sentences[i] + '\t' + sentences[j] + '\n')
                else:
                    f.write('0' + '\t' + sentences[i] + '\t' + sentences[j] + '\n')
        pre_len += pid_num
    print("sample done!")
    

def sample_relation(data_version, data_type):
    if data_version == "v3":
        dir_path = f"{DATASET_PATH}/Aminer-v3"
    elif data_version == "v2":
        dir_path = f"{DATASET_PATH}/Aminer-v2"
    elif data_version == "v1":
        dir_path = f"{DATASET_PATH}/Aminer-v1"
    elif data_version == "na":
        dir_path = f"{DATASET_PATH}/Aminer-na"
    elif data_version == "x":
        dir_path = f"{DATASET_PATH}//CiteSeerX"
    elif data_version == "dblp":
        dir_path = f"{DATASET_PATH}/dblp"
    else:
        raise Exception("data_version must be v1, v2, v3, na or x")

    processed_data_root = f"{dir_path}/processed_data"

    # save
    save_path = os.path.join(processed_data_root, 'rel-embs', data_type)
    save_path = os.path.join(save_path, 'relation_emb.pkl')
    all_embs = load_pickle(save_path)

    labels = []
    labels_num = 0
    embs = []

    pre = -1
    for label, emb in all_embs:
        if pre != label:
            pre = label
            labels.append([])
            labels_num += 1
        labels[labels_num - 1].append(label)
        embs.append(emb)
    print("read relation emb data done!")
    print("----------------------------")
    # sample 100w pair index from range(0, len(labels)) not repeat, each sample is a (index_i, index_j) pair

    pair_index = []
    pre_len = 0
    all_len = len(embs)
    for _, i in tenumerate(labels):
        all_pair = list(combinations(range(pre_len, pre_len + len(i)), 2))
        random.shuffle(all_pair)
        pos_num = len(i) * 5 if data_type == 'train' else len(i) * 2
        if len(all_pair) < pos_num:
            pos_pair = all_pair
        else:
            pos_pair = all_pair[:pos_num]
        pair_index.extend(pos_pair)

        neg_num = 15 if data_type == 'train' else 6
        neg_pair = []
        neg_index = [j for j in range(0, all_len) if j not in range(pre_len, pre_len + len(i))]
        neg_pair = [(k, j) for k in range(pre_len, pre_len + len(i)) for j in random.sample(neg_index, neg_num)]
        pair_index.extend(neg_pair)
        pre_len += len(i)

    labels = sum(labels, [])
    filename = os.path.join(processed_data_root, 'rel-embs', data_type, f'{data_type}_sample.csv')
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['V1_' + str(i) for i in range(1, 100 + 1)] +
                        ['V2_' + str(i) for i in range(1, 100 + 1)] + ['Label'])
        for _, (i, j) in tenumerate(pair_index):
            if labels[i] == labels[j]:
                data = embs[i].tolist() + embs[j].tolist() + [1]
                writer.writerow(data)
            else:
                data = embs[i].tolist() + embs[j].tolist() + [0]
                writer.writerow(data)

    print("sample done!")


if __name__ == '__main__':

    # 1、处理bert所需训练数据
    dataset = ['v3', 'na', 'x']  #
    data_type = ['train'] # , 'test' , 'valid'
    for d in dataset:
        for t in data_type:
            paper_to_sentence(d, t)
            if t in ['train', 'valid']:
                sample_sentence(d, t)
            if t in ['valid', 'test']:
                gen_test_sample(d, t)
            print(f"=============={d}-{t} done!===========")
    
    # 2、生成随机游走所需数据
    from utils import dump_features_relations_to_file
    for d in dataset:
        dump_features_relations_to_file(d)


    sample_relation("v3", "train")
    sample_relation("v3", "valid")