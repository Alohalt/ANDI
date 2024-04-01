import codecs

import numpy as np
import time
import random
import argparse
import os

from os.path import join
from loguru import logger
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from config import DATASET_PATH
from utils import save_json, read_pubs, read_raw_pubs, pairwise_evaluate, set_seed, MODEL_CLASSES, MODEL_PATH_MAP, load_tokenizer
from semantic_feature import SemanticFeatures
from relation_feature import RelationalFeatures
from trainer import Trainer

def tanimoto(p, q):
    """Calculate the tanimoto coefficient.
    Args:
        two texts.
    Returns:
        A float number of coefficient.
    """
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))

def dump_result(pubs, pred):
    """Dump results file.

    Args:
        pubs: papers of this name (List).
        pred: predicted labels (Numpy Array).
    """
    result = []
    for i in set(pred):
        oneauthor = []
        for idx, j in enumerate(pred):
            if i == j:
                oneauthor.append(pubs[idx])
        result.append(oneauthor)
    return result # List[List[pid]]


class SNDTrainer:
    def __init__(self, w_author=1.5, w_org=1.0, w_venue=1.0, w_title=0.33,
                 text_weight=1.0, db_eps=0.2, db_min=4, num_walk=5, walk_len=20):
        self.dataset = args.dataset
        self.data_type = args.data_type  # train valid test

        # Modifying arguments when calling from outside
        self.processed_data_root = f'/home/liutao/workspace/AND/dataset/data/gene/{args.dataset}'
        self.raw_data_root = f'{DATASET_PATH}/{args.dataset}'
        self.w_author = w_author
        self.w_org = w_org
        self.w_venue = w_venue
        self.w_title = w_title
        self.text_weight = text_weight

        self.db_eps = db_eps
        self.db_min = db_min
        self.num_walk = num_walk
        self.walk_len = walk_len

        self.semantic_feature = SemanticFeatures(self.processed_data_root)
        self.relational_feature = RelationalFeatures(self.processed_data_root, num_walk=num_walk, walk_len=walk_len)

        self.model = DBSCAN(eps=db_eps, min_samples=db_min, metric='precomputed')

    def save_pair(self,pubs, name, outlier):
        """Save post-matching paper pair by threshold.
        """
        if self.data_type == 'valid_test':
            dirpath = join(f'{self.processed_data_root}', 'valid', name)
        else:
            dirpath = join(f'{self.processed_data_root}', self.data_type, name)

        paper_org = {}
        paper_conf = {}
        paper_author = {}
        paper_word = {}

        temp = set()
        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_org:
                    paper_org[p] = []
                paper_org[p].append(a)
        temp.clear()

        with open(dirpath + "/paper_venue.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_conf:
                    paper_conf[p] = []
                paper_conf[p] = a
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_author:
                    paper_author[p] = []
                paper_author[p].append(a)
        temp.clear()

        with open(dirpath + "/paper_title.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_word:
                    paper_word[p] = []
                paper_word[p].append(a)
        temp.clear()

        paper_paper = np.zeros((len(pubs), len(pubs)))
        for i, pid in enumerate(pubs):
            if i not in outlier:
                continue
            for j, pjd in enumerate(pubs):
                if j == i:
                    continue
                ca = 0
                cv = 0
                co = 0
                ct = 0

                if pid in paper_author and pjd in paper_author:
                    ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * self.w_author
                if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                    cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd])) * self.w_venue
                if pid in paper_org and pjd in paper_org:
                    co = tanimoto(set(paper_org[pid]), set(paper_org[pjd])) * self.w_org
                if pid in paper_word and pjd in paper_word:
                    ct = len(set(paper_word[pid]) & set(paper_word[pjd])) * self.w_title

                paper_paper[i][j] = ca + cv + co + ct

        return paper_paper

    def post_match(self, pred, tcp, cp, pubs, name):
        """Post-match outliers of clustering results.
        Using threshold based characters' matching.

        Args:
            clustering labels (Numpy Array).

        Returns:
            predicted labels (Numpy Array).

        """
        outlier = set()
        for i in range(len(pred)):
            if pred[i] == -1:
                outlier.add(i)
        for i in tcp:
            outlier.add(i)
        for i in cp:
            outlier.add(i)

        paper_pair = self.save_pair(pubs, name, outlier)
        paper_pair1 = paper_pair.copy()
        K = len(set(pred))
        for i in range(len(pred)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])

            while j in outlier:
                paper_pair[i][j] = -1

                last_j = j
                j = np.argmax(paper_pair[i])
                if j == last_j:
                    break

            if paper_pair[i][j] >= 1.5:
                pred[i] = pred[j]
            else:
                pred[i] = K
                K = K + 1

        for ii, i in enumerate(outlier):
            for jj, j in enumerate(outlier):
                if jj <= ii:
                    continue
                else:
                    if paper_pair1[i][j] >= 1.5:
                        pred[j] = pred[i]
        return pred

    def fit(self, add_sem=True, add_rel=True, if_post_match=True,
        add_a=True, add_o=True, add_v=True):


        raw_pubs = read_raw_pubs(self.raw_data_root, self.dataset, self.data_type)
        result = {}
        if self.data_type in ['train', 'valid_test']:
            wf = codecs.open(join(f'result/{self.dataset}/{self.data_type}_{"sem_" if add_sem else ""}{"rel_" if add_rel else ""}'\
                                  f'{"textweight"+str(self.text_weight) + "_" if add_sem and add_rel else ""}'\
                                  f'{"eps"+str(self.db_eps)+"_"}{"min"+str(self.db_min)+"_"}{"num_walk"+str(self.num_walk)+"_"}'\
                                  f'{"walk_len"+str(self.walk_len)+"_"}results.csv'), 'w', encoding='utf-8')
            wf.write('name,precision,recall,f1,n_pubs,n_clusters,n_pre_clusters\n')
        avg_pre, avg_recall, avg_f1 = 0., 0., 0.
        tokenizer = load_tokenizer(args)
        bert_trainer = Trainer(args, None, None, None)
        bert_trainer.load_model()
        for n, name in enumerate(tqdm(raw_pubs)):
            # if name != "ling_huang":
            #     continue
            if self.data_type in ['train', 'valid_test']:
                pubs = []
                ilabel = 0
                labels = []
                for aid in raw_pubs[name]:
                    if self.dataset == "Aminer-na":
                        pid_list = [pid.split('-')[0] for pid in raw_pubs[name][aid]]
                    else:
                        pid_list = raw_pubs[name][aid]
                    pubs.extend(pid_list)
                    labels.extend([ilabel] * len(raw_pubs[name][aid]))
                    ilabel += 1
            elif self.data_type == 'test':
                # test
                pubs = raw_pubs[name]
            else:
                print("Invalid type!")

            tcp = set()
            cp = set()
            # 逐name获取特征
            if add_sem and not add_rel:
                sem_dis, tcp = self.semantic_feature.cal_semantic_similarity(args, bert_trainer, tokenizer, name)
                dis = sem_dis
            elif not add_sem and add_rel:
                rel_dis, cp = self.relational_feature.cal_relational_similarity(pubs, name, self.data_type, add_a, add_o, add_v)
                dis = rel_dis
            elif add_sem and add_rel:
                sem_dis, tcp = self.semantic_feature.cal_semantic_similarity(args, bert_trainer, tokenizer, name)
                rel_dis, cp = self.relational_feature.cal_relational_similarity(pubs, name, self.data_type, add_a, add_o, add_v)
                dis = (np.array(rel_dis) + self.text_weight * np.array(sem_dis)) / (1 + self.text_weight)
            # 逐name cluster  结果为pred
            pred = self.model.fit_predict(dis)
            if if_post_match:
                pred = self.post_match(pred, tcp, cp, pubs, name)

            # 输出结果
            if self.data_type in ['train', 'valid_test']:
                pre, recall, f1 = pairwise_evaluate(labels, pred)
                avg_pre += pre / (len(raw_pubs) + 1)
                avg_recall += recall / (len(raw_pubs) + 1)
                avg_f1 += f1 / (len(raw_pubs) + 1)
                print(len(set(labels)), len(set(pred)))
                print(f'{name}:{pre},{recall},{f1}\n')
                wf.write(f'{name},{pre},{recall},{f1},{len(pubs)},{len(set(labels))},{len(set(pred))}\n')

            # 逐name保存聚类结果
            result[name] = []
            result[name].extend(dump_result(pubs, pred))

        save_dir = 'result/' + self.dataset + '/'
        os.makedirs(save_dir, exist_ok=True)
        save_json(result, save_dir,
                  f'{args.data_type}_{"sem_" if add_sem else ""}{"rel_" if add_rel else ""}result.{self.data_type}.json')
        if self.data_type in ['train', 'valid_test']:
            wf.write(f',{avg_pre},{avg_recall},{avg_f1},,,\n')
            wf.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # parser.add_argument('--dataset', type=str, default='Aminer-v3', help='dataset name')
    # parser.add_argument('--feature_type', type=str, default='bert', help='feature type')
    parser.add_argument('--data_type', type=str, default='train', help='data type')

        # 给实例增加属性
    parser.add_argument("--task", default=None, required=False, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./experiments/outputs", required=False, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default=f"{DATASET_PATH}/data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="entailment classification Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=512, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=300, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")


    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    # linear 层
    parser.add_argument(
        "--linear_learning_rate", default=5e-5, type=float, help="The initial learning rate for CRF layer."
    )
    # 使用focal_loss，如果样本类别不均衡
    parser.add_argument(
        "--use_focal_loss", action="store_true",
        help="Whether to use focal loss as the loss objective"
    )
    # focal_loss中的gamma参数
    parser.add_argument(
        "--focal_loss_gamma", default=2.0, type=float,
        help="gamma in focal loss"
    )
    # 类别权重
    parser.add_argument(
        "--class_weights", default=None, type=float,
        help="class_weights, written in string like '1.0,1.0,1.0,2.0,1.0' "
    )
    # 池化方法
    parser.add_argument("--mention_pooling", default="start", type=str,
                        help="mention pooling should be in type selected in the list: [start, avg, max]" )

    # 添加参数，用于控制对抗训练
    parser.add_argument("--at_method", default=None, type=str,
                        help="aversarial training should be in type selected in the list: [fgm, pgd, None]")
    parser.add_argument("--probs_for_at", default=0.5, type=float,
                        help="probs for adv training: only probs_for_at percent of times, use adv attacks")
    # 用于设置对抗训练的系数，FGM中用到的
    parser.add_argument("--epsilon_for_at", default=0.5, type=float,
                        help="epsilon coefficient for adv training: step size")
    # 用于设置对抗训练的系数，PGD中用到的,包括epsilon
    parser.add_argument("--alpha_for_at", default=0.5, type=float,
                        help="alpha coefficient for adv training: step size for PGD ")
    parser.add_argument("--steps_for_at", default=3, type=int,
                        help="num of steps at each adv sample: for PGD ")
    # 属性给与args实例:把parser中设置的所有"add_argument"给返回到args子类实例当中，
    # 那么parser中增加的属性内容都会在args实例中，使用即可。
    args = parser.parse_args()
    # 设置模型名字与路径
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args.model_type = "bert"
    args.task = "v3"
    args.model_dir = os.path.join(args.model_dir, "cls_bert9")
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    set_seed(args)

    args.train = False
    args.val = True
    args.test = False
    args.dataset = 'Aminer-v3'
    # args.feature_type = 'word2vec'


    processed_data_root = f'./gene/{args.dataset}'
    raw_data_root = f'{DATASET_PATH}/{args.dataset}'
    # save relation to file
    # dump_features_relations_to_file(raw_data_root, processed_data_root, args.dataset)

    main_start_time = time.time()
    now_time = time.strftime("%Y-%m-%d %H-%M", time.localtime(time.time()))
    logger.remove(handler_id=None)
    logger.add(sink=f'./log/{args.dataset}/{now_time}_walk.log')
    logger.info(f"start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    logger.info(f"args: {args}")

    if args.train:
        logger.info(f"start train: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        trainer = SNDTrainer()
        trainer.fit(add_rel=False)
        logger.info(f"end train: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    # logger.info(f"end name_dis: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    if args.val:
        logger.info(f"start valid: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        args.data_type = 'valid_test'
        trainer = SNDTrainer(text_weight=1, db_eps=0.17, db_min=4, num_walk=5, walk_len=15)
        trainer.fit(add_sem=True, add_rel=True)
        logger.info(f"end valid: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    if args.test:
        logger.info(f"start test: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
        args.data_type = 'test'
        trainer = SNDTrainer(text_weight=0.5, db_eps=0.17, db_min=4, num_walk=5, walk_len=15)
        trainer.fit(add_sem=True, add_rel=True)
        logger.info(f"end test: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    main_end_time = time.time()
    logger.info(f"end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    logger.info(f"total time: {main_end_time - main_start_time}s")