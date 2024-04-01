import os
import torch
from os.path import join
import random
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from gensim.models import word2vec
from itertools import combinations

import tqdm
from rel_trainer import SiameseNetwork

class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, dirpath):
        temp = set()

        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_org:
                    self.paper_org[p] = []
                self.paper_org[p].append(a)
                if a not in self.org_paper:
                    self.org_paper[a] = []
                self.org_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_author:
                    self.paper_author[p] = []
                self.paper_author[p].append(a)
                if a not in self.author_paper:
                    self.author_paper[a] = []
                self.author_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_venue.txt", encoding='utf-8') as pcfile:
            for line in pcfile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_conf:
                    self.paper_conf[p] = []
                self.paper_conf[p].append(a)
                if a not in self.conf_paper:
                    self.conf_paper[a] = []
                self.conf_paper[a].append(p)
        temp.clear()

        print("#papers ", len(self.paper_conf))
        print("#authors", len(self.author_paper))
        print("#org_words", len(self.org_paper))
        print("#confs  ", len(self.conf_paper))

    def generate_WMRW(self, outfilename, numwalks, walklength, add_a, add_o, add_v):
        outfile = open(outfilename, 'w')
        for paper0 in self.paper_conf:
            for j in range(0, numwalks):  # wnum walks
                paper = paper0
                outline = ""
                i = 0
                while i < walklength:
                    i = i + 1
                    if add_a and paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len(authors)
                        authorid = random.randrange(numa)
                        author = authors[authorid]

                        papers = self.author_paper[author]
                        nump = len(papers)
                        # if nump == 1 --> self-loop
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                    if add_o and paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw)
                        word = words[wordid]

                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                    r_index = random.random()
                    if add_v and r_index >= 0.9:
                        if paper in self.paper_conf:
                            words = self.paper_conf[paper]
                            numw = len(words)
                            wordid = random.randrange(numw)
                            word = words[wordid]

                            papers = self.conf_paper[word]
                            nump = len(papers)
                            if nump > 1:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                                while paper1 == paper:
                                    paperid = random.randrange(nump)
                                    paper1 = papers[paperid]
                                paper = paper1
                                outline += " " + paper

                outfile.write(outline + "\n")
        outfile.close()
        # print("walks done")


class RelationalFeatures:
    def __init__(self, processed_data_root=None, repeat_num: int = 10, num_walk: int = 5, walk_len: int = 20,
                 rw_dim: int = 100, w2v_neg: int = 25, w2v_window: int = 10):

        self.processed_data_root = processed_data_root

        self.repeat_num = repeat_num
        self.num_walk = num_walk
        self.walk_len = walk_len
        self.rw_dim = rw_dim
        self.w2v_neg = w2v_neg
        self.w2v_window = w2v_window

        self.rel_cls_model = SiameseNetwork()
        self.rel_cls_model.load_state_dict(torch.load(join('experiments/outputs', 'rel_cls2', 'siamese_network_state_dict.pth')))
        self.rel_cls_model.eval()
        self.rel_cls_model.cuda()


    def cal_relational_similarity(self, pubs, name, mode, add_a, add_o, add_v):
        mpg = MetaPathGenerator()
        if mode == 'valid_test':
            mode = 'valid'
        mpg.read_data(join(self.processed_data_root, mode, name))

        cp = set()
        rw_path = join(self.processed_data_root, mode)
        os.makedirs(rw_path, exist_ok=True)
        rw_file = join(rw_path, 'RW.txt')
        mpg.generate_WMRW(rw_file, self.num_walk, self.walk_len, add_a, add_o, add_v)
        sentences = word2vec.Text8Corpus(rw_file)
        model = word2vec.Word2Vec(sentences, vector_size=self.rw_dim, negative=self.w2v_neg,
                                    min_count=1, window=self.w2v_window)
        embs = []
        for i, pid in enumerate(pubs):
            if pid in model.wv:
                embs.append(model.wv[pid])
            else:
                embs.append(np.zeros(100))
                cp.add(i)

        all_pair = list(combinations(range(len(embs)), 2))

        # 根据pair和all_embs构建输入数据
        inputs = []
        for i, j in all_pair:
            inputs.append(np.concatenate((embs[i], embs[j]), axis=0))
        inputs = np.array(inputs)

        # 输入到模型中
        inputs = torch.from_numpy(inputs).float().cuda()
        # 将输出转为dataloader
        dataloader = torch.utils.data.DataLoader(inputs, batch_size=1024, shuffle=False)

        preds = None
        for input in tqdm.tqdm(dataloader, desc="Evaluating"):
            # torch.no_grad():它包裹的不需要进行梯度计算
            with torch.no_grad():
                # 通过前向传播得到outputs
                output = self.rel_cls_model(input[:,:100], input[:,100:]).cpu().detach().numpy()
                
            # Intent prediction
            if preds is None:
                # detach()阻断反向传播，不再有梯度
                # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
                preds = output
            # 如果re_preds存在，则进行拼接
            else:
                preds = np.append(preds, output, axis=0)

        sim_matrix = None
        pair_sim = preds[:, 1]

        n = int(np.sqrt(2 * pair_sim.shape[0])) + 1
        upper_tri_matrix = np.zeros((n, n))
        raw, col = np.triu_indices(n, 1)
        upper_tri_matrix[raw, col] = pair_sim
        sim_matrix = upper_tri_matrix + upper_tri_matrix.T + np.eye(n)

        # nomalize emb_dis to [-1,1]
        sim_matrix = (sim_matrix - sim_matrix.min()) / (sim_matrix.max() - sim_matrix.min()) * 2 - 1
        # [0, 2]
        sim_matrix = 1 - sim_matrix

        return sim_matrix, cp