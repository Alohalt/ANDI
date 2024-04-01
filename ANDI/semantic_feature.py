from os.path import join
from sklearn.metrics.pairwise import pairwise_distances
from utils import load_json, load_pickle
from data_loader import load_and_cache_examples

class SemanticFeatures:
    def __init__(self, raw_data_root=None):
        self.raw_data_root = raw_data_root

    def cal_semantic_similarity(self, args, trainer, tokenizer, name):
        """Calculate semantic matrix of paper's by semantic feature.
        Args:
            Disambiguating name.
        Returns:
            Papers' similarity matrix (Numpy Array).
        """
        args.name = name
        if args.data_type == 'valid':
            test_dataset = load_and_cache_examples(args, tokenizer, mode='valid_test')
        else:
            test_dataset = load_and_cache_examples(args, tokenizer, mode=args.data_type)
        trainer.test_dataset = test_dataset
    
        _, emb_dis = trainer.evaluate("test")

        # emb_dis = emb_dis.cpu().numpy().reshape(-1, 1)  # (N, N)
        # nomalize emb_dis to [-1,1]
        emb_dis = (emb_dis - emb_dis.min()) / (emb_dis.max() - emb_dis.min()) * 2 - 1
        # [0, 2]
        emb_dis = 1 - emb_dis


        return emb_dis, set()