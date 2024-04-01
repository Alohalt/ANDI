import argparse
import csv
import os
from config import DATASET_PATH
# 设置绝对路径

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    # 初始化日志
    init_logger()
    # 设置随机数种子
    set_seed(args)
    # 加载分词模型
    tokenizer = load_tokenizer(args)

    # print(len(tokenizer.vocab))
    # print(type(tokenizer.vocab))

    # 加载训练、验证dataset
    args.name = None
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    valid_dataset = load_and_cache_examples(args, tokenizer, mode="valid")

    trainer = Trainer(args, train_dataset, valid_dataset, None)
    # 训练
    if args.do_train:
        trainer.train()
    # 评估
    if args.do_eval:
        # 加载模型
        trainer.load_model()
        # 在验证集上评估
        result, _ = trainer.evaluate("valid")
    
    # 测试
    if args.do_test:
        names = [name[:-7] for name in os.listdir(f'{DATASET_PATH}/data/v3/valid/AND') if name.endswith(".txt")]
        # open a csv file
        trainer.load_model()
        with open('./result_test.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'acc', 'precision', 'recall', 'f1_score'])
            for name in names:
                args.name = name
                test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
                trainer.test_dataset = test_dataset
            
                result, _ = trainer.evaluate("test")
                # write the result to csv file
                writer.writerow([name, result['acc'], result["precision"], result["recall"], result['f1_score']])



if __name__ == '__main__':
    # 建立解析对象，在main里面接受命令行传入的参数，然后做训练
    parser = argparse.ArgumentParser()
    # 给实例增加属性
    parser.add_argument("--task", default=None, required=False, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./experiments/outputs", required=False, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default=f"{DATASET_PATH}/data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="entailment classification Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=1024, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=1024, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=300, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", default=False, help="Whether to run eval on the valid set.")
    parser.add_argument("--do_test", action="store_true", default=False, help="Whether to run eval on the test set.")
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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args.model_type = "bert"
    args.task = "all"
    args.do_train = True
    args.do_eval = True
    args.at_method = "pgd"
    args.num_train_epochs = 4
    args.model_dir = os.path.join(args.model_dir, "cls_bert9")
    # args.do_test = True
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    # 调用主程序
    main(args)
