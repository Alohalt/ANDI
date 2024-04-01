import os
import logging
import random

from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from adversarial_training import FGM, PGD
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels
# 日志对象初始化
logger = logging.getLogger(__name__)


class Trainer(object):
    """
     Trainer类定义关系分类任务的训练与评估
    """
    def __init__(self, args, train_dataset=None, valid_dataset=None, test_dataset=None):
        # 参数
        self.args = args
        # 加载数据集
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        # 获取意图标签id
        self.id2label = get_intent_labels(args)
        # 构建字典，键是label，值是索引
        self.label2id = {k_: v_ for v_, k_ in enumerate(self.id2label)}
        # 损失计算时，忽略label损失的索引
        self.pad_token_label_id = args.ignore_index
        # 加载预训练好的模型
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        # 导入配置文件
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        # BERT模型
        self.model = self.model_class.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            args=args,
            num_labels=len(self.label2id),
        )

        # 加载配置：GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

        # 对抗训练
        self.adv_trainer = None
        # 如果使用对抗训练方法
        if self.args.at_method:
            # 如果对抗训练方法为FGM
            if self.args.at_method == "fgm":
                # 实例化对抗训练方法FGM
                self.adv_trainer = FGM(
                    self.model,
                    epsilon=self.args.epsilon_for_at)
            # 如果对抗训练方法为PGD
            elif self.args.at_method == "pgd":
                # 实例化对抗训练方法PGD
                self.adv_trainer = PGD(
                    self.model,
                    epsilon=self.args.epsilon_for_at,
                    alpha=self.args.alpha_for_at,
                )
            else:
                raise ValueError(
                    "un-supported adversarial training method: {} !!!".format(self.args.at_method)
                )

    def train(self):
        """
        微调部分
        """
        # 实例化TensorboardX-writer
        writer = SummaryWriter()
        # 训练集通过DataLoader进行加载
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        # 计算需要多少步梯度更新，t_total用于learning rate的更新
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            # 计算训练epochs的数目
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            # 计算训练epochs的数目
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        for n, p in self.model.named_parameters():
            print(n)

        # Prepare optimizer and schedule (linear warmup and decay)

        optimizer_grouped_parameters = []

        # BERT部分参数，设置一个较低的学习率
        bert_params = list(self.model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        # 部分参数设置权重衰减，部分参数不设置
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.learning_rate,
            }
        ]

        # 线性层参数
        linear_params = list(self.model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        # 部分参数设置权重衰减，部分参数不设置
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in linear_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                "lr": self.args.linear_learning_rate,
            },
            {
                'params': [p for n, p in linear_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.linear_learning_rate,
            }
        ]
        # 优化器
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # 学习率的衰减
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        # 将梯度清空
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        # 循环遍历每一个epoch
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            # 循环遍历每一个batch的数据
            for step, batch in enumerate(epoch_iterator):
                # 模型训练
                self.model.train()
                # GPU or CPU
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                # 输入数据为输入样本序号，attention_mask,意图标签索引
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                # 通过模型的前向传播得到outputs
                outputs = self.model(**inputs)
                # 得到损失
                loss = outputs[0]
                # 如果梯度累积步骤大于1，则需要求平均损失
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                # 反向传播求梯度
                writer.add_scalar('Loss/train', loss, global_step)   # 调用可视化方法
                loss.backward()

                # 如果要进行对抗训练
                if self.args.at_method is not None:
                    # 以一定概率probs_for_at，随机进行对抗训练
                    if random.uniform(0, 1) > self.args.probs_for_at:
                        # logger.info("not to do adv training at this step!")
                        pass

                    else:
                        # logger.info(" do adv training at this step!")
                        # 如果对抗训练方法为FGM
                        if self.args.at_method == "fgm":
                            # 这个时候embedding参数被修改了
                            self.adv_trainer.attack()
                            # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消

                            # embedding参数被修改，此时，输入序列得到的embedding表征不一样
                            # 在对抗样本上求损失函数
                            outputs_at = self.model(**inputs)
                            loss_at = outputs_at[0]
                            # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                            loss_at.backward()

                            # 恢复Embedding的参数
                            self.adv_trainer.restore()
                        # 如果对抗训练方法为PGD
                        elif self.args.at_method == "pgd":
                            # 保存正常的grad
                            self.adv_trainer.backup_grad()
                            # PGD要走多步，迭代走多步
                            for t in range(self.args.steps_for_at):
                                # 在embedding上添加对抗扰动, first attack时备份param.data
                                self.adv_trainer.attack(is_first_attack=(t == 0))
                                # 中间过程，梯度清零
                                if t != self.args.steps_for_at - 1:
                                    optimizer.zero_grad()
                                # 最后一步，恢复正常的grad
                                else:
                                    self.adv_trainer.restore_grad()
                                # embedding参数被修改，此时，输入序列得到的embedding表征不一样
                                outputs_at = self.model(**inputs)
                                # 对抗样本上的损失
                                loss_at = outputs_at[0]
                                # 反向传播
                                loss_at.backward()
                            # 恢复embedding参数
                            self.adv_trainer.restore()

                tr_loss += loss.item()
                # 每隔一个梯度累积步数(gradient_accumulation_steps)做一次梯度更新
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # 防止训练过程中梯度爆炸，进行梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    # 梯度下降，更新参数
                    optimizer.step()
                    # learning_rate进行梯度更新
                    scheduler.step()  # Update learning rate schedule
                    # 将梯度清零
                    self.model.zero_grad()
                    global_step += 1
                    # 每隔logging_steps在验证集上进行评估
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        result, _ = self.evaluate("valid")
                        writer.add_scalar('f1/valid', result['f1_score'], global_step//self.args.logging_steps)
                    # 每隔save_steps进行保存模型
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()
                # 超过最大迭代步数，则停止
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            # 超过最大迭代步数，则停止
            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        # 返回迭代步数与损失
        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        """
        评估部分
        :param mode: 区分验证集与测试集
        :return: 返回评估results
        """
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'valid':
            dataset = self.valid_dataset
        else:
            raise Exception("Only valid and test dataset available")

        # 评估时，数据集不需要打乱
        # SequentialSampler:按顺序进行采样
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        # 收集所有样本的预测结果
        preds = None
        # 收集所有样本的意图标签
        out_label_ids = None
        # 模型不需要进行梯度更新，将模型固定
        self.model.eval()
        # 循环每一个batch
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            # torch.no_grad():它包裹的不需要进行梯度计算
            with torch.no_grad():
                # 输入数据为输入样本序号，attention_mask，意图标签索引
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                # 通过前向传播得到outputs
                outputs = self.model(**inputs)
                # 损失与预测结果
                tmp_eval_loss, logits = outputs[:2]
                # 平均损失
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if preds is None:
                # detach()阻断反向传播，不再有梯度
                # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            # 如果re_preds存在，则进行拼接
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

        # 评估平均损失
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        sim_matrix = None
        if mode == 'test':
            pair_sim = preds[:, 1]
            n = int(np.sqrt(2 * pair_sim.shape[0])) + 1
            upper_tri_matrix = np.zeros((n, n))
            raw, col = np.triu_indices(n, 1)
            upper_tri_matrix[raw, col] = pair_sim
            sim_matrix = upper_tri_matrix + upper_tri_matrix.T + np.eye(n)

        # 意图标签结果
        preds = np.argmax(preds, axis=1)
        # 评估打分
        total_result = compute_metrics(preds, out_label_ids)
        # 结果更新
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results, sim_matrix

    def save_model(self):
        # 如果路径不存在，则构造路径
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # 保存模型
        model_to_save.save_pretrained(self.args.model_dir)

        # 将训练参数与训练好的模型一起保存
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # 检查路径是否存在
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            # 加载预训练模型
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          config=self.config,
                                                          args=self.args,
                                                          num_labels=len(self.label2id),)
            # cpu or cuda
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
