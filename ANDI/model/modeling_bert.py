import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
# from torchcrf import CRF
from .module import Classifier

"""
定义ClsBERT模型
"""
class ClsBERT(BertPreTrainedModel):
    def __init__(self, config, args, num_labels):
        super(ClsBERT, self).__init__(config)
        # 参数
        self.args = args
        # 意图标签数目
        self.num_labels = num_labels
        # 加载BERT预训练模型
        self.bert = BertModel(config=config)  # Load pretrained bert
        # 加载意图标签分类层
        self.classifier = Classifier(input_dim=config.hidden_size,
                                     num_intent_labels=self.num_labels,
                                     dropout_rate=args.dropout_rate)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        """
        定义前向传播
        :param input_ids: 输入样本序列在bert词表里的索引
        :param attention_mask: 注意力mask，padding的部分为0，其他为1
        :param token_type_ids: token_type_ids表示每个token属于句子1还是句子2
        :param intent_label_ids: 意图标签id
        """
        # BERT模型输出:sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # 序列的向量表征
        sequence_output = outputs[0]
        # 池化的向量表征
        pooled_output = outputs[1]  # [CLS]
        # 经过分类层得到的意图标签
        intent_logits = self.classifier(pooled_output)
        # 输出
        outputs = ((intent_logits),) + outputs[2:]  # add hidden states and attention if they are here

        # 如果意图标签id存在
        if intent_label_ids is not None:
            # 如果意图标签的数目为1
            if self.num_labels == 1:
                # MSE损失函数
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                # 交叉熵损失函数
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_labels), intent_label_ids.view(-1))
            # 输出
            outputs = (intent_loss,) + outputs
        # 返回(loss), logits, (hidden_states), (attentions)
        return outputs