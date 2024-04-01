import logging

import torch
# 日志对象初始化
logger = logging.getLogger(__name__)


class FGM():
    """
    定义对抗训练方法FGM,对模型embedding参数进行扰动
    """
    def __init__(self, model, epsilon=0.25,):
        # BERT模型
        self.model = model
        # 求干扰时的系数值
        self.epsilon = epsilon

        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        """
        得到对抗样本
        :param emb_name:模型中embedding的参数名
        :return:
        """
        # 循环遍历模型所有参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 把真实参数保存起来
                self.backup[name] = param.data.clone()
                # 对参数的梯度求范数
                norm = torch.norm(param.grad)
                # 如果范数不等于0并且norm中没有缺失值
                if norm != 0 and not torch.isnan(norm):
                    # 计算扰动，param.grad / norm=单位向量，起到了sgn(param.grad)一样的作用
                    r_at = self.epsilon * param.grad / norm
                    # 在原参数的基础上添加扰动
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        将模型原本的参数复原
        :param emb_name:模型中embedding的参数名
        """
        # 循环遍历模型所有参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 断言
                assert name in self.backup
                # 取出模型真实参数
                param.data = self.backup[name]
        # 清空self.backup
        self.backup = {}


class PGD():
    """
    定义对抗训练方法PGD
    """
    def __init__(self, model, epsilon=1.0, alpha=0.3):
        # BERT模型
        self.model = model
        # 两个计算参数
        self.epsilon = epsilon
        self.alpha = alpha
        # 用于存储embedding参数
        self.emb_backup = {}
        # 用于存储梯度，与多步走相关
        self.grad_backup = {}

    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        """
        对抗
        :param emb_name: 模型中embedding的参数名
        :param is_first_attack: 是否是第一次攻击
        """
        # 循环遍历模型的每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                # 如果是第一次攻击
                if is_first_attack:
                    # 存储embedding参数
                    self.emb_backup[name] = param.data.clone()
                # 求梯度的范数
                norm = torch.norm(param.grad)
                # 如果范数不等于0
                if norm != 0:
                    # 计算扰动,param.grad / norm=单位向量相当于sgn符号函数
                    r_at = self.alpha * param.grad / norm
                    # 在原参数的基础上添加扰动
                    param.data.add_(r_at)
                    # 控制扰动后的模型参数值
                    # 投影到以原参数为原点，epsilon大小为半径的球上面
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self, emb_name='word_embeddings'):
        """
        将模型原本参数复原
        :param emb_name: 模型中embedding的参数名
        """
        # 循环遍历每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息，并且包含了模型中embedding的参数名
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                # 取出模型真实参数
                param.data = self.emb_backup[name]
        # 清空emb_backup
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        """
        控制扰动后的模型参数值
        :param param_name:
        :param param_data:
        :param epsilon:
        """
        # 计算加了扰动后的参数值与原始参数的差值
        r = param_data - self.emb_backup[param_name]
        # 如果差值的范数大于epsilon
        if torch.norm(r) > epsilon:
            # 对差值进行截断
            r = epsilon * r / torch.norm(r)
        # 返回新的加了扰动后的参数值
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        """
        对梯度进行备份
        """
        # 循环遍历每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息
            if param.requires_grad:
                # 如果参数没有梯度
                if param.grad is None:
                    print("{} param has no grad !!!".format(name))
                    continue
                # 将参数梯度进行备份
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        """
        将梯度进行复原
        """
        # 循环遍历每一个参数
        for name, param in self.model.named_parameters():
            # 如果当前参数在计算中保留了对应的梯度信息
            if param.requires_grad:
                # 如果没有备份
                if name not in self.grad_backup:
                    continue
                # 如果备份了，就将原始模型参数梯度取出
                param.grad = self.grad_backup[name]