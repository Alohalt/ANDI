import argparse
import csv
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import DATASET_PATH
from utils import set_seed


# 1. 数据集
class SimilarityDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        vector1 = torch.tensor(row[:100], dtype=torch.float32)
        vector2 = torch.tensor(row[100:200], dtype=torch.float32)
        label = torch.tensor(int(row[200]), dtype=torch.long)
        return vector1, vector2, label


# 2. Siamese网络模型
class SiameseNetwork(nn.Module):
    def __init__(self, dropout_rate=0.):
        super(SiameseNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, input1, input2):
        combined_output = torch.cat((input1, input2), dim=1)
        combined_output = self.dropout(combined_output)
        predictions = self.network(combined_output)
        return predictions

def train(train_loader, test_loader, device):
    model = SiameseNetwork(dropout_rate=args.dropout).to(device)  # 将模型移到GPU

    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 创建一个SummaryWriter实例
    writer = SummaryWriter()

    # 4. 训练模型
    for epoch in tqdm.tqdm(range(args.epochs)):  # 训练轮数
        bar = tqdm.tqdm(train_loader)
        batch_idx = 0
        for vector1, vector2, labels in train_loader:
            vector1, vector2, labels = vector1.to(device), vector2.to(device), labels.to(device)  # 数据移到GPU
            
            optimizer.zero_grad()
            outputs = model(vector1, vector2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # bar.set_description(f'Epoch {epoch+1}, Loss: {loss.item()}')

            
            # 记录损失
            # writer.add_scalar('Loss/train', loss.item(), batch_idx)

            # 每100个批次，使用一个批次的测试集进行测试
            if batch_idx % 100 == 0:
                test_vector1, test_vector2, test_labels = next(iter(test_loader))
                test_vector1, test_vector2, test_labels = test_vector1.to(device), test_vector2.to(device), test_labels.to(device)
                test_outputs = model(test_vector1, test_vector2)
                test_loss = criterion(test_outputs, test_labels)
                writer.add_scalar('Loss/test', test_loss.item(), batch_idx)

                # 计算准确率
                _, predicted = torch.max(test_outputs.data, 1)
                total = test_labels.size(0)
                correct = (predicted == test_labels).sum().item()
                accuracy = correct / total
                writer.add_scalar('Accuracy/test', accuracy, batch_idx)
                bar.update(100)
                bar.set_description(f'Epoch {epoch+1}, Loss: {loss.item()}, valid Loss: {test_loss.item()}, Accuracy: {accuracy}')
            batch_idx += 1


        bar.close()

    # 保存模型
    # 仅保存模型的状态字典（推荐）
    torch.save(model.state_dict(), 'experiments/outputs/rel_cls2/siamese_network_state_dict.pth')

    # 5. 评估模型
    correct = 0
    total = 0
    with torch.no_grad():
        for vector1, vector2, labels in test_loader:
            vector1, vector2, labels = vector1.to(device), vector2.to(device), labels.to(device)  # 数据移到GPU
            outputs = model(vector1, vector2)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

    


def main(data_version):
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

    # 读取数据
    save_path = os.path.join(processed_data_root, 'rel-embs')
    train_df = pd.read_csv(os.path.join(save_path, 'train/train_sample.csv'))  
    test_df = pd.read_csv(os.path.join(save_path, 'valid/valid_sample.csv'))  

    train_dataset = SimilarityDataset(train_df)
    test_dataset = SimilarityDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=8)


    train(train_loader, test_loader, args.device)


if __name__ == '__main__':

    # 建立解析对象，在main里面接受命令行传入的参数，然后做训练
    parser = argparse.ArgumentParser()
    # 给实例增加属性
    parser.add_argument("--model_dir", default="./experiments/outputs", required=False, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default=f"{DATASET_PATH}/data", type=str, help="The input data dir")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for CRF layer."
    )

    parser.add_argument("--dropout", default=0.6, type=float, help="dropout rate")
    parser.add_argument("--train_batch_size", default=2048, type=int, help="train batch size")
    parser.add_argument("--eval_batch_size", default=1024, type=int, help="eval batch size")
    parser.add_argument("--epochs", default=10, type=int, help="train epochs")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    set_seed(args)

    data_version = "v3"
    main(data_version)