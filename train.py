import csv
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
import warnings
import logging
warnings.filterwarnings('ignore')

logging.basicConfig(filename='training.log', level=logging.INFO)


"""
读取评论文件的评论信息
"""
def read_file(file_name):
    comments_data = None
    
    # 读取评论信息
    with open(file_name, 'r', encoding='UTF-8') as f:
        reader = csv.DictReader(f)
        # 读取评论数据和对应的标签信息
        comments_data = [[line['review'], int(line['label'])] for line in reader if len(line['review']) > 0]                      
    
    # 打乱数据集
    random.shuffle(comments_data)
    data = pd.DataFrame(comments_data)
    data = data.drop_duplicates()                                                     # 删除重复的样本信息
    f.close()
    
    return data


"""
定义超参数类
"""
class BERTConfig:
    def __init__(self, batch_size,learning_rate,epoches):
        self.output_dim = 2
        self.hidden_layer = 768
        self.pretrained_name = 'bert-base-chinese'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = nn.CrossEntropyLoss()  

"""
定义BERTClassifier分类器模型
"""
class BERTClassifier(nn.Module):

    # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
    def __init__(self, config):
        super(BERTClassifier, self).__init__()
        self.output_dim = config.output_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 定义 Bert 模型
        self.bert = BertModel.from_pretrained(config.pretrained_name)
        # 外接全连接层
        self.mlp = nn.Linear(config.hidden_layer, config.output_dim)


    def forward(self, batch_sectences):
        input_ids = torch.tensor(batch_sectences["input_ids"]).to(self.device)
        attention_mask = torch.tensor(batch_sectences["attention_mask"]).to(self.device)

        bert_output = self.bert(input_ids, attention_mask=attention_mask)

        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态

        linear_output = self.mlp(bert_cls_hidden_state)

        return linear_output


"""
评估函数，用以评估数据集在神经网络下的精确度
"""
def evaluate(clf_model, comments_data, labels_data, config):

    tokenizer = config.tokenizer
    device = config.device

    sum_correct, i = 0, 0

    while i < len(comments_data):
        comments = comments_data[i: min(i + config.batch_size, len(comments_data))]

        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device)

        res = clf_model(tokens_X).to(device)
        
        y = torch.tensor(labels_data[i: min(i + config.batch_size, len(comments_data))]).reshape(-1).to(device)

        sum_correct += (res.argmax(axis=1) == y).sum()
        i += config.batch_size

    return sum_correct/len(comments_data)                           # 返回(总正确结果/所有样本)，精确率


"""
训练bert_classifier分类器
"""
def train_bert_classifier(clf_model, train_comments, train_labels, config):

    tokenizer = config.tokenizer
    device = config.device
    loss = config.loss

    # 累计训练6万条数据 epochs 次，优化模型
    for epoch in tqdm(range(config.epoches), desc="Training Process:"):

        i, sum_loss = 0, 0  # 每次开始训练时， i 为 0 表示从第一条数据开始训练

        # 开始训练模型
        while i < len(train_comments):
            comments = train_comments[i: min(i + config.batch_size, len(train_comments))]  # 批量训练，每次训练8条样本数据
            tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device)

            # 将数据输入到bert分类器模型中，获得结果
            res = clf_model(tokens_X)

            # 批量获取实际结果信息
            y = torch.tensor(train_labels[i: min(i + config.batch_size, len(train_comments))]).reshape(-1).to(device)

            optimizer.zero_grad()  # 清空梯度
            l = loss(res, y)  # 计算损失
            l.backward()  # 后向传播
            optimizer.step()  # 更新梯度

            sum_loss += l.detach()  # 累加损失
            i += config.batch_size  # 样本下标累加            

        # 计算训练集与测试集的精度
        train_acc = evaluate(clf_model, train_comments, train_labels, config)
        logging.info(f'Training Accuracy: {train_acc.item()}')
        
        torch.cuda.empty_cache()

    torch.save(clf_model, 'checkpoints/clf_model.pt')


if __name__ == "__main__":

    comments_data = read_file('online_shopping_10_cats.csv')

    config = BERTConfig(batch_size=8, learning_rate=1e-5,epoches=20)

    split = 0.8
    split_line = int(len(comments_data) * split)

    # 划分训练集与测试集，并将pandas数据类型转化为列表类型
    train_X, train_y = list(comments_data[: split_line][0]), list(comments_data[: split_line][1])
    test_X, test_y = list(comments_data[split_line:][0]), list(comments_data[split_line:][1])

    clf_model = BERTClassifier(config=config).to(config.device)                     # BERTClassifier分类器，因为最终结果为2分类，所以输出维度为2，代表概率分布

    optimizer = torch.optim.AdamW(clf_model.parameters(), lr=config.learning_rate)      # 小批量随机梯度下降算法

    train_bert_classifier(clf_model, train_X, train_y, config)

    accurate = evaluate(clf_model, test_X, test_y, config)

    print('accurate:',accurate)
