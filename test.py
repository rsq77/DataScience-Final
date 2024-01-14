import pandas as pd
import torch
import pandas as pd
import torch
from train import BERTClassifier
from train import read_file
from train import BERTConfig
from transformers import BertTokenizer
# from train import predict
import warnings
warnings.filterwarnings('ignore')


# 读取CSV文件
df = pd.read_csv('data/用户评论（高中低端）-约26w-高端.csv')
print(len(df))
df = df.dropna(subset=['描述'])
print(len(df))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = BERTConfig(batch_size=8, learning_rate=1e-5,epoches=20)
clf_model = BERTClassifier(config=config).to(device)                     # BERTClassifier分类器，因为最终结果为2分类，所以输出维度为2，代表概率分布
clf_model = torch.load('checkpoints/clf_model.pt').to(device)

"""
测试函数，用以输入评论并输出预测结果
"""
def predict(cls_model, comments_data, config):
    tokenizer = config.tokenizer
    device = config.device

    all_res = []  # 用于保存每次循环的结果

    i = 0
    while i < len(comments_data):
        comments = comments_data[i: min(i + config.batch_size, len(comments_data))]

        tokens_X = tokenizer(comments, padding=True, truncation=True, return_tensors='pt').to(device)

        res = clf_model(tokens_X).to(device)
        res = res.argmax(axis=1)

        all_res.append(res)
        i += config.batch_size

    total_res = torch.cat(all_res).view(-1)  # 将所有结果拼接并展平为一维数组

    return total_res.tolist()

# 调用predict函数进行情感分析
predictions = predict(clf_model, df['描述'].tolist(), config)

# 将预测结果添加到DataFrame中
df['正负面'] = predictions

# 将结果保存到新的CSV文件
df.to_csv('output/用户评论（高中低端）-约26w-高端.csv', index=False)