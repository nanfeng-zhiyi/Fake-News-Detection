import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import jieba
import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from model import CNNLSTMTransformerClassifier  # 导入模型
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard


# 检查 GPU 是否可用
device = torch.device('cuda')
print(f"Using device: {device}")

# 加载数据
test_raw = pd.read_csv('./combined.csv')
val_raw = pd.read_csv('./train.csv')
print('训练大小:', test_raw.shape)
print('验证大小:', val_raw.shape)

# 文本清理函数
def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        soup = BeautifulSoup(content, 'html.parser')

        # 删除脚本和样式内容
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # 提取纯文本
        text = soup.get_text(separator='\n')

        # 去除多余的空白行
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)

        return text

def clean_text(text):
    text = re.sub(r'[^A-Za-z一-龥]+', ' ', text)
    return text.strip()

# 数据清理
# 修正列名拼写
titles = test_raw['Title'].fillna('') + test_raw['Ofiicial Account Name'].fillna('') + test_raw["Report Content"].fillna('')
train_num = len(titles)

val_titles = val_raw['Title'].fillna('') + val_raw['Ofiicial Account Name'].fillna('') + val_raw["Report Content"].fillna('')

# 划分训练集和测试集
X = titles
y = test_raw['label']

_X_val = val_titles
_y_val = val_raw['label']

# 分词
X = X.apply(lambda x: ' '.join(jieba.cut(clean_text(x))))

_X_val = _X_val.apply(lambda x: ' '.join(jieba.cut(clean_text(x))))

X = X.fillna('')  # 将 NaN 填充为空字符串
_X_val = _X_val.fillna('')

# 划分训练集、验证集
X_train, y_train = X, y

X_val, y_val = _X_val, _y_val

# 文本向量化部分
word_dict = {}  # 手动构建词汇表
for text in X_train:
    for word in text.split():
        if word not in word_dict:
            word_dict[word] = len(word_dict) + 1

# 将文本转换为序列
def texts_to_sequences(texts, word_index):
    sequences = []
    for text in texts:
        sequences.append([word_index.get(word, 0) for word in text.split()])
    return sequences

X_train_seq = texts_to_sequences(X_train, word_dict)
X_val_seq = texts_to_sequences(X_val, word_dict)

# max_length = max([len(x) for x in X_train_seq])
# max_length_val = max([len(x) for x in X_val_seq])
max_length = 256
max_length_val = 256
print('max_length:', max_length)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length_val, padding='post')

y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

# 数据集封装为 PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 学习率调度器
def adjust_learning_rate(optimizer, factor=0.5):
    """将学习率按指定系数下调"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor
    print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']}")

# 初始化 TensorBoard
writer = SummaryWriter('/root/tf-logs')

# 模型训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    global best_f1_score, best_accuracy, best_model_path  # 引用外部变量用于记录最优分数
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 在每个 epoch 内部为 DataLoader 添加进度条
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
            for texts, labels in tepoch:
                texts, labels = texts.to(device), labels.to(device)  # 移动数据到GPU
                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 将每个 batch 的损失值和准确率传递给进度条
                tepoch.set_postfix(loss=total_loss / len(train_loader), accuracy=100 * correct / total)

        # 调整学习率
        scheduler.step(total_loss)

        # 记录训练损失和准确率到 TensorBoard
        writer.add_scalar('Training Loss', total_loss / len(train_loader), epoch)
        writer.add_scalar('Training Accuracy', 100 * correct / total, epoch)

        # 评估模型在验证集上的表现
        f1, accuracy = evaluate_model(model, val_loader)
        print(f'验证集上的表现：f1分数{f1}, 准确率：{accuracy}')

        # 如果验证集的 F1 score 优于之前的最优 F1 score，或者在 F1 score 相同情况下准确率更高，则保存模型
        if f1 > best_f1_score:
            best_f1_score = f1
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with F1 Score: {best_f1_score:.4f}, Accuracy: {best_accuracy:.4f} at Epoch {epoch + 1}')

# 模型评估函数
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)  # 移动数据到GPU
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, accuracy

# 初始化训练集和验证集的 Dataset 和 DataLoader
train_dataset = TextDataset(X_train_pad, y_train)
val_dataset = TextDataset(X_val_pad, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 初始化模型
vocab_size = len(word_dict) + 1
embed_dim = 384
hidden_dim = 384  # 增加隐藏层大小
output_dim = len(set(y_train))

# 初始化模型并移动到GPU
model = CNNLSTMTransformerClassifier(vocab_size, embed_dim, hidden_dim, output_dim).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # 增加 weight_decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)  # 调整学习率调度策略

# 初始化最优 F1 score、最优准确率和模型保存路径
best_f1_score = 0.0
best_accuracy = 0.0
best_model_path = 'best_model_state.pth'

# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50)
# 保存模型状态
torch.save(model.state_dict(), 'model_state.pth')

# 保存词典
with open('word_dict.pkl', 'wb') as f:
    pickle.dump(word_dict, f)
