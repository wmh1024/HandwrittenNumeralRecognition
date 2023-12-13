import time
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

print('---------------INFO----------------------')
# cuda是否可用
cuda = torch.cuda.is_available()
# 打印mps是否可用
print("cuda: " + str(cuda))
# mps是否可用
mps = torch.backends.mps.is_available()
# 打印mps是否可用
print("msp: " + str(mps))
# 优化器 => 学习率
learning_rate = 0.003
# 正确率列表
correct_list = []
# mps设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# epoch 学习轮次
epoch = 20
# 训练集批次
batch_size_train = 32
# 测试集批次
batch_size_test = 100
# 批次大小
BATCH_SIZE = 2


class MyDataset(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        # 转换处理
        self.transforms = transform

    # 进行切片
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img).convert('L')
        data = self.transforms(pil_img)
        # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        # 将label转换为长整型
        return data, torch.tensor(label, dtype=torch.long)

    # 返回长度
    def __len__(self):
        return len(self.imgs)


# 使用glob方法来获取数据图片的所有路径
imgs_path = list()
all_labels = list()
for i in range(10):
    # 读取 ./MyData/{i}/*.jpg 的所有图片路径
    p = glob.glob(f'./MyData/{i}/*.jpg')
    imgs_path += p
    # 保存所有标签
    all_labels += [i] * len(p)

# 对数据进行转换处理
transform = transforms.Compose([
    # 做的第一步转换
    transforms.Resize((32, 32)),
    # 第二步转换：将图片转换成Tensor
    transforms.ToTensor(),
    # 第三步转换：归一化
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# 划分测试集和训练集
index = np.random.permutation(len(imgs_path))

# 将所有图片路径和标签打乱
all_imgs_path = np.array(imgs_path)[index]
all_labels = np.array(all_labels)[index]

# 打印数据大小
print("len(all_labels): ", len(all_labels))

# 划分训练集和测试集
s = int(len(all_imgs_path) * 0.8)

# 划分训练集和测试集
train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]

# 创建训练集和测试集
train_ds = MyDataset(train_imgs, train_labels, transform)
test_ds = MyDataset(test_imgs, test_labels, transform)

# 创建训练集和测试集的dataloader
train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

# 数据大小
train_data_size = len(train_loader)
test_data_size = len(test_loader)

# 打印数据大小
print('train_data_size: ', train_data_size)
print('test_data_size: ', test_data_size)


# 定义模型 继承nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 构建简单的序贯模型
        self.model = nn.Sequential(
            # nn.Conv2d：一个二维卷积层，输入通道数为1，输出通道数为32，卷积核大小为3x3，步长为1，填充为1。
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU：一个ReLU激活函数层，用于引入非线性。
            nn.ReLU(),
            # nn.MaxPool2d：一个二维最大池化层，池化核大小为2x2，步长为2。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d：另一个二维卷积层，输入通道数为32，输出通道数为64，卷积核大小为3x3，步长为1，填充为1。
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU：另一个ReLU激活函数层。
            nn.ReLU(),
            # nn.MaxPool2d：另一个二维最大池化层。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Flatten：将输入的多维张量展平为一维向量。
            nn.Flatten(),
            # nn.Linear：一个全连接层，输入特征数为40000，输出特征数为128。
            nn.Linear(in_features=4096, out_features=128),
            # nn.Linear：另一个全连接层，输入特征数为128，输出特征数为10。
            nn.Linear(in_features=128, out_features=10)
        )
        # 若使用GPU，则将模型移动到GPU上。
        if cuda:
            self.model = self.model.cuda()
        elif mps:
            self.model = self.model.to(device)

    # 前向传播
    def forward(self, x):
        return self.model(x)


# 定义损失函数
if cuda:
    loss_fn = nn.CrossEntropyLoss().cuda()
elif mps:
    loss_fn = nn.CrossEntropyLoss().to(device)
else:
    loss_fn = nn.CrossEntropyLoss()

# 构建模型
net = Net()
# 构建优化器
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# 构建TensorBoard
writer = SummaryWriter(log_dir='logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')))

# 记录训练步数
total_train_step = 0
# 记录损失率列表
train_loss_list = []


# 训练函数 (epoch: 训练轮次)
def train(epoch):
    # 记录训练步数
    global total_train_step
    total_train_step = 0

    # 遍历训练集
    for train_data in train_loader:
        # 获取数据
        imgs, targets = train_data
        # 若使用GPU，则将数据移动到GPU上，使用msp。
        if cuda:
            imgs = imgs.cuda()
            targets = targets.cuda()
        elif mps:
            imgs = imgs.to(device)
            targets = targets.to(device)
        # 清零梯度
        optimizer.zero_grad()
        # 前向计算
        outputs = net(imgs)
        # 计算损失
        loss = loss_fn(outputs, targets)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 记录损失率
        train_loss_list.append(loss.item())
        # 打印训练信息
        if total_train_step % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, total_train_step, train_data_size,
                100. * total_train_step / train_data_size, loss.item()
            ))
        # 记录损失
        writer.add_scalar('loss', loss.item(), total_train_step)
        # 训练步数加 1
        total_train_step += 1


# 测试函数
def FunTest():
    # 正确数
    correct = 0
    # 总数
    total = 0
    # 遍历测试集
    with torch.no_grad():
        # 遍历测试集
        for test_data in test_loader:
            # 获取数据
            imgs, targets = test_data
            # 若使用GPU，则将数据移动到GPU上。
            if cuda:
                imgs, targets = imgs.cuda(), targets.cuda()
            elif mps:
                imgs = imgs.to(device)
                targets = targets.to(device)
            # 经行前向计算
            outputs = net(imgs)
            # 获取最大值的位置
            # _: 表示忽略第一个返回值
            # predicted: 预测值
            _, predicted = torch.max(outputs.data, 1)
            # 统计正确数
            total += targets.size(0)
            # 统计预测正确数
            correct += (predicted == targets).sum().item()
    # 打印测试信息
    print('Test Accuracy: {}/{} ({:.3f}%)'.format(correct, total, 100. * correct / total))
    # 记录正确率到correct_list
    correct_list.append(correct / total)
    # 返回正确率
    return correct / total


if __name__ == '__main__':
    start = time.time()
    # 训练模型
    for i in range(1, epoch + 1):
        # 打印训练信息
        print(f"---------------Epoch: {i}----------------------")
        # 训练模型
        train(i)
        # 测试模型
        FunTest()
        # 记录测试准确率
        writer.add_scalar('test_accuracy', FunTest(), total_train_step)
        # 保存模型
        torch.save(net, 'model/my_data_model.pth')
        # 打印保存信息
        print('Saved Model')
    end = time.time()
    print('---------------Train End----------------------')
    print(f"总共用时: {end - start : .2f}s")
    print(f"正确率: {max(correct_list) * 100. : .2f}%")

    # 绘制损失曲线
    iterations = range(1, len(train_loss_list) + 1)
    plt.plot(iterations, train_loss_list)
    # 设置坐标轴标签和标题
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    # 显示图形
    plt.show()
