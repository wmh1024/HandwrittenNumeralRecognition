import time
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# cuda
cuda = torch.cuda.is_available()
# 打印mps是否可用
print("cuda: " + str(cuda))
# mps是否可用
mps = torch.backends.mps.is_available()
# 打印mps是否可用
print("msp: " + str(mps))
# 优化器 => 学习率
learning_rate = 0.01
# epoch 学习轮次
epoch = 10

# mps设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 训练集批次
batch_size_train = 64
# 测试集批次
batch_size_test = 100

# # 加载数据
# # tarnsform 转换: 归一化
# train_loader = DataLoader(torchvision.datasets.MNIST("data", train=True, download=True,
#                                                      transform=torchvision.transforms.Compose(
#                                                          [torchvision.transforms.ToTensor(),
#                                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
#                           batch_size=batch_size_train, shuffle=True)
# test_loader = DataLoader(torchvision.datasets.MNIST("data", train=False, download=True,
#                                                     transform=torchvision.transforms.Compose(
#                                                         [torchvision.transforms.ToTensor(),
#                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
#                          batch_size=batch_size_test, shuffle=True)

# todo

# 数据大小
train_data_size = len(train_loader)
test_data_size = len(test_loader)
print('train_data_size: ', train_data_size)
print('test_data_size: ', test_data_size)

'''
# enumerate()迭代器 用于遍历test_loader中的数据
examples = enumerate(test_loader)
# next() 获取迭代器下一个元素
# batch_idx => 当前批次的索引
# (example_data, example_targets)  => （当前批次的输入数据，对应的目标数据）
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
print(example_targets)

# 创建新的图形对象
fig = plt.figure()
# 包含20个子图
for i in range(20):
    # 创建一个4行5列的子图网格
    plt.subplot(4, 5, i + 1)
    # 自动调整子图的布局
    plt.tight_layout()
    # 用于显示输入数据
    # example_data[i]表示第i个样本的输入数据
    # example_data[i][0]表示该输入数据的第一个通道（假设输入数据是一个灰度图像）。
    # cmap='gray'指定了使用灰度色彩映射来显示图像。
    # interpolation='none'表示不进行插值，以保持图像的原始像素。
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    # 隐藏子图的刻度标签
    plt.xticks([])
    plt.yticks([])
# 展示图形
plt.show()
'''


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
            # nn.Linear：一个全连接层，输入特征数为3136，输出特征数为128。
            nn.Linear(in_features=3136, out_features=128),
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


# 损失函数
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


# 训练函数 epoch: 训练轮次
def train(epoch):
    # 记录训练步数
    global total_train_step
    total_train_step = 0
    # 记录损失率列表
    train_loss_list = []
    # 遍历训练集
    for data in train_loader:
        # 获取数据
        imgs, targets = data
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
        if total_train_step % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.6f}'.format(
                epoch, total_train_step, train_data_size,
                100. * total_train_step / train_data_size, loss.item()
            ))
        # 记录损失
        writer.add_scalar('loss', loss.item(), total_train_step)
        # 训练步数加1
        total_train_step += 1
    iterations = range(1, len(train_loss_list) + 1)
    plt.plot(iterations, train_loss_list)
    # 设置坐标轴标签和标题
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - epoch: {epoch}')
    # 显示图形
    plt.show()


# 测试函数
def FunTest():
    # 正确数
    correct = 0
    # 总数
    total = 0
    # 遍历测试集
    with torch.no_grad():
        # 遍历测试集
        for data in test_loader:
            # 获取数据
            imgs, targets = data
            # 若使用GPU，则将数据移动到GPU上。
            if cuda:
                imgs, targets = imgs.cuda(), targets.cuda()
            elif mps:
                imgs, targets = imgs.to(device), targets.to(device)
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
    # 返回正确率
    return correct / total


if __name__ == '__main__':
    start = time.time()
    # 训练模型
    for i in range(1, epoch + 1):
        # 打印训练信息
        print("---------------Epoch: {}----------------------".format(i))
        # 训练模型
        train(i)
        # 测试模型
        FunTest()
        # 记录测试准确率
        writer.add_scalar('test_accuracy', FunTest(), total_train_step)
        # 保存模型
        torch.save(net, 'model/mnist_model.pth')
        # 打印保存信息
        print('Saved Model')
    end = time.time()
    print(f"总共用时: {end - start}s")
