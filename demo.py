import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


# 通过创建data.Dataset子类MyDataset来创建输入
class MyDataset(data.Dataset):
    # 类初始化: 初始化方法，传入数据文件夹路径。
    def __init__(self, root):
        self.imgs_path = root

    # 进行切片: 根据索引下标，获得相应的图片。
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path

    # 返回长度: 计算长度方法，返回整个数据文件夹下所有文件的个数。
    def __len__(self):
        return len(self.imgs_path)


class MyDatasetPro(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img).convert('L')
        data = self.transforms(pil_img)
        return data, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)


# 使用glob方法来获取数据图片的所有路径
imgs_path = list()
all_labels = list()
for i in range(3):
    # 读取./data/*/*.jpg的所以图片路径
    p = glob.glob(f'./MyData/{i}/*.jpg')
    imgs_path += p
    # 保存所有标签
    all_labels += [i] * len(p)

print(imgs_path)
print(all_labels)

# 利用自定义类MyDataset创建对象number_dataset
number_dataset = MyDataset(imgs_path)
# 创建dataloader
number_dataloader = torch.utils.data.DataLoader(number_dataset, batch_size=2)
# 打印第一个batch
print(next(iter(number_dataloader)))

# 对数据进行转换处理
transform = transforms.Compose([
    # 做的第一步转换
    transforms.Resize((100, 100)),
    # 第二步转换
    # 第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
    transforms.ToTensor()
])

# 定义batch_size 批次
BATCH_SIZE = 10

# 定义数据集
number_dataset = MyDatasetPro(imgs_path, all_labels, transform)
number_dataloader = data.DataLoader(
    number_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 打印第一个batch
imgs_batch, labels_batch = next(iter(number_dataloader))
print(imgs_batch.shape)

# 展示图片
plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i + 1)
    plt.title(label.item())
    plt.imshow(img)
plt.show()

# 划分测试集和训练集
index = np.random.permutation(len(imgs_path))

# 打乱数据
all_imgs_path = np.array(imgs_path)[index]
all_labels = np.array(imgs_path)[index]

# 划分训练集和测试集
s = int(len(all_imgs_path) * 0.8)

# 划分训练集和测试集
train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_imgs_path[s:]

# 创建训练集和测试集
train_ds = MyDatasetPro(train_imgs, train_labels, transform)
test_ds = MyDatasetPro(test_imgs, test_labels, transform)

# 创建训练集和测试集的dataloader
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
