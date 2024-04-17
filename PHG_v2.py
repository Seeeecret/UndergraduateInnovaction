import cv2
import numpy as np
import torch
import torch.nn as tnn
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.nn import init
from torchvision import transforms, datasets
import torch.nn.functional as F
from tqdm import tqdm
import json

def conv_layer(chann_in, chann_out, k_size, stride, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, stride=stride, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.LeakyReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, s_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], s_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(layers)


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Dropout(p=0.5),
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class PHGNet(tnn.Module):
    def __init__(self, n_classes=1000):
        super(PHGNet, self).__init__()
        self.layer1_1 = conv_layer(3, 7, 3, 1, 1)
        self.layer1_2 = conv_layer(3, 7, 3, 1, 1)
        self.layer2 = conv_layer(16, 32, 5, 1, 0)
        self.layer3 = vgg_conv_block([32], [64], [5], [1], [0], 2, 2)
        self.layer4 = conv_layer(64, 128, 3, 1, 1)
        self.layer5 = vgg_conv_block([128], [256], [3], [3], [1], 4, 4)

        # FC layers
        self.layer6 = vgg_fc_layer(256  10  10, 4096)
        self.layer7 = vgg_fc_layer(4096, 9)

    def forward(self, images_1, images_2, image1_edge, image2_edge):
        out1 = self.layer1_1(images_1)
        out2 = self.layer1_2(images_2)
        out = torch.cat([out1, out2, image1_edge, image2_edge], 1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)

        out = self.layer6(out)
        out = self.layer7(out)

        return out

#edge detection
def edge_detection(images):
    results=[]
    for i, img in enumerate(images):
        img = cv2.imread(img, 3)
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
        # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        results.append(result)

    return results

#Skin color detection
def color_detection(images):
    results=[]
    for i, img in enumerate(images):
        img = cv2.imread(img, 3)
        ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        (y,cr,cb)= cv2.split(ycrcb)
        skin = np.zeros(cr.shape,dtype= np.uint8)
        (x,y)= cr.shape
        for i in range(0,x):
            for j in range(0,y):
                if (cr[i][j]>140) and (cr[i][j])<175 and (cr[i][j]>100) and (cb[i][j])<120:
                    skin[i][j]= 255
                else:
                    skin[i][j] = 0
        dst = cv2.bitwise_and(img,img,mask=skin)
        results.append(dst)

    return results

# 定义模型初始化函数
def weigth_init(m):
    if isinstance(m, tnn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, tnn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, tnn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def main():
    # 定义参数
    image_size = 244
    batch_size = 16
    ngpu = 1
    epochs = 500
    learning_rate = 0.0001
    n_class = 9

    # device = torch.device(cuda0 if (torch.cuda.is_available() and ngpu  0) else cpu)
    #     正确语句:
    device = torch.device('cuda0' if (torch.cuda.is_available() and ngpu ) else 'cpu')
    # print(using {} device.format(device))

    #     正确语句:
    print('using {} device.'.format(device))

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.2]),
    ])

    # 定义数据集、loader数据集a
    train_dataset1 = datasets.ImageFolder(r'C\\PHG\\Dataset\\PHG_Angle\\train\\pitch', transform)
    train_dataset2 = datasets.ImageFolder(r'C\\PHG\\Dataset\\PHG_Angle\\train\\side', transform)
    val_dataset = datasets.ImageFolder(r'C\\PHG\\Dataset\\PHG_Angle\\val\\pitch',transform)

    nw = min([os.cpu_count(), batch_size if batch_size  1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    trainLoader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=batch_size, shuffle=False, num_workers=nw)
    trainLoader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=batch_size, shuffle=False, num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    phg_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in phg_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 将模型转移到GPU上运算
    phgnet = PHGNet().to(device)
    # 初始化模型参数
    phgnet.apply(weigth_init)

    # 定义损失和优化函数
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(phgnet.parameters(), lr=learning_rate)
    train_steps = len(trainLoader)
    image2_list = []
    image2_list.append(data for data in enumerate(trainLoader2))
    for epoch in range(epochs):
        running_loss = 0.0
        phgnet.train()
        train_bar = tqdm(trainLoader)
        for step, data in enumerate(train_bar):
            images1, labels = data
            images2 = image2_list[step]
            image1_edge = color_detection(images1)
            image2_edge = color_detection(images2)
            optimizer.zero_grad()
            outputs = phgnet(images1.to(device), images2.to(device), image1_edge.to(device), images2_edge.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = train epoch[{}{}] loss{.3f}.format(epoch + 1,
                                                                     epochs,
                                                                     loss)
    net.eval()
    acc = 0.0  # accumulate accurate number  epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader, colour='green')
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_images_color = color_detection(val_images)
            outputs = net(val_images.to(device), val_images.to(device), val_images_color.to(device), val_images_color.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc  val_num
    print('[epoch %d] train_loss %.3f  val_accuracy %.3f' %
          (epoch + 1, running_loss  train_steps, val_accurate))

    if val_accurate  best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(),'PHG.pth')


print('Finished Training')

if __name__ == '__main__':
    main()
