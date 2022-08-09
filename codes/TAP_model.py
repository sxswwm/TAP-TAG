import math
import random

import torch
from torch import nn
import torch.nn.functional as F
import heapq
import numpy as np
from sklearn.preprocessing import Normalizer
from TAP_dataloader import TAPTestDataset, ENTITY_LABEL_NAMES
from torch.utils.data import DataLoader
from gensim.models import word2vec


# 借用COMPOSE模型对文本信息的处理方法
# 使用一维卷积对句向量进行特征提取，用high-way CNN提取句子的语义信息
from TAP_dataloader import get_all_labeled_recipes


class CausalConv1d(torch.nn.Conv1d):  # 一维卷积
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,  # 输入信号的通道，文本分类中就是词向量的维度
            out_channels,  # 卷积产生的通道，有多少个通道，就需要多少个一维卷积
            kernel_size=kernel_size,  # 卷积核的尺寸
            stride=stride,  # 卷积步长
            padding=0,
            dilation=dilation,  # 卷积核元素之间的间距
            groups=groups,  # 从输入通道到输出通道的阻塞连接数
            bias=bias)  # 如果bias=true，添加偏置

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


# torch.nn.functional.pad(input, pad, mode='constant', value=0)
# input需要扩充的tensor，可以是图像数据，抑或是特征矩阵数据
# pad扩充维度，用于预先定义出某维度上的扩充参数
# mode扩充方法，’constant‘, ‘reflect’ or ‘replicate’三种模式，分别表示常量，反射，复制
# value扩充时指定补充值，但是value只在mode='constant’有效，即使用value填充在扩充出的新维度位置，
# 而在’reflect’和’replicate’模式下，value不可赋值

class HighwayBlock(nn.Module):
    def __init__(self, input_dim, kernel_size):
        super(HighwayBlock, self).__init__()
        self.conv_t = CausalConv1d(input_dim, input_dim, kernel_size)
        self.conv_z = CausalConv1d(input_dim, input_dim, kernel_size)

    # input输入，input_dim输入信号的通道数
    def forward(self, input):
        t = torch.sigmoid(self.conv_t(input))
        z = t * self.conv_z(input) + (1 - t) * input
        return z


class TAP_CNN(nn.Module):
    def __init__(self, input_dim=128, output_dim=32):
        super(TAP_CNN, self).__init__()
        # Input: batch_size * sentence_len * embd_dimm
        self.input_dim = input_dim
        self.output_dim = output_dim

        # High-way network for word features
        self.init_conv1 = CausalConv1d(input_dim, output_dim, 1)
        self.init_conv2 = CausalConv1d(input_dim, output_dim, 3)
        self.init_conv3 = CausalConv1d(input_dim, output_dim, 5)
        self.init_conv4 = CausalConv1d(input_dim, output_dim, 7)
        self.highway1 = HighwayBlock(4 * output_dim, 3)
        self.highway2 = HighwayBlock(4 * output_dim, 3)
        self.highway3 = HighwayBlock(4 * output_dim, 3)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            # nn.Linear(2000, 1000),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
        )
        # 在由多个输入平面组成的输入信号上应用一维自适应最大池化，帮助将输入维度以需要的维度输出

    def forward(self, input):
        # Input: B * L * embd_dim
        input = input.permute(0, 2, 1)
        conv1 = self.init_conv1(input)
        conv2 = self.init_conv2(input)
        conv3 = self.init_conv3(input)
        conv4 = self.init_conv4(input)
        concat = torch.cat((conv1, conv2, conv3, conv4), dim=1)

        highway_res = self.highway1(concat)
        highway_res = torch.relu(highway_res)

        highway_res = self.highway2(highway_res)
        highway_res = torch.relu(highway_res)
        highway_res = self.highway3(highway_res)
        highway_res = torch.relu(highway_res)

        pooled_res = self.pool(highway_res)
        output = pooled_res.squeeze(-1)
        #  1*embd_dim
        return output

    def TAP_convd(self, tap_sample):
        # 1.选取正样本,并进行卷积
        tap_convd = torch.FloatTensor().cuda()
        for vec in tap_sample:
            vec_convd = self.forward(vec.float().cuda())
            vec_convd = (vec_convd / torch.norm(vec_convd, p=2))
            tap_convd = torch.cat((tap_convd, vec_convd), dim=0)

        return tap_convd

    def recipe_sim(self, pos_convd, neg_convd, args):
        cos_sim = torch.nn.CosineEmbeddingLoss(margin=0.1, reduction='mean')
        a = pos_convd.shape[0]
        b = neg_convd.shape[0]
        a_rand = torch.randperm(a)  # 打乱顺序排列正样本和负样本
        b_rand = torch.randperm(b)
        pos_recipes_rand = torch.FloatTensor(a, 128).cuda()
        neg_recipes_rand = torch.FloatTensor(b, 128).cuda()
        for i in range(a):
            pos_recipes_rand[i] = pos_convd[a_rand[i]]
        for i in range(b):
            neg_recipes_rand[i] = neg_convd[b_rand[i]]

        pos = torch.ones(a).cuda()
        neg = -1 * torch.ones(b).cuda()

        pos_match_loss = cos_sim(pos_convd, pos_recipes_rand, pos)

        if a > args.tap_negative_sample_size:
            pos_recipes_rand = pos_recipes_rand[:args.tap_negative_sample_size]
        elif a < args.tap_negative_sample_size:
            a_rand = np.random.randint(a, size=args.tap_negative_sample_size)
            pos_recipes_rand = torch.FloatTensor(args.tap_negative_sample_size, 128).cuda()
            for i in range(args.tap_negative_sample_size):
                pos_recipes_rand[i] = pos_convd[a_rand[i]]

        neg_match_loss = cos_sim(pos_recipes_rand, neg_recipes_rand, neg)

        return pos_match_loss, neg_match_loss

    def TAP_KGE_match_loss(self, tap_positive_nums, pos_convd, relation_embd):
        cos_sim = torch.nn.CosineEmbeddingLoss(margin=0.1, reduction='mean')
        length = len(tap_positive_nums)
        pos_relation_embd = torch.FloatTensor(length, 128).cuda()
        pos = torch.ones(length).cuda()
        for i in range(length):
            x = torch.flatten(relation_embd[tap_positive_nums[i]], 1)
            x = self.classifier(x)
            x = torch.relu(x)
            pos_relation_embd[i] = x
        match_loss = cos_sim(pos_convd, pos_relation_embd, pos)

        return match_loss

    def test_step(self, tap_model, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        tap_model.eval()
        # Prepare dataloader for evaluation
        test_tap_dataloader = DataLoader(
            TAPTestDataset(args.tap_data_path),
            batch_size=len(ENTITY_LABEL_NAMES),
            shuffle=False,
            num_workers=0,
            collate_fn=TAPTestDataset.collate_fn
        )
        test_dataset_list = [test_tap_dataloader]

        recipe_model = word2vec.Word2Vec.load(
            'save/recipe.model'
        )
        with torch.no_grad():
            TRUE = []
            FALSE = []
            all_labels_rules_embds, all_labels = get_all_labeled_recipes(recipe_model)
            all_labels_rules_embds = self.TAP_convd(all_labels_rules_embds)
            for positive_rules_embds, positive_labels in test_dataset_list:
                for positive_rule_embd, positive_label in zip(positive_rules_embds, positive_labels):
                    pos_convd = self.TAP_convd(positive_rule_embd)
                    score = pos_convd * all_labels_rules_embds
                    topk = heapq.nlargest(1, range(len(score)), score.__getitem__)
                    k = topk[0]
                    label = all_labels[k]
                    if k == positive_label:
                        TRUE.append(1)
                    else:
                        FALSE.append(-1)

        return len(TRUE)/(len(TRUE) + len(FALSE))


ENTITY_LABEL_NAMES = [
    'air_conditioner', 'Android', 'calendar', 'camera', 'car', 'dog', 'timer', 'book',
    'door', 'dryer', 'email', 'Facebook', 'light', 'oven', 'rain', 'window', 'photo',
    'refrigerator', 'room', 'switch', 'telephone', 'thermostat', 'Google', 'humidity',
    'message', 'video'
]
