import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock # 用于更新特征和上下文网络
from extractor import BasicEncoder, SmallEncoder # 用于提取特征
from corr import CorrBlock, AlternateCorrBlock # 用于计算相关性
from utils.utils import bilinear_sampler, coords_grid, upflow8 # 用于计算光流

try:
    autocast = torch.cuda.amp.autocast # 用于自动混合精度计算
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__() # 初始化父类
        self.args = args # 这里的self.args是RAFT模型的参数, 用于配置RAFT模型

        if args.small: # 如果args.small为True, 则使用小模型, 否则使用大模型
            self.hidden_dim = hdim = 96 # 这里的hdim是隐藏维度, 用于配置RAFT模型
            self.context_dim = cdim = 64 # 这里的cdim是上下文维度, 用于配置RAFT模型
            args.corr_levels = 4 # 这里的args.corr_levels是相关级别, 用于配置RAFT模型
            args.corr_radius = 3 # 这里的args.corr_radius是相关半径, 用于配置RAFT模型
        
        else: # 如果args.small为False, 则使用大模型
            self.hidden_dim = hdim = 128 # 这里的hdim是隐藏维度, 用于配置RAFT模型
            self.context_dim = cdim = 128 # 这里的cdim是上下文维度, 用于配置RAFT模型
            args.corr_levels = 4 # 这里的args.corr_levels是相关级别, 用于配置RAFT模型
            args.corr_radius = 4 # 这里的args.corr_radius是相关半径, 用于配置RAFT模型
        if 'alternate_corr' not in self.args: # 如果args.alternate_corr不为True, 则使用交替相关, 否则使用普通相关
            self.args.alternate_corr = False # 这里的self.args.alternate_corr是交替相关, 用于配置RAFT模型

        # feature network, context network, and update block
        # SmallEncoder 和 BasicEncoder 都是编码器, 用于提取特征, 用于提取特征
        if args.small: # 如果args.small为True, 则使用小模型, 否则使用大模型
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)   #  self.fnet表示特征网络
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)   #  self.cnet表示上下文网络
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)   #  self.update_block表示更新块, 用于更新特征和上下文网络

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)          #  self.fnet表示特征网络
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)   #  self.cnet表示上下文网络
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)   #  self.update_block表示更新块, 用于更新特征和上下文网络

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape # 这里的N是批量大小, C是通道数, H是高度, W是宽度
        coords0 = coords_grid(N, H//8, W//8, device=img.device) # 创建坐标网格,coords0表示初始坐标网格
        coords1 = coords_grid(N, H//8, W//8, device=img.device) # 创建坐标网格,coords1表示目标坐标网格

        # optical flow computed as difference: flow = coords1 - coords0
        # 光流计算为两个坐标网格的差: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    # 前向传播, 用于估计光流 ，输入两张图像，输出光流
    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision): # 用于自动混合精度计算
            fmap1, fmap2 = self.fnet([image1, image2])   # 提取特征
        
        fmap1 = fmap1.float() # 将fmap1转换为浮点型
        fmap2 = fmap2.float() # 将fmap2转换为浮点型
        if self.args.alternate_corr: # 如果args.alternate_corr为True, 则使用交替相关, 否则使用普通相关
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius) # 创建交替相关函数
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius) # 创建普通相关  函数

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1) # 提取上下文特征
            net, inp = torch.split(cnet, [hdim, cdim], dim=1) # 分割上下文特征，net表示隐藏层特征，inp表示输入层特征
            net = torch.tanh(net) # 将net转换为tanh函数,作用是将net的值限制在-1到1之间，用于激活函数
            inp = torch.relu(inp) # 将inp转换为relu函数,作用是将inp的值限制在0到无穷大之间，用于激活函数

        coords0, coords1 = self.initialize_flow(image1) # 初始化坐标网格

        if flow_init is not None: # 如果flow_init不为None, 则将flow_init添加到coords1中
            coords1 = coords1 + flow_init # 将flow_init添加到coords1中

        flow_predictions = []
        for itr in range(iters): # 迭代次数
            coords1 = coords1.detach() # 将coords1转换为不可训练的.detach()表示将coords1转换为不可训练的，用于计算梯度
            corr = corr_fn(coords1) # index correlation volume # 计算相关性

            flow = coords1 - coords0 # 光流计算为两个坐标网格的差: flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision): # autocast表示自动混合精度计算
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0) # 上采样光流
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
