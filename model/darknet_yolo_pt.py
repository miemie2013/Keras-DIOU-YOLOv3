#!/usr/bin/python
# -*- coding: utf-8 -*-
# pytorch_yolov3

import torch
import numpy as np

class YoloLoss(torch.nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def forward(self, y_pred, y_true):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            y_pred = y_pred.cuda()
            y_true = y_true.cuda()
        anchors_13x13 = [[116, 90], [156, 198], [373, 326]]
        anchors_26x26 = [[30, 61], [62, 45], [59, 119]]
        anchors_52x52 = [[10, 13], [16, 30], [33, 23]]

        grid_r = y_pred.shape[1]  # 表格行数，13、26、52
        grid_c = y_pred.shape[2]  # 表格列数，13、26、52
        class_num = y_pred.shape[4] - 5  # 类别数
        input_size = 416  # 输入图片大小

        loss = 0.0
        conf_weight = 1.0
        class_weight = 1.0
        giou_weight = 1.0

        m = y_pred.shape[0]
        mf = float(m)

        # 每个格子的3个框都计算损失
        for i in range(3):
            # 置信度loss
            conf_true = y_true[..., i, 4:5]
            conf_pred = y_pred[..., i, 4:5]
            conf_pred = torch.sigmoid(conf_pred)
            conf_loss = -conf_true * torch.log(conf_pred) - (1 - conf_true) * torch.log(1 - conf_pred)
            conf_loss = torch.sum(conf_loss) / mf
            loss += conf_weight*conf_loss

            # 分类loss
            class_true = y_true[..., i, 5:5+class_num]
            class_pred = y_pred[..., i, 5:5+class_num]
            class_pred = torch.sigmoid(class_pred)
            class_loss = -class_true * torch.log(class_pred) - (1 - class_true) * torch.log(1 - class_pred)
            class_loss = conf_true * class_loss  # 关键！这一步忽略那些没有物体的格子的分类损失，剩下3物体
            class_loss = torch.sum(class_loss) / mf
            loss += class_weight*class_loss

            # giou_loss
            xy_true = y_true[..., i, :2]
            wh_true = y_true[..., i, 2:4]
            txtytwth_pred = y_pred[..., i, :4]

            # 真实xywh 转换成x0 y0 x1 y1
            true_x0y0 = xy_true - wh_true / 2
            true_x1y1 = xy_true + wh_true / 2

            # 确定 格子边长 和 anchor大小
            grid_a = 0
            anchor = None
            if grid_r == 13:
                grid_a = 32
                anchor = anchors_13x13[i]
            elif grid_r == 26:
                grid_a = 16
                anchor = anchors_26x26[i]
            elif grid_r == 52:
                grid_a = 8
                anchor = anchors_52x52[i]
            anchor = torch.Tensor(anchor)

            offset = np.zeros((1, grid_c, grid_r, 2))
            for i1 in range(grid_c):
                for i2 in range(grid_r):
                    offset[0][i1][i2][0] = grid_a*i2
                    offset[0][i1][i2][1] = grid_a*i1
            offset = torch.Tensor(offset)
            if use_cuda:
                offset = offset.cuda()
                anchor = anchor.cuda()
            bxby = torch.sigmoid(txtytwth_pred[..., :2])*grid_a + offset
            bwbh = torch.exp(txtytwth_pred[..., 2:]) * anchor

            # left_up
            pred_x0y0 = bxby - bwbh / 2
            # right_down
            pred_x1y1 = bxby + bwbh / 2

            # 计算真实矩形 和 预测矩形面积
            true_boxes_area = wh_true[..., 0] * wh_true[..., 1]
            pred_boxes_area = bwbh[..., 0] * bwbh[..., 1]

            true_x0_pred_x0 = torch.cat([true_x0y0[..., 0:1], pred_x0y0[..., 0:1]], dim=3)
            true_y0_pred_y0 = torch.cat([true_x0y0[..., 1:2], pred_x0y0[..., 1:2]], dim=3)
            true_x1_pred_x1 = torch.cat([true_x1y1[..., 0:1], pred_x1y1[..., 0:1]], dim=3)
            true_y1_pred_y1 = torch.cat([true_x1y1[..., 1:2], pred_x1y1[..., 1:2]], dim=3)

            # 对应位置取最大，获取重叠区域矩形的左上角坐标
            inter_section_left  = torch.max(true_x0_pred_x0, dim=3, keepdim=True).values
            inter_section_up    = torch.max(true_y0_pred_y0, dim=3, keepdim=True).values
            # 对应位置取最小，获取重叠区域矩形的右下角坐标
            inter_section_right = torch.min(true_x1_pred_x1, dim=3, keepdim=True).values
            inter_section_down  = torch.min(true_y1_pred_y1, dim=3, keepdim=True).values

            inter_section_w = inter_section_right - inter_section_left
            inter_section_h = inter_section_down - inter_section_up
            # 宽或高只要有一个是负数，证明相离，iou=0
            inter_section_w = torch.relu(inter_section_w)
            inter_section_h = torch.relu(inter_section_h)

            # 计算iou
            inter_area = inter_section_w[..., 0] * inter_section_h[..., 0]  # 重叠区域面积
            union_area = true_boxes_area + pred_boxes_area - inter_area
            iou = inter_area / union_area

            # giou是2019年提出的，是在yolov3发布一年之后才提出的。。。
            # 对应位置取最小，获取C区域矩形的左上角坐标
            enclose_section_left  = torch.min(true_x0_pred_x0, dim=3, keepdim=True).values
            enclose_section_up    = torch.min(true_y0_pred_y0, dim=3, keepdim=True).values
            # 对应位置取最大，获取C区域矩形的右下角坐标
            enclose_section_right = torch.max(true_x1_pred_x1, dim=3, keepdim=True).values
            enclose_section_down  = torch.max(true_y1_pred_y1, dim=3, keepdim=True).values

            enclose_section_w = enclose_section_right - enclose_section_left
            enclose_section_h = enclose_section_down - enclose_section_up

            enclose_area = enclose_section_w[..., 0] * enclose_section_h[..., 0]  # C区域面积
            giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
            giou = torch.reshape(giou, (m, grid_c, grid_r, 1))

            # bbox损失 的系数  =  2-物体面积占图片面积的比例。  物体越小系数越大越受重视
            bbox_loss_scale = 2.0 - 1.0 * wh_true[:, :, :, 0:1] * wh_true[:, :, :, 1:2] / (input_size ** 2)

            giou_loss = conf_true * bbox_loss_scale * (1 - giou)  # 关键！这一步忽略那些没有物体的格子的giou损失
            giou_loss = torch.sum(giou_loss) / mf
            loss += giou_weight * giou_loss
        return loss

class Conv2dUnit(torch.nn.Module):
    def __init__(self, input_dim, filters, kernels, stride, padding):
        super(Conv2dUnit, self).__init__()
        self.conv = torch.nn.Conv2d(input_dim, filters, kernel_size=kernels, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(filters)
        self.leakyreLU = torch.nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyreLU(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dUnit(input_dim, filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters, 2*filters, (3, 3), stride=1, padding=1)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x

class StackResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters, n):
        super(StackResidualBlock, self).__init__()
        self.sequential = torch.nn.Sequential()
        for i in range(n):
            self.sequential.add_module('stack_%d' % (i+1,), ResidualBlock(input_dim, filters))
    def forward(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x

class Darknet(torch.nn.Module):
    def __init__(self, num_classes, initial_filters=32):
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        ''' darknet53部分，这里所有卷积层都没有偏移bias=False '''
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        self.stack_residual_block_1 = StackResidualBlock(i64, i32, n=1)

        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        self.stack_residual_block_2 = StackResidualBlock(i128, i64, n=2)

        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        self.stack_residual_block_3 = StackResidualBlock(i256, i128, n=8)

        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        self.stack_residual_block_4 = StackResidualBlock(i512, i256, n=8)

        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)
        self.stack_residual_block_5 = StackResidualBlock(i1024, i512, n=4)
        ''' darknet53部分结束 '''

        self.conv53 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.conv54 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv55 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.conv56 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv57 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)

        self.conv58 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv59 = torch.nn.Conv2d(i1024, 3*(num_classes + 5), kernel_size=(1, 1))

        self.conv60 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.conv61 = Conv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0)
        self.conv62 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv63 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.conv64 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv65 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)

        self.conv66 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv67 = torch.nn.Conv2d(i512, 3*(num_classes + 5), kernel_size=(1, 1))

        self.conv68 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.conv69 = Conv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0)
        self.conv70 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        self.conv71 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.conv72 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        self.conv73 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.conv74 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)

        self.conv75 = torch.nn.Conv2d(i256, 3*(num_classes + 5), kernel_size=(1, 1))

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stack_residual_block_1(x)
        x = self.conv3(x)
        x = self.stack_residual_block_2(x)
        x = self.conv4(x)
        act11 = self.stack_residual_block_3(x)
        x = self.conv5(act11)
        act19 = self.stack_residual_block_4(x)
        x = self.conv6(act19)
        act23 = self.stack_residual_block_5(x)

        x = self.conv53(act23)
        x = self.conv54(x)
        x = self.conv55(x)
        x = self.conv56(x)
        lkrelu57 = self.conv57(x)

        x = self.conv58(lkrelu57)
        y1 = self.conv59(x)
        y1 = y1.view(y1.size(0), 3, (self.num_classes + 5), y1.size(2), y1.size(3))  # reshape

        x = self.conv60(lkrelu57)
        x = self.upsample1(x)
        x = torch.cat((x, act19), dim=1)

        x = self.conv61(x)
        x = self.conv62(x)
        x = self.conv63(x)
        x = self.conv64(x)
        lkrelu64 = self.conv65(x)

        x = self.conv66(lkrelu64)
        y2 = self.conv67(x)
        y2 = y2.view(y2.size(0), 3, (self.num_classes + 5), y2.size(2), y2.size(3))  # reshape

        x = self.conv68(lkrelu64)
        x = self.upsample2(x)
        x = torch.cat((x, act11), dim=1)

        x = self.conv69(x)
        x = self.conv70(x)
        x = self.conv71(x)
        x = self.conv72(x)
        x = self.conv73(x)
        x = self.conv74(x)
        y3 = self.conv75(x)
        y3 = y3.view(y3.size(0), 3, (self.num_classes + 5), y3.size(2), y3.size(3))  # reshape

        # 相当于numpy的transpose()，交换下标
        y1 = y1.permute(0, 3, 4, 1, 2)
        y2 = y2.permute(0, 3, 4, 1, 2)
        y3 = y3.permute(0, 3, 4, 1, 2)
        return y1, y2, y3
