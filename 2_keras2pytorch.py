#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2019-12-20 14:38:26
#   Description : 将keras模型导出为pytorch模型。
#                 需要修改362行（读取的模型名'aaaa_bgr.h5'，这个模型是1_lambda2model.py脚本提取出来的模型）、
#                 365行（物品类别数80、初始卷积核个数32）、366行（导出的pytorch模型名'aaaa_bgr.pt'）。
#
#================================================================

import keras
import torch

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


def find(base_model, conv2d_name, batch_normalization_name):
    i1, i2 = -1, -1
    for i in range(len(base_model.layers)):
        if base_model.layers[i].name == conv2d_name:
            i1 = i
        if base_model.layers[i].name == batch_normalization_name:
            i2 = i
    return i1, i2


def aaaaaaa(conv, bn, cccccccc):
    w = conv.get_weights()[0]
    y = bn.get_weights()[0]
    b = bn.get_weights()[1]
    m = bn.get_weights()[2]
    v = bn.get_weights()[3]

    w = w.transpose(3, 2, 0, 1)

    conv2, bn2 = cccccccc.conv, cccccccc.bn

    conv2.weight.data = torch.Tensor(w)
    bn2.weight.data = torch.Tensor(y)
    bn2.bias.data = torch.Tensor(b)
    bn2.running_mean.data = torch.Tensor(m)
    bn2.running_var.data = torch.Tensor(v)


def aaaaaaa2(conv, cccccccc):
    w = conv.get_weights()[0]
    b = conv.get_weights()[1]

    w = w.transpose(3, 2, 0, 1)

    cccccccc.weight.data = torch.Tensor(w)
    cccccccc.bias.data = torch.Tensor(b)

def bbbbbbbbbbb(base_model, stack_residual_block, start_index):
    i = start_index
    for residual_block in stack_residual_block.sequential:
        conv1 = residual_block.conv1
        conv2 = residual_block.conv2

        i1, i2 = find(base_model, 'conv2d_%d'%(i, ), 'batch_normalization_%d'%(i, ))
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], conv1)
        i1, i2 = find(base_model, 'conv2d_%d'%(i+1, ), 'batch_normalization_%d'%(i+1, ))
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], conv2)
        i += 2


class Darknet(torch.nn.Module):
    def __init__(self, base_model, num_classes, initial_filters=32):
        super(Darknet, self).__init__()
        self.num_classes = num_classes



        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        ''' darknet53部分，这里所有卷积层都没有偏移bias=False '''

        i1, i2 = find(base_model, 'conv2d_1', 'batch_normalization_1')
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv1)


        dd = 2
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv2)


        self.stack_residual_block_1 = StackResidualBlock(i64, i32, n=1)
        bbbbbbbbbbb(base_model, self.stack_residual_block_1, start_index=3)




        dd = 5
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv3)
        self.stack_residual_block_2 = StackResidualBlock(i128, i64, n=2)
        bbbbbbbbbbb(base_model, self.stack_residual_block_2, start_index=6)



        dd = 10
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv4)
        self.stack_residual_block_3 = StackResidualBlock(i256, i128, n=8)
        bbbbbbbbbbb(base_model, self.stack_residual_block_3, start_index=11)



        dd = 27
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv5)
        self.stack_residual_block_4 = StackResidualBlock(i512, i256, n=8)
        bbbbbbbbbbb(base_model, self.stack_residual_block_4, start_index=28)



        dd = 44
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv6)
        self.stack_residual_block_5 = StackResidualBlock(i1024, i512, n=4)
        bbbbbbbbbbb(base_model, self.stack_residual_block_5, start_index=45)


        ''' darknet53部分结束 '''

        dd = 53
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv53 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv53)
        dd = 54
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv54 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv54)
        dd = 55
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv55 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv55)
        dd = 56
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv56 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv56)
        dd = 57
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv57 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv57)



        dd = 58
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv58 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv58)



        dd = 59
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv59 = torch.nn.Conv2d(i1024, 3*(num_classes + 5), kernel_size=(1, 1))
        aaaaaaa2(base_model.layers[i1], self.conv59)





        dd = 60
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
        self.conv60 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv60)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
        self.conv61 = Conv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv61)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
        self.conv62 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv62)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
        self.conv63 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv63)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
        self.conv64 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv64)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
        self.conv65 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv65)

        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
        self.conv66 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv66)



        dd = 67
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv67 = torch.nn.Conv2d(i512, 3*(num_classes + 5), kernel_size=(1, 1))
        aaaaaaa2(base_model.layers[i1], self.conv67)














        dd = 68
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
        self.conv68 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv68)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
        self.conv69 = Conv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv69)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
        self.conv70 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv70)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
        self.conv71 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv71)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
        self.conv72 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv72)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
        self.conv73 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv73)
        dd += 1
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
        self.conv74 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv74)



        dd = 75
        i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
        self.conv75 = torch.nn.Conv2d(i256, 3*(num_classes + 5), kernel_size=(1, 1))
        aaaaaaa2(base_model.layers[i1], self.conv75)

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




base_model = keras.models.load_model('aaaa_bgr.h5')
# keras.utils.vis_utils.plot_model(base_model, to_file='aaaaaaa.png', show_shapes=True)

net = Darknet(base_model, 80, initial_filters=32)
torch.save(net.state_dict(), 'aaaa_bgr.pt')






