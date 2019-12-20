# Keras-DIOU-YOLOv3
## 概述
Keras上700行代码复现YOLOv3！使用DIOU loss。支持将模型导出为pytorch模型！
请查看
[- **`diou_loss的论文`**](https://arxiv.org/pdf/1911.08287.pdf)
参考了3个仓库：
https://github.com/YunYang1994/tensorflow-yolov3
https://github.com/xiaochus/YOLOv3
https://github.com/qqwweee/keras-yolo3
这个仓库有很大一部分照搬了YunYang1994的代码（label的填写以及损失函数部分），这里致敬大佬！
后处理部分参考了xiaochus的代码。
keras复杂损失层参考了qqwweee的代码。
导出为pytorch模型、实现diou+ciou、以及其它一些部分为自己原创。

YunYang1994的仓库训练出的模型很优秀，为了达到同等优秀的效果，所以损失函数部分照搬了大佬仓库里的代码，但是有点不同的地方是将giou_loss改成了ciou_loss。根据我自己的训练结果，
发现使用同等超参数的条件下，使用ciou_loss训练比使用giou_loss训练能达到更高mAP。


## 文件下载
两个在coco上的预训练模型（yolo_bgr_mAP_46.h5和yolo_bgr_mAP_47.pt），一个为keras版，一个为pytorch版本。在release处下载。

coco2017数据集下载：
http://images.cocodataset.org/zips/train2017.zip 
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
http://images.cocodataset.org/zips/val2017.zip 
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
http://images.cocodataset.org/zips/test2017.zip 
http://images.cocodataset.org/annotations/image_info_test2017.zip



## 仓库文件介绍

```
train.py            训练yolov3，用的是ciou loss。
1_lambda2model.py   将训练模型中yolov3的所有部分提取出来。
2_keras2pytorch.py  将keras模型导出为pytorch模型。
demo_kr.py          用keras模型进行预测。对视频进行预测的话需要解除注释。
demo_pt.py          用pytorch模型进行预测。对视频进行预测的话需要解除注释。
evaluate_kr.py      对keras模型评估。跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。
evaluate_pt.py      对pytorch模型评估。跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。


annotation/  存放训练集、验证集的注解文件。
data/        存放数据集物品类别名称文件（一行一个类别名称），类别名称最好不要有空格、斜杠、反斜杠，不然后面计算mAP时会报错。
images/      用于测试的图片，放在子目录test/下。预测输出在子目录res/下。
mAP/         对模型评估时产生的中间临时文件。
model/       存放yolov3算法后处理的脚本。
videos/      用于测试的视频，放在子目录test/下。
xxxiou/      里面有giou、diou、ciou的代码，将train.py中bbox_ciou(boxes1, boxes2)函数替换掉就实现了更换损失函数来训练。
```

## 训练
使用train.py进行训练。train.py不支持命令行参数设置使用的数据集、超参数。
而是通过修改train.py源代码来进行更换数据集、更改超参数（减少冗余代码）。
1.如果你要使用自己的数据集训练，那么请修改
```
train_path = 'annotation/coco2017_train.txt'
val_path = 'annotation/coco2017_val.txt'
classes_path = 'data/coco_classes.txt'

```

注解文件的格式如下：
```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```
和YunYang1994的注解文件格式是完全一样的，这里再次致敬大佬！

2.本仓库有pattern=0、pattern=1、pattern=2这3种训练模式。
0-从头训练，1-读取model_body继续训练（包括解冻，但需要先运行1_lambda2model.py脚本取得model_body），2-读取coco预训练模型训练
你只需要修改pattern的值即可指定训练模式。
然后在这3种模式的if-else分支下，你再指定批大小batch_size、学习率lr等超参数。

3.如果你决定从头训练一个模型（即pattern=0），而且你的显卡显存比较小，比如说只有6G。
又或者说你想训练一个小模型，因为你的数据集比较小。
那么你可以设置initial_filters为一个比较小的值，比如说8。
initial_filters会影响到后面的卷积层的卷积核个数（除了最后面3个卷积层的卷积核个数不受影响）。
yolov3的initial_filters默认是32，你调小initial_filters会使得模型变小，运算量减少，适合在小数据集上训练。


## 评估
训练完成后，用1_lambda2model.py将训练模型中yolov3的所有部分提取出来。
如果你想把模型导出为pytorch模型，运行2_keras2pytorch.py脚本。
运行evaluate_kr.py对keras模型（1_lambda2model.py提取出来的模型）评估，跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。
运行evaluate_pt.py对pytorch模型评估，跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。


## 预测
运行demo_kr.py或者是demo_pt.py。

## 后话
扫一扫关注我们的公众号：
<p align="center">
    <img width="100%" src="https://github.com/miemie2013/Keras-DIOU-YOLOv3/blob/master/weixin/qrcode_for_gh_989f6358f007_258.jpg" style="max-width:100%;">
    </a>
</p>
或者直接搜索公众号：猿生物语
不定时推送一些技术文章哦！



