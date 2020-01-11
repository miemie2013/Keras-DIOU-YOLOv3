#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2019-12-20 14:29:47
#   Description : 将训练模型中yolov3的所有部分提取出来。
#                 需要修改'ep000020-loss24.876-val_loss51.209.h5'为你最后训练得到的文件名。
#
#================================================================
import keras
from train import decode, loss_layer


model = keras.models.load_model('ep000020-loss24.876-val_loss51.209.h5', custom_objects={'decode': decode, 'loss_layer': loss_layer, '<lambda>': keras.losses.mean_squared_error, })

# 训练好的模型 传入模型的图片是 bgr格式
# 传入模型的图片是否转换为rgb？本地评估的话选择False
# translate2rgb = True
translate2rgb = False

saved_model_name = 'aaaa'


name0, name1, name2, name3 = 'input_1', 'conv2d_59', 'conv2d_67', 'conv2d_75'
i0, i1, i2, i3 = 0, 0, 0, 0
for i in range(len(model.layers)):
    ly = model.layers[i]
    if ly.name == name0:
        i0 = i
    elif ly.name == name1:
        i1 = i
    elif ly.name == name2:
        i2 = i
    elif ly.name == name3:
        i3 = i


model2 = keras.models.Model(inputs=model.layers[i0].input, outputs=[model.layers[i1].output, model.layers[i2].output, model.layers[i3].output])


if translate2rgb:
    weights = model2.layers[1].get_weights()[0]
    h, w, c, k = weights.shape
    w2 = weights.copy()
    w2[:, :, 0:1, :] = weights[:, :, 2:3, :]
    w2[:, :, 2:3, :] = weights[:, :, 0:1, :]
    model2.layers[1].set_weights([w2])

# keras.utils.vis_utils.plot_model(model2, to_file='model2.png', show_shapes=True)
model2.compile(loss=[keras.losses.mean_squared_error, keras.losses.mean_squared_error, keras.losses.mean_squared_error], loss_weights=[1., 1., 1.], optimizer=keras.optimizers.Adam(lr=0.001))


if translate2rgb:
    model2.save(saved_model_name+'_rgb.h5')
else:
    model2.save(saved_model_name+'_bgr.h5')


