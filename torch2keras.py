#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 20:09
# @Author  : li PH
# @Project : modelfile_torch2tensorflow
# @File    : torch2keras

"""
    torch pth模型 转 keras h5模型
"""

# 主要使用库
from pytorch2keras import pytorch_to_keras
from tensorflow.keras import models
from Vgg_models.VggX import get_configs, VGGAutoEncoder
import tensorflow as tf
import torch.onnx


#   获取torch模型预训练权重文件
def load_weights(path="imagenet-vgg16.pth"):
    pretrained_dict = torch.load(path, map_location='cpu')
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict['state_dict'].items()}  # 使用的预训练权重文件层名对应的生成模型层名不一致,进行对应操作，若一致，可不进行此操作
    return pretrained_dict


#   生成对应torch预训练权重对应torch模型,并加载预训练权重
def load_torch_model(arch='vgg16', pretrained_dict=None):
    configs = get_configs(arch=arch)
    vgg16_ae = VGGAutoEncoder(configs)
    vgg16_ae.load_state_dict(pretrained_dict)
    return vgg16_ae


#   开始torch模型转keras模型
def torch2keras(torch_model):
    dummy_input = torch.rand(1, 3, 256, 256)
    torch_model.eval()
    keras_model = pytorch_to_keras(torch_model, dummy_input, (3, 256, 256), verbose=True)
    keras_model.summary()
    keras_model.save('imagenet-vgg16.h5')
    return keras_model


#   合法化模型层名
#   reference:https://nrasadi.medium.com/change-model-layer-name-in-tensorflow-keras-58771dd6bf1b
def replcae_layer_name(model, replce_str: str, custom_objects=None, save_path="Vgg16-imagenet.h5"):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary.
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''

    config = model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config['layers']:
        new_name = layer['name'].replace(replce_str, "")
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]

    config['name'] = config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)

    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())

    model = models.Model(new_model.input, new_model.output)
    model.save(save_path)
    return model


if __name__ == '__main__':
    pretrained_dict = load_weights()
    torch_vgg16 = load_torch_model(pretrained_dict=pretrained_dict)
    keras_vgg16 = torch2keras(torch_vgg16)
    #   若出现raise ValueError("'%s' is not a valid scope name" % name),
    #   ↑是因为pytorch_to_keras将torch模型转为keras模型时,会中转成onnx模型，onnx模型对每层的命名不符合keras模型的命名规则
    #   进入ops.py文件,找到条件语句if not _VALID_SCOPE_NAME_REGEX.match(name),将内容暂时替换为pass,这样会暂时让模型转换合法化

    # 对应keras命名规则, 合法化层名
    new_keras_vgg16 = replcae_layer_name(keras_vgg16, replce_str="onnx::")
    print(models.summary())
    #合法化模型层名后,进入ops.py文件,找到条件语句if not _VALID_SCOPE_NAME_REGEX.match(name),将内容换回

