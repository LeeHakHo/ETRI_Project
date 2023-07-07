#-*- coding: utf-8 -*-
"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import cv2
import numpy as np
import torch.nn as nn
import torch
import torchvision.utils

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor, \
    SENet_FeatureExtractor, SENet_FeatureExtractor_large, vovNet_FeatureExtractor, vovNet_FPN_FeatureExtractor, \
    vovNet_1_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.language_classifier import language_classifier
from modules.prediction import Attention
from PIL import Image

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
#os.environ["CUDA_VISIBLE_DEVICES"]= "2,3,4,5,6,7"  # Set the GPU 2 to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'SENet':
            self.FeatureExtraction = SENet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'SENetL':
            self.FeatureExtraction = SENet_FeatureExtractor_large(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'vovNet':
            self.FeatureExtraction = vovNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'vovFPNNet':
            self.FeatureExtraction = vovNet_FPN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'vovNet1':
            self.FeatureExtraction = vovNet_1_FeatureExtractor(opt.input_channel, opt.output_channel)
            #self.FeatureExtraction = self.FeatureExtraction.to(device)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

        # if opt.lg == True:
        #     self.lg_classifer = language_classifier(self.FeatureExtraction_output, opt.hidden_size, opt.num_class)
        self.lgmix = opt.lgmix

    def forward(self, input, text, is_train=True, cf = False):

        if cf is True:
            """ Prediction stage """
            if self.stages['Pred'] == 'CTC':
                prediction = self.Prediction(input.contiguous())
            else:
                prediction = self.Prediction(input.contiguous(), text, is_train,
                                             batch_max_length=self.opt.batch_max_length)
            return prediction

        """ Transformation stage """
        #for i in range(1):
        #    t_input = input[i]
        #    torchvision.utils.save_image(t_input, "./result/tps_image_test/img_i_" + str(i) + ".jpg")
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        #    for i in range(1):
        #        t_input = input[i]
        #        torchvision.utils.save_image(t_input, "./result/tps_image_test/img_t_"+ str(i) + ".jpg")
        """ Feature extraction stage """
        #input = input.to('cpu')
        visual_feature = self.FeatureExtraction(input)
        #visual_feature, lg = self.FeatureExtraction(input)

        """ language classifier stage """
        #if not self.stages['language_classifier'] == "None":
        #    lg_output = self.lg_classifer(visual_feature)

        #print(visual_feature)
        if self.stages['Feat'] == 'vovNet':

            visual_feature = visual_feature['stage5']
            #print(visual_feature.size())
        if self.stages['Feat'] == 'vovFPNNet':
            #print(visual_feature.keys()) #p2,~p6
            visual_feature = visual_feature['p2']
            #visual_feature = visual_feature.to(input.device)
        #print(visual_feature.size()) #32 512 1 33
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        #print(visual_feature.device)
        visual_feature = visual_feature.to(input.device) #Leehakho
        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)


        output =[]
        output.append(prediction)
        output.append(contextual_feature)
        return output
