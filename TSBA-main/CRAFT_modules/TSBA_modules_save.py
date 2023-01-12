#-*- coding: utf-8 -*-
#--exp_name /  --Transformation TPS  --SequenceModeling BiLSTM --Prediction Attn
import collections
import string
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import os

from torchvision import transforms
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate, CustomDataset
from model import Model
from torch.autograd import Variable
from PIL import Image
from bounding_box import bounding_box as bb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
cudnn.deterministic = True

def TSBA_result_save(opt, image, img_name):
    """
    Inference each model and create submission with max confidence scored preds

        pth_list : load model parameter files
        imgH : each models H parameter
        imgW : each models W parameter
        FeatureExtraction : each models Feature Extractor paramter
    """
    pth_list = [opt.model1, opt.model2, opt.model3]
    imgH= [224,224,224]
    imgW= [224,224,224]
    FeatureExtraction = ['SENet', 'SENet', 'SENetL']

    """
    Save confidence score and preds as list
    """
    confidence_score_list=[]
    model_preds_list = []
    img_name_list = []
    for i in range(len(pth_list)):
        """ model configuration """
        opt.imgH = imgH[i]
        opt.imgW = imgW[i]
        opt.FeatureExtraction = FeatureExtraction[i]
        print(i)
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % pth_list[i])
        model.load_state_dict(torch.load(pth_list[i], map_location=device))

        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

        demo_dataset = CustomDataset(image)
        os.makedirs(f"./demo/{opt.current_time}/{opt.exp_name}/demo/", exist_ok=True)
        log = open(f'./demo/{opt.current_time}/{opt.exp_name}/demo/log_dataset.txt', 'w')
        demo_loader = torch.utils.data.DataLoader(
            demo_dataset, batch_size=opt.batch_size,
            shuffle=False,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)
        print('-' * 80)

        # predict
        model.eval()

        current_model_confidence_score_list = []
        current_model_preds_list = []

        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index, preds_size)

                else:
                    print(image.shape, text_for_pred.shape)
                    preds = model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)
                    current_model_preds_list += preds_str

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                cnt = 0
                for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    if (i==0):
                        img_name_list.append(img_name.split('/')[-1])
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    except Exception as e:
                        print(e)
                        cnt += 1
                        continue
                    print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                    os.makedirs(f"./demo/{opt.current_time}/{opt.exp_name}/demo/result", exist_ok=True)
                    current_model_confidence_score_list.append(confidence_score)
            confidence_score_list.append(current_model_confidence_score_list)
            model_preds_list.append(current_model_preds_list)

    submit = open(f'./result/sample_submission.csv', 'w')
    r_s = open(f'./result/report_sample.csv', 'w')
    submit.write('img_path,text\n')
    r_s.write('img_path,text\n')
    for i in range(len(confidence_score_list[0])):
        max_confidence = -1
        max_idx = -1
        for j in range(len(confidence_score_list)):
            model_preds_list[j][i] = model_preds_list[j][i][:model_preds_list[j][i].find('[s]')]
            if confidence_score_list[j][i] >= max_confidence:
                max_confidence = confidence_score_list[j][i]
                max_idx = j

        print(model_preds_list[max_idx][i]+ ' ' + f'{max_confidence:0.4f}')
        submit.write('./test/' + img_name_list[i] + ',' + model_preds_list[max_idx][i] + '\n')
        r_s.write('./test/' + img_name_list[i] + ',' + model_preds_list[max_idx][i] + ','+f'{max_confidence:0.4f}'+'\n')


    submit.close()