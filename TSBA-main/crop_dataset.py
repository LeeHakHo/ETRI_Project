#-*- coding: utf-8 -*-
import json
import os
import cv2
import argparse
import numpy as np


def crop(name,img_path,label_path,save_path,start,end,train=True):

    """
    Crop Images with bounding box in annotations
    ARGS:
        img_path : input folder path where starts imagePath
        label_path : list of label path
        save_path     : list of image path
        start : start of cropped img number for each folder
        end : end of cropped img number for each folder
        train : if true, Load/Save Image in Train , if False, Load/Save Image in Validation
    """
    os.makedirs(save_path + "train/",exist_ok=True)
    os.makedirs(save_path + "validation/",exist_ok=True)
    gt_train_file = open(save_path + 'gt_train.txt', 'a')
    gt_valid_file = open(save_path + 'gt_valid.txt', 'a')

    for i in range(start,end):
        try:
            #
            #label_path = label_path.replace("1.가로형간판", "03.세로형간판")
            #label_path = label_path.replace("n/", "/")
            print(label_path+name+'_'+ (str(i).zfill(6)) + '.json')
            #label_path = label_path[:-1]
            annotations = json.load(open(label_path + name+'_'+ (str(i).zfill(6)) + '.json'))

            #image = cv2.imread(img_path + name+ '_'+(str(i).zfill(6)) + '.jpg')

            img_array = np.fromfile(img_path + name+ '_'+(str(i).zfill(6)) + '.jpg', np.uint8)  # 컴퓨터가 읽을수 있게 넘파이로 변환
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 이미지를 읽어옴
            for idx, annotation in enumerate(annotations['annotations']):
                x, y, w, h = annotations['annotations'][idx]['bbox']

                text = annotations['annotations'][idx]['text']

                if "X" in text: continue
                if "x" in text: continue
                if x < 0: continue
                if y < 0: continue
                crop_img = image[y:y + h, x:x + w]
                crop_file_name = name+ '_'+ (str(i).zfill(6)) + '_{:02}.jpg'.format(idx + 1)
                print(crop_file_name)
                if os.path.isfile(save_path + "train/" + crop_file_name):
                    print("already image existed")
                    continue
                elif train == True:
                    cv2.imwrite(save_path + "train/" + crop_file_name, crop_img)
                    gt_train_file.write("train/{}\t{}\n".format(crop_file_name, text))
                else:
                    cv2.imwrite(save_path + "validation/" + crop_file_name, crop_img)
                    gt_valid_file.write("validation/{}\t{}\n".format(crop_file_name, text))
        except:
            try:
                #print(label_path)
                annotations = json.load(open(label_path + name + '_' + (str(i).zfill(6)) + '.json'))

                # image = cv2.imread(img_path + name+ '_'+(str(i).zfill(6)) + '.jpg')

                img_array = np.fromfile(img_path + name + '_' + (str(i).zfill(6)) + '.JPG',
                                        np.uint8)  # 컴퓨터가 읽을수 있게 넘파이로 변환
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 이미지를 읽어옴
                for idx, annotation in enumerate(annotations['annotations']):
                    x, y, w, h = annotations['annotations'][idx]['bbox']

                    text = annotations['annotations'][idx]['text']

                    if "X" in text: continue
                    if "x" in text: continue
                    if x < 0: continue
                    if y < 0: continue
                    crop_img = image[y:y + h, x:x + w]
                    crop_file_name = name + '_' + (str(i).zfill(6)) + '_{:02}.jpg'.format(idx + 1)
                    print(crop_file_name)
                    if os.path.isfile(save_path + "train/" + crop_file_name):
                        print("already image existed")
                        continue
                    elif train == True:
                        cv2.imwrite(save_path + "train/" + crop_file_name, crop_img)
                        gt_train_file.write("train/{}\t{}\n".format(crop_file_name, text))
                    else:
                        cv2.imwrite(save_path + "validation/" + crop_file_name, crop_img)
                        gt_valid_file.write("validation/{}\t{}\n".format(crop_file_name, text))
            except:
                try:
                    #print(label_path)
                    annotations = json.load(open(label_path + name + '_' + (str(i).zfill(6)) + '.json'))

                    # image = cv2.imread(img_path + name+ '_'+(str(i).zfill(6)) + '.jpg')

                    img_array = np.fromfile(img_path + name + '_' + (str(i).zfill(6)) + '.JPEG',
                                            np.uint8)  # 컴퓨터가 읽을수 있게 넘파이로 변환
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 이미지를 읽어옴
                    for idx, annotation in enumerate(annotations['annotations']):
                        x, y, w, h = annotations['annotations'][idx]['bbox']

                        text = annotations['annotations'][idx]['text']

                        if "X" in text: continue
                        if "x" in text: continue
                        if x < 0: continue
                        if y < 0: continue
                        crop_img = image[y:y + h, x:x + w]
                        crop_file_name = name + '_' + (str(i).zfill(6)) + '_{:02}.jpg'.format(idx + 1)
                        print(crop_file_name)
                        if os.path.isfile(save_path + "train/" + crop_file_name):
                            print("already image existed")
                            continue
                        elif train == True:
                            cv2.imwrite(save_path + "train/" + crop_file_name, crop_img)
                            gt_train_file.write("train/{}\t{}\n".format(crop_file_name, text))
                        else:
                            cv2.imwrite(save_path + "validation/" + crop_file_name, crop_img)
                            gt_valid_file.write("validation/{}\t{}\n".format(crop_file_name, text))
                except :
                    print('file not found' + ": " + img_path + name + '_' + (str(i).zfill(6)) + '.jpg')
                    continue
    gt_train_file.close()
    gt_valid_file.close()


def data_preprocessing(opt, train=False):
    """
    Make Image Path - Bounding box lists for cropping
     ARGS:
         opt has
         datset_path : Original dataset path ( AI HUB )
         save_path : save_path to save cropped image and gt file
         train : Decide Target Folder name is Training or Validation
     """
    root = opt.dataset_path
    save_path = opt.save_path
    if(train== True):
        root = root +'Training'
    else:
        root = root + 'Validation'
    folderlist = os.listdir(root + '/image/')
    folderlist.sort()
    print(folderlist)

    ser = '현수막'
    ser_n = 1
    ser_b = ''
    ser_b_n = 0
    for i, folder in enumerate(folderlist):

        img_list  = os.listdir(root+'/image/' + folder)
        img_list.sort()
        img_path = root + '/image/'+folder +'/'
        tmp_s = img_list[0].split('_')
        if (img_list[0][-1] == 'p'):
            tmp_s = img_list[1].split('_')
        if(tmp_s[0] == '간판'):

            start = int(tmp_s[2][:6])
            tmp_s = img_list[-1].split('_')
            end = int(tmp_s[2][:6])
            name = tmp_s[0] + '_' + tmp_s[1]
            label_root = root + '/label/1.간판/'
            if(tmp_s[1] != ser):
                ser = tmp_s[1]
                ser_n = ser_n + 1
                label_path = label_root +str(ser_n) +'.' + tmp_s[1]+'/'
                temp = os.listdir(label_path)
                for n in temp:
                    if( train ==True):
                        label_path = label_root +str(ser_n) +'.' + tmp_s[1]+'/' + tmp_s[1]+ n[-1] +'/'
            else:
                label_path = label_root + str(ser_n) + '.' + tmp_s[1] + '/'
                temp = os.listdir(label_path)
                for n in temp:
                    if( train ==True):
                        label_path = label_root + str(ser_n) + '.' + tmp_s[1] + '/' + tmp_s[1]+ n[-1] + '/'
        if(train == True):
            crop(name,img_path,label_path,save_path,start,end,True)
        else:
            crop(name,img_path,label_path,save_path,start,end,False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/ohh/dataset/korean_dataset/", help='image and annotations path')
    parser.add_argument('--save_path', default="/home/ohh/dataset/korean_dataset/croped/",
                        help='save_path to save cropped image and gt file')
    opt = parser.parse_args()
    #data_preprocessing(opt, True)
    data_preprocessing(opt, False)