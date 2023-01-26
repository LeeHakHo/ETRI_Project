#-*- coding: utf-8 -*-
""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
#--inputPath data_all/ --gtFile data_all/gtmix.txt --outputPath data_lmdb_all
import io

import fire
import os
import lmdb
import cv2

import numpy as np
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    inputPath1 = '/home/ohh/dataset/korean_dataset/croped/'
    gtFile1 = '/home/ohh/dataset/korean_dataset/croped/gt_train.txt'
    with open(gtFile1, 'r', encoding='utf-8') as data1:
        datalist1 = data1.readlines()

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        #print(imagePath)
        imagePath = os.path.join(inputPath, imagePath)

        imagePath1, label1 = datalist1[i].strip('\n').split('\t')
        #print(imagePath1)
        imagePath1 = os.path.join(inputPath1, imagePath1)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue

        if(cnt % 3 == 0):
            image = Image.open(imagePath)
            image1 = Image.open(imagePath1)
            # 사이즈 조정
            leftimg = image.resize((112, 224))
            rightimg = image1.resize((112, 224))
            if(cnt % 2 == 0):
                # 이미지 좌우 합치기
                dst = Image.new('RGB', (leftimg.width + leftimg.width, rightimg.height))
                dst.paste(leftimg, (0, 0))
                dst.paste(rightimg, (leftimg.width, (leftimg.height - rightimg.height) // 2))
                # image.save expects a file-like as a argument
                label = label + label1
            else:
                tmp = leftimg
                leftimg = rightimg
                rightimg = tmp
                # 이미지 좌우 합치기
                dst = Image.new('RGB', (leftimg.width + leftimg.width, rightimg.height))
                dst.paste(leftimg, (0, 0))
                dst.paste(rightimg, (leftimg.width, (leftimg.height - rightimg.height) // 2))
                # image.save expects a file-like as a argument
                label = label1 + label
            try:
                dst.save('/home/ohh/dataset/cat_samples/sample_' + label + '.jpg')
                with open('/home/ohh/dataset/cat_samples/sample_' + label +'.jpg', 'rb') as f:
                    imageBin = f.read()
                print('/home/ohh/dataset/cat_samples/sample_' + label +'.jpg' + "_" + label)
            except:
                print("error: " + '/home/ohh/dataset/cat_samples/sample_' + label +'.jpg')
        elif(cnt % 3 == 1):
            with open(imagePath, 'rb') as f:
                imageBin = f.read()
            print(imagePath+ "_" + label)
        else:
            with open(imagePath1, 'rb') as f:
                imageBin = f.read()
            label = label1
            print(imagePath1 + "_" + label1)

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        #print(label)
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)

