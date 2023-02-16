#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import re
import os
import numpy as np
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
from util.misc import norm2


class TD500Text(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = os.path.join(data_root, 'obj_train_onlyone' if is_training else 'Test/image')
        self.annotation_root = os.path.join(data_root,'obj_train_annotation' if is_training else 'Test/bbox')
        #self.image_root = os.path.join(data_root, 'TD500' if is_training else 'Test')
        #self.annotation_root = os.path.join(data_root, 'TD500' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)

        p = re.compile('.rar|.txt')
        self.image_list = [x for x in self.image_list if not p.findall(x)]
        p = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        self.annotation_list = ['{}'.format(p.sub("", img_name)) for img_name in self.image_list]

    @staticmethod
    def parse_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path+".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = line.split(',')
            #gt[7] = gt[7].replace('\n','')

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, map(float, gt[:8])))
            xx = [x1, x2, x3, x4]
            yy = [y1, y2, y3, y4]
            #print(xx, "/", yy)

            label = gt[-1].strip().replace("###", "#")
            pts = np.stack([xx, yy]).T.astype(np.int32)
            polygons.append(TextInstance(pts, 'c', label))


        return polygons

    def __getitem__(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        # Read image data
        image = pil_load_img(image_path)

        try:
            # Read annotation
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, "gt_" + annotation_id)
            polygons = self.parse_txt(annotation_path)

            if(len(polygons) > 1):
                polygons = None
        except:
            polygons = None
            
        if self.is_training:
            return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)
        else:
            polygons = None #Leehakho
            return self.get_test_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import os
    import cv2
    from util.augmentation import Augmentation
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(size=640, mean=means, std=stds) #Leehahko 640 -> 224

    trainset = TD500Text(
        data_root='/home/ohh/dataset/GCN_105',
        is_training=False,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask = trainset[idx]
        img, train_mask, tr_mask = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        #for i in range(tr_mask.shape[2]):
        #    cv2.imshow("tr_mask_{}".format(i),
        #               cav.heatmap(np.array(tr_mask[:, :, i] * 255 / np.max(tr_mask[:, :, i]), dtype=np.uint8)))

        #cv2.imshow('imgs', img)
        #cv2.waitKey(0)
