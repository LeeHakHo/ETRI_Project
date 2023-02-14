import copy
import cv2
import torch
import numpy as np
from PIL import Image
from scipy import ndimage as ndimg
from cfglib.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    vector_sin, get_sample_point

import time
from CRAFT_modules.imgproc import loadImage, resize_aspect_ratio, normalizeMeanVariance,cvt2HeatmapImg
from torch.autograd import Variable
from CRAFT_modules.craft_utils import adjustResultCoordinates, getDetBoxes
from CRAFT_modules.craft import CRAFT
from collections import OrderedDict
import torch.backends.cudnn as cudnn

def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image

        # Leehakho

def CRAFT_net(network, image, text_threshold, link_threshold, low_text, cuda, poly):

    t0 = time.time()
    #cv2.imwrite('/home/ohh/PycharmProject/TextBPN-main/output/temp/' + 'a.jpg', image)
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    #print(x.shape)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = network(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    #render_img = score_text.copy()
    #render_img = np.hstack((render_img, score_link))
    #ret_score_text = cvt2HeatmapImg(render_img)

    # print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    # inds.append(0)
    # inds = torch.where(input['ignore_tags'] > 0)
    # inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(image.device)

    return boxes, polys

def merge_bbox(bbox, img):
    # originImg = img
    # print(img.shape)
    img_w, img_h, img_c = img.shape
    mp = []
    mp.append(img_w)
    mp.append(0)
    mp.append(0)
    mp.append(img_h)
    i = 0
    # bbox merge
    for p0, p1, p2, p3 in bbox:
        i += 1
        if int(p0[0]) < mp[0]:
            mp[0] = int(p0[0])

        if int(p2[1]) > mp[1]:
            mp[1] = int(p2[1])

        if int(p2[0]) > mp[2]:
            mp[2] = int(p2[0])

        if int(p0[1]) < mp[3]:
            mp[3] = int(p0[1])
    return torch.FloatTensor([[[mp[0], mp[1]], [mp[2], mp[1]], [mp[2], mp[3]], [mp[0], mp[3]]]]).cuda()

class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def get_sample_point(self, size=None):
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [self.points.astype(np.int32)], color=(1,))
        control_points = get_sample_point(mask, cfg.num_points, cfg.approx_factor)
        return control_points

    def get_control_points(self, size=None):
        n_disk = cfg.num_control_points // 2 - 1
        sideline1 = split_edge_seqence(self.points, self.e1, n_disk)
        sideline2 = split_edge_seqence(self.points, self.e2, n_disk)[::-1]
        if sideline1[0][0] > sideline1[-1][0]:
            sideline1 = sideline1[::-1]
            sideline2 = sideline2[::-1]
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top = sideline2
            bot = sideline1
        else:
            top = sideline1
            bot = sideline2

        control_points = np.concatenate([np.array(top), np.array(bot[::-1])], axis=0).astype(np.float32)

        return control_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):
    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.min_text_size = 4
        self.jitter = 0.8
        self.th_b = 0.4

        # Leehakho
        #torch.multiprocessing.set_start_method('spawn', force=True)
        self.cuda = True
        self.net = CRAFT()  # initialize
        self.trained_model = 'CRAFT_weights/craft_mlt_25k.pth'
        # print('Loading weights from checkpoint (' + trained_model + ')')
        if self.cuda:
            self.net.load_state_dict(self.copyStateDict(torch.load(self.trained_model)))
        else:
            self.net.load_state_dict(self.copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
        self.net.eval()

    # Leehakho
    def copyStateDict(info, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

        # Leehakho

    #Leehakho
    def get_proposal_CRAFT(self, img):
        bboxes, polys = CRAFT_net(network=self.net, image=img, text_threshold=0.4, link_threshold=0.2, low_text=0.2, cuda=True,
                                       poly=True)  # 0.7 0.4 0.4 -> 0.3 0.2 0.2
        polys = merge_bbox(bboxes, img)
        ctrl_points = polys
        return ctrl_points

    @staticmethod
    def sigmoid_alpha(x, k):
        betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
        dm = max(np.max(x), 0.0001)
        res = (2 / (1 + np.exp(-x * k / dm)) - 1) * betak
        return np.maximum(0, res)

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """
        # cv2.drawContours(mask, [polygon.astype(np.int32)], -1, value, -1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))
        # rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0],mask.shape[1]))
        # mask[rr, cc] = value

    @staticmethod
    def generate_proposal_point(text_mask, num_points, approx_factor, jitter=0.0, distance=10.0):
        # get  proposal point in contours
        h, w = text_mask.shape[0:2]
        contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = approx_factor * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
        pts_num = approx.shape[0]
        e_index = [(i, (i + 1) % pts_num) for i in range(pts_num)]
        ctrl_points = split_edge_seqence(approx, e_index, num_points)
        ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

        if jitter > 0:
            x_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance*jitter
            y_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance*jitter
            ctrl_points[:, 0] += x_offset.astype(np.int32)
            ctrl_points[:, 1] += y_offset.astype(np.int32)
        ctrl_points[:, 0] = np.clip(ctrl_points[:, 0], 1, w - 2)
        ctrl_points[:, 1] = np.clip(ctrl_points[:, 1], 1, h - 2)
        return ctrl_points

    @staticmethod
    def compute_direction_field(inst_mask, h, w):
        _, labels = cv2.distanceTransformWithLabels(inst_mask, cv2.DIST_L2,
                                                    cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        # # compute the direction field
        index = np.copy(labels)
        index[inst_mask > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        y = nearCord[:, :, 0]
        x = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = y
        nearPixel[1, :, :] = x
        grid = np.indices(inst_mask.shape)
        grid = grid.astype(float)
        diff = nearPixel - grid

        return diff

    def make_text_region(self, img, polygons, origin_img):
        h, w = img.shape[0], img.shape[1]
        mask_zeros = np.zeros(img.shape[:2], np.uint8)

        train_mask = np.ones((h, w), np.uint8)
        tr_mask = np.zeros((h, w), np.uint8)
        weight_matrix = np.zeros((h, w), dtype=np.float)
        direction_field = np.zeros((2, h, w), dtype=np.float)
        distance_field = np.zeros((h, w), np.float)

        gt_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float)
        proposal_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float)
        ignore_tags = np.zeros((cfg.max_annotation,), dtype=np.int)
        #print(polygons)
        if polygons is None:
            return train_mask, tr_mask, \
                   distance_field, direction_field, \
                   weight_matrix, gt_points, proposal_points, ignore_tags
        for idx, polygon in enumerate(polygons):
            if idx >= cfg.max_annotation:
                break
            polygon.points[:, 0] = np.clip(polygon.points[:, 0], 1, w - 2)
            polygon.points[:, 1] = np.clip(polygon.points[:, 1], 1, h - 2)

            #gt_points[idx, :, :] = polygon.get_sample_point(size=(h, w))
            gt_points[idx, :, :] = polygon.points #Leehakho
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int)], color=(idx + 1,))

            inst_mask = mask_zeros.copy()
            cv2.fillPoly(inst_mask, [polygon.points.astype(np.int32)], color=(1,))
            dmp = ndimg.distance_transform_edt(inst_mask)  # distance transform

            if polygon.text == '#' or np.max(dmp) < self.min_text_size:
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
                ignore_tags[idx] = -1
            else:
                ignore_tags[idx] = 1

            # proposal_points[idx, :, :] = \
            #     self.generate_proposal_point(dmp / (np.max(dmp)+1e-9) >= self.th_b, cfg.num_points,
            #                                  cfg.approx_factor, jitter=self.jitter, distance=self.th_b * np.max(dmp))  #Leehakho

            points = self.get_proposal_CRAFT(origin_img)
            #print(points)
            proposal_points[idx, :, :] = points.cpu()
            #print(proposal_points[idx, :, :])
            distance_field[:, :] = np.maximum(distance_field[:, :], dmp / (np.max(dmp)+1e-9))

            weight_matrix[inst_mask > 0] = 1. / np.sqrt(inst_mask.sum())
            # weight_matrix[inst_mask > 0] = 1. / inst_mask.sum()
            diff = self.compute_direction_field(inst_mask, h, w)
            direction_field[:, inst_mask > 0] = diff[:, inst_mask > 0]

        # ### background ######
        weight_matrix[tr_mask == 0] = 1. / np.sqrt(np.sum(tr_mask == 0))
        # weight_matrix[tr_mask == 0] = 1. / np.sum(tr_mask == 0)
        # diff = self.compute_direction_field((tr_mask == 0).astype(np.uint8), h, w)
        # direction_field[:, tr_mask == 0] = diff[:, tr_mask == 0]

        train_mask = np.clip(train_mask, 0, 1)

        return train_mask, tr_mask, \
               distance_field, direction_field, \
               weight_matrix, gt_points, proposal_points, ignore_tags

    def get_training_data(self, image, polygons, image_id=None, image_path=None):
        np.random.seed()
        origin_img = image
        if self.transform:
            origin_img = ResizeSqr(origin_img,[224,224])
            image, polygons = self.transform(image, copy.copy(polygons))
        #cv2.imwrite('/home/ohh/PycharmProject/TextBPN-main/output/temp/' + 'a.jpg', image)
        train_mask, tr_mask, \
        distance_field, direction_field, \
        weight_matrix, gt_points, proposal_points, ignore_tags = self.make_text_region(image, polygons, origin_img)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        train_mask = torch.from_numpy(train_mask).bool()
        tr_mask = torch.from_numpy(tr_mask).int()
        weight_matrix = torch.from_numpy(weight_matrix).float()
        direction_field = torch.from_numpy(direction_field).float()
        distance_field = torch.from_numpy(distance_field).float()
        gt_points = torch.from_numpy(gt_points).float()
        proposal_points = torch.from_numpy(proposal_points).float()
        ignore_tags = torch.from_numpy(ignore_tags).int()

        return image, train_mask, tr_mask, distance_field, \
               direction_field, weight_matrix, gt_points, proposal_points, ignore_tags

    def get_test_data(self, image, polygons, image_id=None, image_path=None):
        H, W, _ = image.shape
        origin_img = image
        #polygons = list(TextInstance(points = np.array([[[196, 582],[1209, 591],1210, 754],[185, 749]]), orient ='c', text='undefined'))
        if self.transform:
            origin_img = ResizeSqr(origin_img,[224,224])
            image, polygons = self.transform(image, copy.copy(polygons))

        polygons = self.get_proposal_CRAFT(origin_img).cpu() #Leehakho
        points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
        length = np.zeros(cfg.max_annotation, dtype=int)
        label_tag = np.zeros(cfg.max_annotation, dtype=int)

        # if polygons is not None:
        #     for i, polygon in enumerate(polygons):
        #         pts = polygon.points
        #         points[i, :pts.shape[0]] = polygon.points
        #         length[i] = pts.shape[0]
        #         if polygon.text != '#':
        #             label_tag[i] = 1
        #         else:
        #             label_tag[i] = -1

        #Leehakho
        if polygons is not None:
            idx = 0
            for i, polygon in enumerate(polygons):
                pts = polygon
                points[i, :pts.shape[0]] = polygon
                length[i] = pts.shape[0]
                label_tag[i] = 1
                idx = i


        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'index': idx,
            'label_tag': label_tag,
            'Height': H,
            'Width': W
        }

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        return image, meta

    def __len__(self):
        raise NotImplementedError()

def ResizeSqr(image, size):
    h, w, _ = image.shape
    img_size_min = min(h, w)
    img_size_max = max(h, w)

    if img_size_min < size[0]:
        im_scale = float(size[0]) / float(img_size_min)  # expand min to size[0]
        if np.round(im_scale * img_size_max) > size[1]:  # expand max can't > size[1]
            im_scale = float(size[1]) / float(img_size_max)
    elif img_size_max > size[1]:
        im_scale = float(size[1]) / float(img_size_max)
    else:
        im_scale = 1.0

    new_h = int(int(h * im_scale / 32) * 32)
    new_w = int(int(w * im_scale / 32) * 32)
    image = cv2.resize(image, (new_w, new_h))
    scales = np.array([new_w / w, new_h / h])

    return image