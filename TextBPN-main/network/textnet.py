import torch
import torch.nn as nn
from network.layers.model_block import FPN
from cfglib.config import config as cfg
import numpy as np
from network.layers.snake import Snake
from network.layers.gcn import GCN
from network.layers.rnn import Rnn
from network.layers.gcn_rnn import GCN_RNN
#from network.layers.transformer import Transformer
#from network.layers.transformer_rnn import Transformer_RNN
import cv2
from util.misc import get_sample_point, fill_hole
from network.layers.gcn_utils import get_node_feature, \
    get_adj_mat, get_adj_ind, coord_embedding, normalize_adj

import time
from CRAFT_modules.imgproc import loadImage, resize_aspect_ratio, normalizeMeanVariance,cvt2HeatmapImg
from torch.autograd import Variable
from CRAFT_modules.craft_utils import adjustResultCoordinates, getDetBoxes
from CRAFT_modules.craft import CRAFT
from collections import OrderedDict
import torch.backends.cudnn as cudnn

def merge_bbox(bbox, img):
    # originImg = img
    batch, img_c, img_h, img_w = img.shape
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
    return torch.FloatTensor([[[mp[0], mp[1]],[mp[2],mp[1]],[mp[2],mp[3]],[mp[0],mp[3]]]]).cuda()

def CRAFT_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280 ,interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

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
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = cvt2HeatmapImg(render_img)

    #print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    #inds.append(0)
    #inds = torch.where(input['ignore_tags'] > 0)
    #inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(image.device)

    return boxes, polys

class Evolution(nn.Module):
    def __init__(self, node_num, adj_num, is_training=True, device=None, model="snake"):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.adj_num = adj_num
        self.device = device
        self.is_training = is_training
        self.clip_dis = 16

        self.iter = 3 #3-> 30
        if model == "gcn":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = GCN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "rnn":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Rnn(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "gcn_rnn":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = GCN_RNN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "transformer":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Transformer(36, 512, num_heads=8,
                                         dim_feedforward=2048, drop_rate=0.0, if_resi=True, block_nums=4)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "transformer_rnn":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Transformer_RNN(36, 512, num_heads=8,
                                             dim_feedforward=2048, drop_rate=0.1, if_resi=True, block_nums=4)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        else:
            self.adj = get_adj_ind(self.adj_num, self.node_num, self.device)
            for i in range(self.iter):
                evolve_gcn = Snake(state_dim=128, feature_dim=36, conv_type='dgrid')
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # self.net = CRAFT()  # initialize
        # self.cuda = True
        # self.trained_model = 'CRAFT_weights/craft_mlt_25k.pth'
        # print('Loading weights from checkpoint (' + self.trained_model + ')')
        # if self.cuda:
        #     self.net.load_state_dict(self.copyStateDict(torch.load(self.trained_model)))
        # else:
        #     self.net.load_state_dict(self.copyStateDict(torch.load(self.trained_model, map_location='cpu')))
        #
        # if self.cuda:
        #     self.net = self.net.cuda()
        #     self.net = torch.nn.DataParallel(self.net)
        #     cudnn.benchmark = False
        #
        # self.net.eval()

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):
        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)
        return init_polys, inds

    @staticmethod
    def get_boundary_proposal_eval(input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, :].detach().cpu().numpy()

        inds = []
        init_polys = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = (dis_pred / np.max(dis_pred)) > cfg.dis_threshold
            dis_mask = fill_hole(dis_mask)
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8)
            for idx in range(1, ret):
                text_mask = labels == idx
                if np.sum(text_mask) < 150 \
                        or cls_preds[bid][text_mask].mean() < cfg.cls_threshold:
                        # or dis_preds[bid][text_mask].mean() < cfg.dis_th:
                    continue
                inds.append([bid, 0])
                poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                init_polys.append(poly)
        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device).float()
        return init_polys, inds

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        i_poly = i_it_poly + torch.clamp(snake(node_feats, self.adj).permute(0, 2, 1), -self.clip_dis, self.clip_dis)
        if self.is_training:
            i_poly = torch.clamp(i_poly, 1, w-2)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 1, w - 2)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 1, h - 2)
        return i_poly

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

    # def forward(self, cnn_feature, input=None, seg_preds=None, switch="gt"):
    #     embed_feature = cnn_feature
    #     #print(embed_feature)
    #     #print(inds)
    #
    #     j = 0
    #     inds = torch.where(input['ignore_tags'] > 0)
    #     batch_img = input['img']
    #     for img in batch_img:
    #         bboxes, polys= CRAFT_net(self.net, img, text_threshold=0.1, link_threshold=0.1, low_text=0.1, cuda=True,
    #                                   poly=True)  # 0.7 0.4 0.4 -> 0.3 0.2 0.2
    #         polys = merge_bbox(bboxes, input['img'])
    #         init_polys = polys
    #         # init_polys = torch.FloatTensor([[[100, 100],[950,100],[950,250],[100,250]]]).cuda()
    #
    #         if init_polys.shape[0] == 0:
    #             return [init_polys for i in range(self.iter)], init_polys, inds
    #
    #         py_preds = []
    #         py_pred = init_polys
    #         for i in range(self.iter):
    #             evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
    #             py_pred = self.evolve_poly(evolve_gcn, embed_feature, py_pred, j) #inds[0])
    #             py_preds.append(py_pred)
    #         #print(init_polys.shape)#[13,10,2]
    #         #print(py_pred.shape)#[21,10,2]
    #         j += 1
    #
    #     return py_preds, init_polys, inds

    def forward(self, cnn_feature, input=None, seg_preds=None, switch="gt"):
        # b, h, w = cnn_feature.size(0), cnn_feature.size(2), cnn_feature.size(3)
        # embed_xy = coord_embedding(b, w, h, self.device)
        # embed_feature = torch.cat([cnn_feature, embed_xy], dim=1)
        embed_feature = cnn_feature
        if self.is_training:
            init_polys, inds = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
            # TODO sample fix number
        else:
            init_polys, inds = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter)], init_polys, inds
        #print(inds) #(tensor([0, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9]), tensor([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0]))
        py_preds = []
        py_pred = init_polys
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            py_pred = self.evolve_poly(evolve_gcn, embed_feature, py_pred, inds[0])
            py_preds.append(py_pred)
        return py_preds, init_polys, inds


class TextNet(nn.Module):

    def __init__(self, backbone='resnet', is_training=True):#vgg -> resnet
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, is_training = self.is_training)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )
        self.BPN = Evolution(cfg.num_points, adj_num=4,
                             is_training=is_training, device=cfg.device, model="gcn_rnn")

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, input_dict):
        output = {}
        b, c, h, w = input_dict["img"].shape
        if not self.is_training:
            image = torch.zeros((b, c, cfg.test_size[1], cfg.test_size[1]), dtype=torch.float32).to(cfg.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]
        else:
            image = input_dict["img"]

        #proposal 만드는 듯
        up1, up2, up3, up4, up5 = self.fpn(image)
        up1 = up1[:, :, :h, :w]
        preds = self.seg_head(up1)
        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)
        # fy_preds = torch.sigmoid(preds[:, 0:2, :, :])
        cnn_feats = torch.cat([up1, fy_preds], dim=1)

        #print(cnn_feats.shape) #12,16,360,360
        #print(preds.shape) #12,4,640,640
        #print(fy_preds.shape) #12,4 640, 640
        #GCN

        #print(input_dict['ignore_tags'])
        py_preds, init_polys, inds = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["init_polys"] = init_polys
        output["inds"] = inds

        return output
