# ------------------------------------------------------------------------------
# Dataset class for 3D cars.
# Jingyi Wang (Jingyi.Wang@imotion.ai)
# Oct. 28, 2021
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import logging
import os
import random
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

out_img_dir = "F:\\img"
out_anno_dir = "F:\\anno"

class CarJointsDataset(Dataset):
    def __init__(self, cfg, transform=None):
        self.num_joints = 4
        self.num_edges = 2
        self.num_out_channels = 6

        self.img_dir = "F:\\img"
        self.anno_dir = "F:\\anno"

        self.image_raw_size = np.array([1664, 512])     # [Width, Height]
        self.image_size = cfg.MODEL.IMAGE_SIZE          # [Width, Height]
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE    # [Width, Height]
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        self.data_db = []

        file_cnt = 0

        # Find all annotate in dir with json format
        for filename in os.listdir(self.anno_dir):
            if not filename.endswith(".json"):
                continue

            file_cnt += 1
            if file_cnt % 500 == 0:
                print("{} files are processed!".format(file_cnt))

            file_saved_to_list_tag = False
            # filename_list = []

            with open(os.path.join(self.anno_dir, filename), 'r') as json_file:
                data = json.load(json_file)

                # Find all labels with shape "LShape" in one image
                for obj in data['objects']:
                    if obj['shape'] != "LShape":
                        continue

                    # Sequence: su -> sl -> ul -> lr
                    joints = np.zeros((self.num_joints, 2), dtype=float)
                    joints_vis = np.ones([self.num_joints], dtype=np.int8)

                    joints[0] = obj['su']
                    joints[1] = obj['sl']
                    joints[2] = obj['ul']
                    joints[3] = obj['lr']

                    # Coordinate of labels less than zero
                    if (joints < 0).sum() > 0:
                        continue
                    # Coordinate of labels out of the image
                    if (joints[:, 0] >= self.image_raw_size[0]).sum() > 0:
                        continue
                    if (joints[:, 1] >= self.image_raw_size[1]).sum() > 0:
                        continue

                    # Height of the car less than 50, too small
                    if joints[3][1] - joints[2][1] < 50:
                        continue

                    # Append the found object to dataset
                    self.data_db.append({
                        'filename': os.path.splitext(filename)[0],
                        'joints': joints,
                        'joints_vis': joints_vis,
                    })

                    # Save filename to list
                    # if not file_saved_to_list_tag:
                    #     file_saved_to_list_tag = True
                    #     # filename_list.append(filename)
                    #     os.system("copy {} {} /n".format(os.path.join(self.anno_dir, filename), out_anno_dir))
                    #     os.system("copy {} {} /n".format(os.path.join(self.img_dir, os.path.splitext(filename)[0] + ".png"),
                    #                                   out_img_dir))
                    #     print("copy {} {}".format(os.path.join(self.anno_dir, filename), out_anno_dir))

                    if len(self.data_db) % 500 == 0:
                        print("{} boxes are loaded!".format(len(self.data_db)))

    def __len__(self,):
        return len(self.data_db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.data_db[idx])

        image_fullname = os.path.join(self.img_dir, db_rec['filename'] + ".png")

        image_numpy = cv2.imread(
            image_fullname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if image_numpy is None:
            logger.error('=> fail to read {}'.format(image_fullname))
            raise ValueError('Fail to read {}'.format(image_fullname))

        joints = db_rec['joints']
        joints_vis = db_rec['joints_vis']

        # Find the ROI region
        # Add 16 pixels for each edge
        left_most = int(max(joints[:, 0].min() - 16, 0))
        right_most = int(min(joints[:, 0].max() + 16, self.image_raw_size[0] - 1))
        width_roi = right_most - left_most

        top_most = int(max(joints[:, 1].min() - 16, 0))
        bottom_most = int(min(joints[:, 1].max() + 16, self.image_raw_size[1] - 1))
        height_roi = bottom_most - top_most

        # Crop and resize the input image
        image_numpy = image_numpy[top_most:bottom_most, left_most:right_most]
        image_numpy = cv2.resize(image_numpy, self.image_size)

        # Apply the same transform to the coordinates of the ground truth
        # Translation
        joints[:, 0] -= left_most
        joints[:, 1] -= top_most
        # Scale
        joints[:, 0] *= self.image_size[0] / width_roi
        joints[:, 1] *= self.image_size[1] / height_roi

        # Apply the transform of magnitude
        if self.transform:
            image_numpy = self.transform(image_numpy)

        heatmap = np.zeros((self.num_out_channels,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        # Generate the target heatmap from the coordinates of the ground truth
        target, target_weight = self.generate_target(joints, joints_vis)
        heatmap[0:self.num_joints] = target

        # Generate target edge heatmap
        bottom_left = np.array([joints[2][0], joints[3][1]])

        # Left or right target edge
        if joints[1][0] < joints[2][0]:
            heatmap[self.num_joints] = self.generate_target_edge(joints[1], bottom_left)
        elif joints[1][0] > joints[3][0]:
            heatmap[self.num_joints] = self.generate_target_edge(joints[1], joints[3])

        # Front or back target edge
        heatmap[self.num_joints + 1] = self.generate_target_edge(bottom_left, joints[3])

        heatmap = torch.from_numpy(heatmap)
        target_weight = torch.from_numpy(target_weight)

        return image_numpy, heatmap



    def generate_target_edge(self, start, stop):

        try:

            target_edge = np.zeros([self.heatmap_size[1], self.heatmap_size[0]], dtype=float)

            normal_distrib = [1, 0.7, 0.4, 0.2, 0.1, 0.05]
            feat_stride = self.image_size / self.heatmap_size

            start = np.floor(start / feat_stride).astype(int)
            stop = np.floor(stop / feat_stride).astype(int)

            start = np.minimum(start, self.heatmap_size - 1)
            stop = np.minimum(stop, self.heatmap_size - 1)

            # Using DDA algorithm to generate the line and push back to the queue
            dx = stop[0] - start[0]
            dy = stop[1] - start[1]
            k = dy / dx

            queue = []

            if abs(k) <= 1:
                if start[0] > stop[0]:
                    tmp = start
                    start = stop
                    stop = tmp

                for cx in range(start[0], stop[0]):
                    cy = round(start[1] + k * (cx - start[0]))
                    queue.append((cx, cy, 0))
                    target_edge[cy][cx] = 1
            else:
                if start[1] > stop[1]:
                    tmp = start
                    start = stop
                    stop = tmp

                for cy in range(start[1], stop[1]):
                    cx = round(start[0] + (1 / k) * (cy - start[1]))
                    queue.append((cx, cy, 0))
                    target_edge[cy][cx] = 1

            queue.append((stop[0], stop[1], 0))
            target_edge[stop[1]][stop[0]] = 1

            # u r d l
            dx = [0, 1, 0, -1]
            dy = [-1, 0, 1, 0]

            while len(queue) > 0:
                cx, cy, depth = queue.pop(0)
                for i in range(4):
                    cx += dx[i]
                    cy += dy[i]

                    if cx >= 0 and cy >= 0 and cx < self.heatmap_size[0] and cy < self.heatmap_size[1]:
                        if target_edge[cy][cx] == 0:
                            if depth < 5:
                                queue.append((cx, cy, depth + 1))
                                target_edge[cy][cx] = normal_distrib[depth + 1]

                    cx -= dx[i]
                    cy -= dy[i]

            return target_edge

        except IndexError:
            print("?????????")


    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:]

        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
