# Reading data back
import json
import numpy as np
from pprint import pprint

dataset_db = []
num_joints = 4

with open('E:/SuperB_Training_3W_20200928/Graviti_SuperB_Training_3W_20200928/vdet/FV0180V9_Label_20200730_120159_008.mf400_remap_4I_screenRGB888_0033326.json', 'r') as f:
    data = json.load(f)

    # Find all labels with shape "LShape" in one image
    for obj in data['objects']:
        if obj['shape'] != "LShape":
            continue

        # Sequence: su -> sl -> ul -> lr
        joints = np.zeros((num_joints, 2), dtype=float)
        joints_vis = np.ones([num_joints], dtype=np.int8)

        joints[0] = obj['su']
        joints[1] = obj['sl']
        joints[2] = obj['ul']
        joints[3] = obj['lr']

        dataset_db.append({
            'filename': "name",
            'joints': joints,
            'joints_vis': joints_vis,
        })

