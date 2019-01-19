# -*- coding: UTF-8 -*-
# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText.h5'

class GEN(object):
    def __init__(self):
        self.RV3 =  RendererV3(DATA_PATH,max_time=SECS_PER_IMG)

    def gen_img(self, IMG_DIR, DEPTH_PATH, SEG_PATH):
        img_lis = os.listdir(IMG_DIR)
        depth_db = h5py.File(DEPTH_PATH, 'r')
        seg_db = h5py.File(SEG_PATH, 'r')

        seg_db = seg_db['mask']
        
        for f in img_lis:
            try:
                #f= u'ant+hill_74.jpg'
                img = cv2.imread(IMG_DIR+f)
                depth_arr = depth_db[f]
                seg = seg_db[f]
                depth = depth_arr[1].T
            except:
                continue

            # print(seg)
            seg = np.array(seg).astype(np.float32)
            label, area = self.get_label_area(seg)

            img = cv2.resize(img, (seg.shape[1], seg.shape[0]))
            depth = cv2.resize(depth, (seg.shape[1], seg.shape[0]))

            # cv2.imshow('img', img)
            # cv2.imshow('seg', (seg * 10).astype(np.uint8))
            # cv2.imshow('depth', (depth * 20).astype(np.uint8))
            # cv2.waitKey(0)

            # print(img.shape)
            # print(depth.shape)
            res = self.RV3.render_text(img,depth,seg,area,label,ninstance=INSTANCE_PER_IMAGE)
            if len(res) > 0:
                new_img = res[0]['img']
                char_info = res[0]['charBB']
                char_info = char_info.astype(int)

                
                print(char_info.shape)
                poi_1 = (char_info[0][0][0], char_info[1][0][0])
                poi_2 = (char_info[0][2][0], char_info[1][3][0])

                print(poi_1)
                print(poi_2)
                # char_box = char_info[0].T
                # char_box = char_box[0].astype(int)
                # print(char_box)
                cv2.rectangle(new_img, poi_1, poi_2, (0,0,255), 2 )
                cv2.imshow('new_img', new_img)
                cv2.waitKey(0)
            else:
                continue

            break
            


    def get_label_area(self, seg):
        m = {}
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if (seg[i][j] in m.keys()):
                    m[seg[i][j]] = m[seg[i][j]] + 1
                else:
                    m[seg[i][j]] = 1

        label = m.keys()
        label = np.array(label)
        label = label.tolist()
        area = np.zeros(len(label))

        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                area[seg[i][j]] = area[seg[i][j]] + 1

        area = area.astype(int)

        label = np.array(label).astype(np.int64)
        area = np.array(area).astype(np.int64)

        return label, area


            

if __name__ == '__main__':
    gen = GEN()
    IMG_DIR = '/home/ffb/workspace/python-srf/SynTextData/bg_img/'
    DEPTH_PATH = '/home/ffb/workspace/python-srf/SynTextData/depth.h5'
    SEG_PATH = '/home/ffb/workspace/python-srf/SynTextData/seg.h5'

    gen.gen_img(IMG_DIR, DEPTH_PATH, SEG_PATH)