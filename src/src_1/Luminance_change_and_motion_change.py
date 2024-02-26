import random
import os
import ssl
import cv2
import numpy as np
import imageio
from IPython import display
from urllib import request
import re
import tempfile
import math
from tqdm.notebook import tqdm
import copy
import shutil
from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
import mmcv
import warnings

warnings.simplefilter('ignore')



class between_two_image:

    def __init__(self,firstframe,secondframe,pix = 10,rate = 5,luminance = 15):
        config_file = r"pwcnet_ft_4x1_300k_sintel_final_384x768.py"
        checkpoint_file = r'pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
        
        self.rate = rate
        self.pix = pix
        self.luminance = luminance
        
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

        self.img1 = firstframe
        self.img2 = secondframe

        self.first_bool = False
        self.second_bool = False
        
        self.flo_out()
        self.remove_values_closeto_0()
        self.flow_magnitude_out()
        self.warp_second_frame_to_first()
    
    
    def __call__(self):

        self.with_movement_greater_than_3()
        self.intensity_levels()
        
        return self.first_bool , self.second_bool
        
    def flo_out(self):
        self.result_12 = inference_model(self.model, self.img1, self.img2)

        
    def remove_values_closeto_0(self):

        condition_12_1 = (self.result_12 < 0.01)
        condition_12_2 = (self.result_12 > -0.01)
        condition_12 = condition_12_1 & condition_12_2

        self.result_12[condition_12] = 0
        
        
    def flow_magnitude_out(self):
        flow_map_12_channel_0 = self.result_12[:,:,0]
        flow_map_12_channel_1 = self.result_12[:,:,1]
        self.flow_magnitude_12, _ = cv2.cartToPolar(flow_map_12_channel_0, flow_map_12_channel_1)
        self.convert_nan_to_0()
        
    def convert_nan_to_0(self):
        nans = np.isnan(self.flow_magnitude_12)
        if np.any(nans):
            nans = np.where(nans)
            self.flow_magnitude_12[nans] = 0.

            
    def warp_second_frame_to_first(self):
        flow_map = copy.copy(self.result_12)
        h,w = flow_map.shape[:2]
        flow_map[:,:,0] += np.arange(w)#変換後の座標を指定
        flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
        read_img2 = cv2.imread(self.img2)
        self.warped_prevImg = cv2.remap(read_img2, flow_map, None, cv2.INTER_LINEAR)
        
        
    def with_movement_greater_than_3(self):
        first_bool = False
        more_number = np.count_nonzero(self.flow_magnitude_12 > self.pix)#ハイパーパラメータ
        pixel_number = self.result_12.shape[0] * self.result_12.shape[1]
        if (more_number/pixel_number * 100) > self.rate:
            self.first_bool = True

    def intensity_levels(self):
        prevImg = cv2.imread(self.img1)

        img_diff = cv2.absdiff(self.warped_prevImg, prevImg)
        sum_img_diff = np.sum(img_diff,axis = 2)
        average_l1_distance = np.mean(sum_img_diff)
        if average_l1_distance < self.luminance:#ハイパーパラメータ
            self.second_bool = True