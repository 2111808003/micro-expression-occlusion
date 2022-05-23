#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image, ImageFile
import cv2
import numpy as np
import random
import glob as glob
import math
import face_recognition

#  """the process for adding random mask to micro-expression sequence"""
# 1. read sequance from orignal dir;
# 2. detect landmarks of the first img;
# 3. add the random block on the first img, and remember the distance between the center point of the block and the landmark of nose;
# 4. detect other imgs landmarks, adding blocks according to the position of the first img.



def randommask_first_image(img_file, occ_path, ratio_orig):
    # Gets the face boundary
    # read landmarks
    file_name = str.split(img_file, '/')[-1]
    ratio = math.sqrt(ratio_orig)
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
    face_image_np = face_recognition.load_image_file(img_file)
    face_locations = face_recognition.face_locations(face_image_np, model='hog')
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
    _face_img = Image.fromarray(face_image_np)
    # _mask_img = Image.open(mask_path)
    found_face = False
    for face_landmark in face_landmarks:
        # check whether facial features meet requirement
        skip = False
        for facial_feature in KEY_FACIAL_FEATURES:
            if facial_feature not in face_landmark:
                skip = True
                break
        if skip:
            continue
        found_face = True
    if found_face == True:
        # find 4 points of the face
        chin = face_landmark['chin']
        cin_array = np.array(chin)
        left_eyebrow = face_landmark['left_eyebrow']
        right_eyebrow = face_landmark['right_eyebrow']
        temp_landm = np.array(chin + left_eyebrow + right_eyebrow)
        y_max=max(temp_landm[:,1])
        y_min=min(temp_landm[:,1])
        x_max=max(temp_landm[:,0])
        x_min=min(temp_landm[:,0])

        nose_bridge = face_landmark['nose_tip']
        nose_point = nose_bridge[0]
        nose_v = np.array(nose_point)
        # genrate block's width and height
        w = int((x_max - x_min) * ratio)
        h = int((y_max - y_min) * ratio)
        # generate the center point of the block randomly
        x_center = random.randrange(x_min + w // 2, x_max - w // 2)
        y_center = random.randrange(y_min + h // 2, y_max - h // 2)
        # compute the distance between the center point to nose
        x_dist = nose_v[0] - x_center
        y_dist = nose_v[1] - y_center
    else:
         raise RuntimeError('not found face, error!')

    """filling with the black"""
    mask = Image.new('RGBA', (int(w), int(h)), (0, 0, 0))
    pil_im = Image.open(img_file)
    place = (int(x_center - w // 2), int(y_center - h // 2), int(x_center + w // 2), int(y_center + h // 2))
    size = (int(int(x_center + w // 2) - int(x_center - w // 2)), int(int(y_center + h // 2) - int(y_center - h // 2)))
    pil_im_crop_resize = mask.resize(size)
    pil_im.paste(pil_im_crop_resize, place)
    pil_im.save(occ_path + '/' + file_name)
    return w, h, x_dist, y_dist, nose_v, found_face



def randommask(img_file, occ_path, ratio_orig, w, h, x_dist, y_dist, nose_v_in, found_face_in):
    file_name = str.split(img_file, '/')[-1]
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
    face_image_np = face_recognition.load_image_file(img_file)
    face_locations = face_recognition.face_locations(face_image_np, model='hog')
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
    _face_img = Image.fromarray(face_image_np)
    found_face = False
    for face_landmark in face_landmarks:
        # check whether facial features meet requirement
        skip = False
        for facial_feature in KEY_FACIAL_FEATURES:
            if facial_feature not in face_landmark:
                skip = True
                break
        if skip:
            continue
        found_face = True
    if found_face == False:
        nose_v = nose_v_in
    else:
        nose_bridge = face_landmark['nose_tip']
        nose_point = nose_bridge[0]
        nose_v = np.array(nose_point)

    x_center = nose_v[0] - x_dist
    y_center = nose_v[1] - y_dist

    """filling with the black"""
    # 获取色块，(0, 0, 0)即黑色
    """filling with the black"""
    # 获取色块，(0, 0, 0)即黑色
    mask = Image.new('RGBA', (int(w), int(h)), (0, 0, 0))
    pil_im = Image.open(img_file)
    place = (int(x_center - w // 2), int(y_center - h // 2), int(x_center + w // 2), int(y_center + h // 2))
    size = (int(int(x_center + w // 2) - int(x_center - w // 2)), int(int(y_center + h // 2) - int(y_center - h // 2)))
    pil_im_crop_resize = mask.resize(size)
    pil_im.paste(pil_im_crop_resize, place)
    pil_im.save(occ_path + '/' + file_name)


    
def main_random_mask(orig_path, occ_path, ratio_orig):
    img_sub_dir = glob.glob(orig_path+'/*/*')
#     print(img_sub_dir)
    for i in range(len(img_sub_dir)):
        sub_dir = img_sub_dir[i].split('/')[-2]
        print(sub_dir)
        occ_subdir = occ_path + '/' + sub_dir
        if os.path.exists(occ_subdir):
            pass
        else:
            os.makedirs(occ_subdir)
        occ_sub_file = occ_subdir + '/' +  img_sub_dir[i].split('/')[-1]
        if os.path.exists(occ_sub_file):
            pass
        else:
            os.mkdir(occ_sub_file)
        print(img_sub_dir[i])
        old_files = os.listdir(img_sub_dir[i])
        print(old_files)
        old_files.sort(key=lambda x: int(x[7:-4]))  
        img_file = img_sub_dir[i] + '/' + old_files[0]
        print(img_file)
        w, h, x_dist, y_dist, nose_v, found_face = randommask_first_image(img_file, occ_sub_file, ratio_orig)

        k = 1
        while k < len(old_files):
            img_file = img_sub_dir[i] + '/' + old_files[k]
            randommask(img_file, occ_sub_file, ratio_orig, w,h, x_dist, y_dist, nose_v, found_face)
            k = k+1


if __name__ == '__main__':

    orig_path = 'orig_imgs/'
    occ_path = 'orig_occ/ramdom-mask/'
    ratio_orig_list = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
    for i in range(len(ratio_orig_list)):
        ratio_orig = ratio_orig_list[i]
        if os.path.exists(occ_path):
            pass
        else:
            os.makedirs(occ_path)
        occ_dir = occ_path + '/' + 'occ' + str(int(ratio_orig * 100))
        if os.path.exists(occ_dir):
            pass
        else:
            os.makedirs(occ_dir)
        main_random_mask(orig_path, occ_dir, ratio_orig)



