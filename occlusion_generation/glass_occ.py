import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFile
import random
import glob as glob
__version__ = '0.3.0'


#  """the process for adding glass mask to micro-expression sequence"""
# 1. read sequance from orignal dir;
# 2. choose one glass for each sequece;
# 3. detect landmark of each img in each sequance, add glass to the face;


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glass_images')

def cli():
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    parser.add_argument('--pic_path',
                        default='orig_imgs/',
                        help='Picture path.')
    parser.add_argument('--new_pic_path',
                        default='orig_imgs_glass/',
                        help='Picture path.')
    parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    masks_path = glob.glob(IMAGE_DIR+'/*.png')
    # get new mask root path
    new_dir_path_root = args.new_pic_path
    if os.path.exists(new_dir_path_root) == False:
        os.makedirs(new_dir_path_root)

    # read imgs of same micro-expression sample
    # subject path
    sub_path = glob.glob(args.pic_path+'/*')
    for sub_dir in sub_path:
        print(sub_dir)
        # subject dir path
        new_sub_dir_path = new_dir_path_root + '/' + sub_dir.split('/')[-1]
        if os.path.exists(new_sub_dir_path) == False:
            os.makedirs(new_sub_dir_path)
        # img_dir
        img_dirs = glob.glob(sub_dir+'/*')
        for img_dir in img_dirs:
            print(img_dir)
            # random a mask
            mask_id = random.randint(0, 9)
            mask_path = masks_path[mask_id]
            print('+++++++++mask_path:', mask_path)

            new_img_dir_path = new_sub_dir_path + '/' + img_dir.split('/')[-1]
            if os.path.exists(new_img_dir_path) == False:
                os.makedirs(new_img_dir_path)
            imgs_path = glob.glob(img_dir+'/*.jpg')
            for img in imgs_path:
                print(img)
                new_face_path = new_img_dir_path +'/' + img.split('/')[-1]
                FaceMasker(img, new_face_path, mask_path, args.show, args.model).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, new_face_path, mask_path, show=False, model='hog'):
        self.face_path = face_path
        self.new_face_path = new_face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)

        if found_face:
            if self.show:
                self._face_img.show()

            # save
            self._save()
        else:
            print('Found no face.')

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[0]
        nose_v = np.array(nose_point)

        left_eyebrow = face_landmark['left_eyebrow']
        left_eyebrow_v = left_eyebrow[2]

        chin = face_landmark['chin']
        # chin_len = len(chin)
        chin_bottom_point = nose_bridge[1]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[0]
        chin_right_point = chin[16]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(left_eyebrow_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        path_splits = os.path.splitext(self.new_face_path)
        new_face_path = path_splits[0] + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    cli()
