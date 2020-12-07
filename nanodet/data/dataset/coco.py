import os
import torch
import numpy as np
import cv2

from tqdm import tqdm
from pycocotools.coco import COCO
from .base import BaseDataset
from nanodet.util.file_operations import find_files_with_suffix, parse_xml


class MyDataset(BaseDataset):
    def __init__(self,
                 img_path,
                 ann_path,
                 input_size,
                 pipeline,
                 keep_ratio=True,
                 use_instance_mask=False,
                 use_seg_mask=False,
                 use_keypoint=False,
                 load_mosaic=False,
                 cache_labels=True,  # whether to cache labels
                 mode='train'):
        """
        :param img_path:
        :param ann_path:
        :param input_size:
        :param pipeline:
        :param keep_ratio:
        :param use_instance_mask:
        :param use_seg_mask:
        :param use_keypoint:
        :param load_mosaic:
        :param mode:
        """
        # Call parent class's init method
        super(MyDataset, self).__init__(img_path=img_path,
                                        ann_path=ann_path,
                                        input_size=input_size,
                                        pipeline=pipeline)

        if not (os.path.isdir(self.img_path) and os.path.isdir(self.ann_path)):
            print('[Err]: invalid image dir or label dir.')
            return

        self.cache_labels = cache_labels
        self.mode = mode
        print('\nDataset in {:s} mode.'.format(self.mode))

        # ---------- Counting image list and label_list
        img_list = []
        txt_list = []

        find_files_with_suffix(self.img_path, suffix='.jpg', f_list=img_list)
        find_files_with_suffix(self.ann_path, suffix='.xml', f_list=txt_list)

        # check and refine
        self.img_list = []  # image path list
        self.xml_list = []  # label path list
        self.img_info_list = []  # image info list

        for img_path in img_list:
            if not os.path.isfile(img_path):
                # print('[Logging]: {:s} do not exists.'.format(img_path))
                continue

            if self.mode == 'train':
                xml_path = img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
            elif self.mode == 'test':
                xml_path = img_path.replace('JPEGImages', '').replace('test', 'Annotations').replace('.jpg', '.xml')

            if not os.path.isfile(xml_path):
                # print('[Logging]: {:s} do not exists.'.format(txt_path))
                continue

            self.img_list.append(img_path)
            self.xml_list.append(xml_path)

        self.N = len(self.img_list)  # number of images(or labels)
        print('Total {:d} files of the current dataset.'.format(self.N))

        # ---------- Caching labels(can handling pure background image)
        if self.cache_labels:
            print('Caching labels...')

            # ----- init labels to zeros
            self.labels = [np.zeros((0, 5), dtype=np.float32)] * self.N

            # ----------
            # valid label(contains foreground) counting
            self.N = 0  # init to 0

            # traverse each image and label pair
            p_bar = tqdm(zip(self.img_list, self.xml_list), desc='Caching labels')
            for img_path, xml_path in p_bar:
                # parse xml into coco label format
                parse_results = parse_xml(xml_path)
                if parse_results is None:
                    print('[Warning]: empty label {:s}.'.format(xml_path))
                    continue

                label_obj_strs, (w, h) = parse_results
                label = self.label_str_format(label_obj_strs)

                if label.size != 0:
                    # --- filling image info list
                    img_info = dict()
                    img_info['height'] = w  # image width
                    img_info['width'] = h  # image height
                    img_info['file_name'] = os.path.split(img_path)[-1]
                    self.img_info_list.append(img_info)

                    # --- filling label
                    self.labels[self.N] = label  # only count non-empty label(and its corresponding image)

                    # --- filling image path
                    self.img_list[self.N] = img_path

                    # update non-empty label counting
                    self.N += 1
            # ----------

            print('Total {:d} non-empty label samples.'.format(self.N))

    def __len__(self):
        return self.N

    def label_str_format(self, label_objs_str):
        """
        :return:
        """
        label = np.zeros((len(label_objs_str), 5))
        for i, obj_str in enumerate(label_objs_str):
            label[i] = np.array([float(x) for x in obj_str.strip().split()])

        return label

    def bbox_format(self, bboxes, w, h):
        """
        :param bboxes: center_x, center_y, bbox_w, bbox_h(normalized by image size)~[0.0, 1.0]
        :param w: image width
        :param h: image height
        :return: bboxes(x1, y1, x2, y2 in pixel coordinates)
        """
        # ----- convert to pixel coordinates from normalized coordinates
        bboxes[:, 0] = bboxes[:, 0] * w  # center_x
        bboxes[:, 1] = bboxes[:, 1] * h  # center_y
        bboxes[:, 2] = bboxes[:, 2] * w  # bbox_w
        bboxes[:, 3] = bboxes[:, 3] * h  # bbox_h

        # ----- convert to x1, y1, x2, y2 format
        bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5  # x1
        bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5  # y1
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x2
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y2

        return bboxes

    def get_train_data(self, idx):
        # ---------- Read image and label
        # ----- read image
        img_path = self.img_list[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # H×W×C
        if img is None:
            print('[Warning]: empty image.')
            return None

        H, W, C = img.shape

        # ----- read label
        if not self.cache_labels:
            img_info = dict()
            img_info['height'] = H
            img_info['width'] = W
            img_info['file_name'] = os.path.split(img_path)[-1]

            # ----- read label from xml label file
            # init label to zeros
            label = [np.zeros((0, 5), dtype=np.float32)]

            # --- parse xml into coco label format
            parse_results = parse_xml(xml_path)
            if parse_results is None:
                print('[Warning]: empty label {:s}.'.format(xml_path))

            label_obj_strs, (w, h) = parse_results
            label = self.label_str_format(label_obj_strs)

            bboxes = label[:, 1:]
            class_labels = label[:, 0]

        else:
            # ----- read label from caching
            img_info = self.img_info_list[idx]
            img_info['height'] = H if img_info['height'] != H else img_info['height']
            img_info['width'] = W if img_info['width'] != W else img_info['width']

            # --- get label
            label = self.labels[idx]

            bboxes = label[:, 1:]
            class_labels = label[:, 0]
        # ----------

        # ---------- Prepare meta data
        meta = dict(img=img,
                    img_info=img_info,
                    gt_bboxes=bboxes,  # x1, y1, x2, y2(in pixel)
                    gt_labels=class_labels)
        # ----------

        # ---------- Warp and resize, color augment
        meta = self.pipeline(meta, self.input_size)
        # ----------

        # Numpy array to torch tensor and H×W×C to C×H×W
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))
        return meta

    def get_data_info(self, ann_path):
        return None

    def get_val_data(self, idx):
        return self.get_train_data(idx)


class CocoDataset(BaseDataset):
    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url': 'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []

        for ann in anns:
            if ann.get('ignore', False):
                continue

            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue

            # bounding box: x1, y1, x2, y2
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann['keypoints'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        annotation = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if self.use_instance_mask:
            annotation['masks'] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation['keypoints'] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation['keypoints'] = np.zeros((0, 51), dtype=np.float32)

        return annotation

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        # ----- Read image
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)  # H×W×C

        # ----- Get image annotation: bbox and class label
        ann = self.get_img_annotation(idx)

        # ----- Prepare meta data
        meta = dict(img=img,
                    img_info=img_info,
                    gt_bboxes=ann['bboxes'],  # x1, y1, x2, y2(in pixel)
                    gt_labels=ann['labels'])

        if self.use_instance_mask:
            meta['gt_masks'] = ann['masks']
        if self.use_keypoint:
            meta['gt_keypoints'] = ann['keypoints']

        # ----- Warp and resize, color augment
        meta = self.pipeline(meta, self.input_size)
        # -----

        # Numpy array to torch tensor and H×W×C to C×H×W
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))

        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)
