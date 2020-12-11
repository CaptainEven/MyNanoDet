# encoding=utf-8

import cv2
import os
import time
import torch
import argparse

from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.util import torch_utils
from nanodet.data.transform import Pipeline

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        """
        :param cfg:
        :param model_path:
        :param logger:
        :param device:
        """
        self.cfg = cfg
        self.device = device

        model = build_model(cfg.model)

        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        print('INFO: {:s} loaded.'.format(model_path))
        self.model = model.to(device).eval()

        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        """
        :param img:
        :return:
        """
        img_info = {}

        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width

        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)

        # pre-processing: normalize
        meta = self.pipeline(meta, self.cfg.data.val.input_size)

        # numpy array to torch tensor, H×W×C to 1×C×H×W, then put to device
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            results_dict = self.model.inference(meta)

        return meta, results_dict

    def visualize(self, img_path, res_dict, meta, class_names, score_thres, wait=0):
        """
        :param img_path:
        :param res_dict: key: cls_id, val: x1, y1, x2, y2, score
        :param meta:
        :param class_names:
        :param score_thres:
        :param wait:
        :return:
        """
        time1 = time.time()

        paths = os.path.split(img_path)
        result_dir = paths[0] + '/results'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        save_img_path = result_dir + '/' + paths[1]
        self.model.head.show_result(meta['raw_img'],
                                    res_dict,
                                    class_names,
                                    score_thres=score_thres,
                                    show=False,
                                    save_path=save_img_path)

        print('viz time: {:.3f}s'.format(time.time() - time1))


def get_image_list(path):
    img_path_list = []

    # for main_dir, sub_dir, file_name_list in os.walk(path):
    #     for filename in file_name_list:
    #         a_path = os.path.join(main_dir, filename)
    #         ext = os.path.splitext(a_path)[1]
    #         if ext in image_ext:
    #             image_names.append(a_path)

    img_list = [path + '/' + x for x in os.listdir(path) if x.endswith('.jpg')]
    for img_path in img_list:
        if not os.path.isfile(img_path):
            print('[Warning]: {:s} do not exists.'.format(img_path))
            continue

        img_path_list.append(img_path)

    return img_path_list


def run():
    args = parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # ----- load and parse config file
    load_config(cfg, args.config)

    # ----- set logger
    logger = Logger(-1, use_tensorboard=False)

    # ----- set device
    device = torch_utils.select_device(args.device, apex=False, batch_size=None)

    # ----- set predictor
    predictor = Predictor(cfg, args.model, logger, device=device)  # 'cuda:0'
    logger.log('Press "Esc", "q" or "Q" to exit.')

    if args.demo == 'image':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]

        files.sort()
        for img_path in files:
            meta, res_dict = predictor.inference(img_path)

            predictor.visualize(img_path, res_dict, meta, cfg.class_names, 0.35)

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
        while True:
            ret_val, frame = cap.read()

            # ----- inference
            meta, res_dict = predictor.inference(frame)
            # -----

            predictor.visualize(res_dict, meta, cfg.class_names, 0.35)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--demo',
                        type=str,
                        default='image',
                        help='demo type, eg. image, video and webcam')
    parser.add_argument('--config',
                        type=str,
                        default='../config/nanodet_mcmot_mbv2.yml',
                        help='model config file path')
    parser.add_argument('--model',
                        type=str,
                        default='/mnt/diskb/even/workspace/nanodet_mcmot_mbv2/epoch2_iter1000.pth',
                        help='model file path')
    parser.add_argument('--path',
                        default='../data/images',
                        help='path to images or video')
    parser.add_argument('--camid',
                        type=int,
                        default=0,
                        help='webcam demo camera id')
    parser.add_argument('--device',
                        type=str,
                        default='6',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    run()
