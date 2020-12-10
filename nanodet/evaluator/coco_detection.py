# encoding=utf-8

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from mAPEvaluate.TestmApDetect import test_tmp_mAP
import shutil
import json
import os
import copy


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class MyDetectionEvaluator(object):
    def __init__(self, dataset, txt_out_dir, num_classes=5):
        """
        :param dataset:
        :param txt_out_dir:
        :param num_classes:
        """
        self.dataset = dataset
        self.txt_out_dir = txt_out_dir
        self.num_classes = num_classes
        if not os.path.isdir(self.txt_out_dir):
            print('[Err]: invalid txt output directory.')
            os.makedirs(self.txt_out_dir)
        else:
            shutil.rmtree(self.txt_out_dir)
            os.makedirs(self.txt_out_dir)

    def format_det_outputs(self, dets, w, h):
        """
        :param dets: detection result input: x1, y1, x2, y2, score, cls_id
        :param w: image's original width
        :param h: image's original height
        :return: list of items: cls_id, conf_score, center_x, center_y,  bbox_w, bbox_h, [0, 1]
        """
        if dets is None:
            return None

        out_list = []
        for det in dets:
            x1, y1, x2, y2, score, cls_id = det
            center_x = (x1 + x2) * 0.5 / float(w)
            center_y = (y1 + y2) * 0.5 / float(h)
            bbox_w = (x2 - x1) / float(w)
            bbox_h = (y2 - y1) / float(h)
            out_list.append([int(cls_id), score, center_x, center_y, bbox_w, bbox_h])

        return out_list

    def evaluate(self, ret_dict):
        """
        :param ret_dict: ret_dict, key: img_id, val: dets_dict,
         dets_dict, key: cls_id, val: list of x1, y1, x2, y2, score
        :return:
        """
        N = len(ret_dict)
        print('Total {:d} images for evaluation.'.format(N))

        # ---------- output results.txt
        for i in range(N):  # process each image
            dets = []  # to store dets of the image
            dets_dict = ret_dict[i]  # detections of this image
            for cls_id in range(self.num_classes):  # process each object class
                cls_dets = dets_dict[cls_id]
                for det in cls_dets:  # process each detected object
                    x1, y1, x2, y2, score = det
                    det = x1, y1, x2, y2, score, cls_id
                    dets.append(det)

            # ----- get image info dict
            img_info = self.dataset.img_info_list[i]

            # ----- whether detection results exist or not
            if dets is None:
                print('\n[Warning]: non objects detected in {}, frame id {:d}\n' \
                      .format(os.path.split(path), fr_id))
                dets_list = []

            else:
                # ----- format output
                w, h = img_info['width'], img_info['height']  # image width and height
                dets_list = self.format_det_outputs(dets, w, h)

            # ----- write output
            img_name = img_info['file_name']
            txt_out_path = self.txt_out_dir + '/' + img_name.replace('.jpg', '.txt')
            with open(txt_out_path, 'w', encoding='utf-8') as f:
                f.write('class prob x y w h total=' + str(len(dets_list)) + '\n')  # write head
                for det in dets_list:
                    f.write('%d %f %f %f %f %f\n' % (det[0], det[1], det[2], det[3], det[4], det[5]))
            print('{} written'.format(txt_out_path))

        print('Total {:d} images tested.'.format(N))

        # ---------- run mAP evaluation
        test_tmp_mAP()
        return None


class CocoDetectionEvaluator:
    def __init__(self, dataset):
        """
        :param dataset:
        """
        assert hasattr(dataset, 'coco_api')

        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ['mAP', 'AP_50', 'AP_75', 'AP_small', 'AP_m', 'AP_l']

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score)
                    json_results.append(detection)
        return json_results

    def evaluate(self, ret_dict, save_dir, epoch, logger, rank=-1):
        """
        :param ret_dict:
        :param save_dir:
        :param epoch:
        :param logger:
        :param rank:
        :return:
        """
        results_json = self.results2json(ret_dict)
        json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
        json.dump(results_json, open(json_path, 'w'))

        coco_dets = self.coco_api.loadRes(json_path)

        coco_eval = COCOeval(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        aps = coco_eval.stats[:6]
        eval_results = {}

        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
            logger.scalar_summary('Val_coco_bbox/' + k, 'val', v, epoch)

        return eval_results
