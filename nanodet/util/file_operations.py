# encoding=utf-8

import os
import xml.etree.ElementTree as ET

target_types = ['car', 'car_front', 'car_rear',
                'bicycle', 'person', 'cyclist',
                'tricycle', 'motorcycle', 'non_interest_zone',
                'non_interest_zones']

classes = ['car',  # 0
           'bicycle',  # 1
           'person',  # 2
           'cyclist',  # 3
           'tricycle',  # 4
           'non_interest_zone']  # 5


def find_files_with_suffix(root, suffix, f_list):
    """
    递归的方式查找特定后缀文件
    """
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(suffix):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            find_files_with_suffix(f_path, suffix, f_list)


def bbox_to_xyxy(bbox):
    """
    :param bbox: x1, x2, y1, y2
    :return: x1, y1, x2, y2
    """
    x1, x2, y1, y2 = bbox
    return x1, y1, x2, y2


def bbox_to_xywh_zero_one(size, box):
    """
    输出center_x, center_y, bbox_w, bbox_h ~[0.0, 1.0]
    :param size:  image size(image width and image height)
    :param box:
    :return:
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_min = box[0]
    x_max = box[1]
    y_min = box[2]
    y_max = box[3]

    if x_min < 0:
        x_min = 0
    if x_max < 0 or x_min >= size[0]:
        return None

    if x_max >= size[0]:
        x_max = size[0] - 1
    if y_min < 0:
        y_min = 0
    if y_max < 0 or y_min >= size[1]:
        return None

    if y_max >= size[1]:
        y_max = size[1] - 1

    # bbox中心点坐标
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0

    # bbox宽高
    w = abs(x_max - x_min)
    h = abs(y_max - y_min)

    # bbox中心点坐标和宽高归一化到[0.0, 1.0]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    if w == 0 or h == 0:
        return None

    return (x, y, w, h)


def parse_xml(xml_path):
    """
    :param xml_path:
    :return:
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # print(root)

    mark_node = root.find('markNode')
    if mark_node is None:
        # print('[Warning]: markNode not found.')
        return

    # 该图片对应的labels
    label_obj_strs = []

    try:
        # 图片宽高
        w = int(root.find('width').text.strip())
        h = int(root.find('height').text.strip())
    except Exception as e:
        # print('[Warning]: invalid (w, h)')
        print(e)
        return

    for obj in mark_node.iter('object'):
        target_type = obj.find('targettype')
        cls_name = target_type.text
        if cls_name not in target_types:
            # print("=> " + cls_name + " is not in targetTypes list.")
            continue

        # classes_c5(5类别的特殊处理)
        if cls_name == 'car_front' or cls_name == 'car_rear':
            cls_name = 'car_fr'
        if cls_name == 'car':
            car_type = obj.find('cartype').text
            if car_type == 'motorcycle':
                cls_name = 'bicycle'
        if cls_name == "motorcycle":
            cls_name = "bicycle"
        if cls_name not in classes:
            # print("=> " + cls_name + " is not in class list.")
            continue
        if cls_name == 'non_interest_zone':
            # print('Non interest zone.')
            continue

        # 获取class_id
        cls_id = classes.index(cls_name)
        assert (0 <= cls_id < 5)

        # 获取bounding box
        xml_box = obj.find('bndbox')
        box = (float(xml_box.find('xmin').text),  # x1
               float(xml_box.find('xmax').text),  # x2
               float(xml_box.find('ymin').text),  # y1
               float(xml_box.find('ymax').text))  # y2

        # bounding box格式化: bbox([0.0, 1.0]): center_x, center_y, width, height
        # bbox = bbox_to_xywh_zero_one((w, h), box)
        bbox = bbox_to_xyxy(box)
        if bbox is None:
            # print('[Warning]: bbox err.')
            continue

        # 生成检测对象的标签行: class_id, bbox_center_x, box_center_y, bbox_width, bbox_height
        obj_str = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            cls_id,  # class_id
            bbox[0],  # x1
            bbox[1],  # y1
            bbox[2],  # x2
            bbox[3])  # y2
        label_obj_strs.append(obj_str)

    return label_obj_strs, (w, h)
