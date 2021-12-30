"""
Thanks to OneDirection9/convert_pascalvoc2coco code heavily if not completely
based from there.
Convert Rock PASCAL VOC annotations to MSCOCO format and save to a json file.
The MSCOCO annotation has following structure:
{
    "images": [
        {
            "file_name": ,
            "height": ,
            "width": ,
            "id":
        },
        ...
    ],
    "type": "instances",
    "annotations": [
        {
            "segmentation": [],
            "area": ,
            "iscrowd": ,
            "image_id": ,
            "bbox": [],
            "category_id": ,
            "id": ,
            "ignore":
        },
        ...
    ],
    "categories": [
        {
            "supercategory": ,
            "id": ,
            "name":
        },
        ...
    ]
}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
from collections import OrderedDict
import math
import numpy as np
import json

import xmltodict

logger = logging.getLogger(__name__)


def bbox_to_ellipse(bbox):

    xmin, ymin, xmax, ymax, ang = bbox

    x_c = (xmin + xmax) / 2
    y_c = (ymin + ymax) / 2

    e_w_2 = np.square(xmax - xmin)
    e_h_2 = np.square(ymax - ymin)
    ang_rad = ang * np.pi / 180.
    cos_2 = np.square(np.cos(ang_rad))

    a = np.sqrt((cos_2 * e_w_2 + (cos_2 - 1) * e_h_2) / (2 * cos_2 - 1))
    b = np.sqrt((cos_2 * e_h_2 + (cos_2 - 1) * e_w_2) / (2 * cos_2 - 1))

    return [x_c, y_c, a, b, ang]


class Ellipse:

    def __init__(self, xc, yc, r_maj, r_min, theta):

        self.xc = xc
        self.yc = yc
        self.r_maj = r_maj / 2
        self.r_min = r_min / 2
        self.theta = (theta % 180.) * np.pi / 180. 

    def get_point(self, ang):

        x_theta = self.r_maj * math.cos(self.theta) * math.cos(ang) - \
            self.r_min * math.sin(self.theta) * math.sin(ang) + self.xc
        y_theta= self.r_maj * math.cos(ang) * math.sin(self.theta) + \
            self.r_min * math.sin(ang) * math.cos(self.theta) + self.yc

        return [x_theta, y_theta]

    def get_polygon(self, num_points: int=16):

        return [sum([self.get_point(ang) for ang in
                     np.linspace(0, 2 * math.pi, num_points, endpoint=False)], [])]


class ROCKVOC2COCO(object):
    """Converters that convert PASCAL VOC annotations to MSCOCO format."""

    def __init__(self):
        self.cat2id = {'rock': 1}

    def get_img_item(self, file_name, image_id, size):
        """Gets a image item."""
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, obj, image_id, ann_id):
        """Gets an annotation item."""
        xc = float(obj['bndbox']['xc'])
        yc = float(obj['bndbox']['yc'])
        a = float(obj['bndbox']['a'])
        b = float(obj['bndbox']['b'])
        ang = float(obj['bndbox']['angle'])
        ang_rad = (ang % 180.) * math.pi / 180.
        # cos_2 = math.cos(ang_rad) ** 2
        # sin_2 = 1 - cos_2
        # a_2 = a ** 2
        # b_2 = b ** 2

        # e_w = math.sqrt(a_2 * cos_2 + b_2 * sin_2)
        # e_h = math.sqrt(b_2 * cos_2 + a_2 * sin_2)

        annotation = OrderedDict()
        annotation['segmentation'] = Ellipse(xc, yc, a, b, ang).get_polygon(32)
        annotation['area'] = math.pi * a * b / 4
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = [xc - e_w / 2, yc - e_h / 2, e_w, e_h]
        annotation['bbox_ellipse'] = [xc - e_w / 2, yc - e_h / 2, e_w, e_h, ang_rad / np.pi * 180.]
        annotation['ellipse'] = [xc, yc, a, b, ang_rad / np.pi * 180.]
        annotation['category_id'] = self.cat2id[obj['name']]
        annotation['id'] = ann_id
        annotation['ignore'] = int(obj['difficult'])
        return annotation

    def get_cat_item(self, name, id):
        """Gets an category item."""
        category = OrderedDict()
        category['supercategory'] = 'none'
        category['id'] = id
        category['name'] = name
        return category

    def convert(self, path, split, save_file):
        """Converts ROCK PASCAL VOC annotations to MSCOCO format. """
        ann_dir = osp.join(path, split)

        name_list = []
        for x in os.listdir(osp.join(path, split)):
            name, ext = osp.splitext(x)
            if ext == ".png":
                name_list.append(name)

        images, annotations = [], []
        ann_id = 1
        image_ctr = 1
        for name in name_list:
            image_id = image_ctr

            xml_file = osp.join(ann_dir, name + '.xml')

            with open(xml_file, 'r') as f:
                ann_dict = xmltodict.parse(f.read(), force_list=('object',))

            # Add image item.
            image = self.get_img_item(name + '.png', image_id, ann_dict['annotation']['size'])
            images.append(image)

            if 'object' in ann_dict['annotation']:
                for obj in ann_dict['annotation']['object']:
                    # Add annotation item.
                    annotation = self.get_ann_item(obj, image_id, ann_id)
                    annotations.append(annotation)
                    ann_id += 1
            else:
                logger.warning('{} does not have any object'.format(name))

            image_ctr += 1

        categories = []
        for name, id in self.cat2id.items():
            # Add category item.
            category = self.get_cat_item(name, id)
            categories.append(category)

        ann = OrderedDict()
        ann['images'] = images
        ann['type'] = 'instances'
        ann['annotations'] = annotations
        ann['categories'] = categories

        logger.info('Saving annotations to {}'.format(save_file))
        with open(save_file, 'w') as f:
            json.dump(ann, f)


if __name__ == '__main__':
    converter = ROCKVOC2COCO()
    path = '../../../data/rock_mixed_dataset'
    split = 'train'
    save_file = f'{path}/{split}.json'
    converter.convert(path, split, save_file)
