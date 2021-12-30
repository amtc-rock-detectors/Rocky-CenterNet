from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from tools.rock_eval.rock_ellipse_coco_eval import RockCOCOeval
from tools.polygon_to_ellipse.labelme_to_coco import ellipse_to_bbox
from tools.polygon_to_ellipse.rock_pascal_voc_to_coco import Ellipse
import numpy as np
import json
import os

import torch.utils.data as data


class RockCOCO(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super().__init__()
        self.data_dir = os.path.join(opt.data_dir, 'rock_dataset' if len(opt.folder_name) == 0 
                                     else opt.folder_name)
        self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            '{}.json'.format(split))
        self.max_objs = 128
        self.class_name = ['__background__', 'rock']
        self._valid_ids = [1, 2]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing rock coco {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:

                    if np.any(np.isnan(bbox[:5])):
                        continue
                        
                    score = bbox[5]
                    bbox_out = list(map(self._to_float, bbox[0:5]))
                    bbox_out[4] = (bbox_out[4] * 180.) % 180.
                    ellipse = bbox_out
                    bbox_ellipse = ellipse_to_bbox(ellipse)
                    bbox_ellipse[2] -= bbox_ellipse[0]
                    bbox_ellipse[3] -= bbox_ellipse[1]
                    xc, yc, a, b, ang = ellipse
                    area = np.pi * a * b / 4
                    if area < self.opt.min_area or area > self.opt.max_area:
                        continue

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox_ellipse": bbox_ellipse,
                        "bbox": bbox_ellipse[:4],
                        "ellipse": ellipse,
                        "segmentation": Ellipse(*ellipse).get_polygon(32),
                        "area": area,
                        "score": self._to_float(score)
                    }
                    if len(bbox) > 6:
                        assert False, "Something wrong"
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'),
                  indent=4)

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)    
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        assert not (self.opt.use_gt_poly and self.opt.use_gt_bbox)
        coco_eval = RockCOCOeval(self.coco, coco_dets, self.opt.areaRng, 
                                 self.opt.use_gt_poly, self.opt.use_gt_bbox)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
