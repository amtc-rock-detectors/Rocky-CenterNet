__author__ = 'tsungyi'

import copy
from collections import defaultdict
import datetime
import numpy as np
import time

from pycocotools.cocoeval import COCOeval
from pycocotools import _mask

from tools.polygon_to_ellipse.rock_pascal_voc_to_coco import Ellipse, bbox_to_ellipse


def get_area(ann, img_w, img_h, areaType):
    area = 0
    
    if areaType == 'ellipse':
        area = float(_mask.area(_mask.frPyObjects(
                   Ellipse(*ann["ellipse"]).get_polygon(32), img_h, img_w)))
    elif areaType == 'bbox_ellipse':
        xmin, ymin, w, h, _ = ann['bbox_ellipse']
        xmax, ymax = xmin + w, ymin + h
        box_coords = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
        area = float(_mask.area(_mask.frPyObjects(box_coords, img_h, img_w)))
    elif areaType == 'bbox':
        xmin, ymin, w, h = ann['bbox']
        xmax, ymax = xmin + w, ymin + h
        box_coords = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
        area = float(_mask.area(_mask.frPyObjects(box_coords, img_h, img_w)))
    elif areaType == 'segmentation':
        area = np.sum(np.array(_mask.area(_mask.frPyObjects(
                   ann["segmentation"], img_h, img_w)), dtype=np.float32))
    else:
        raise ValueError("Invalid areaType, expected 'bbox' | 'bbox_ellipse'" + \
                         f"| 'ellipse' | 'segmentation', got {areaType}")
    
    return area
         


class RockCOCOeval(COCOeval):
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, areaRng=None, 
                 use_gt_poly=False, use_gt_bbox=False, 
                 areaType='ellipse'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        iouType = 'segm'
        self.use_gt_poly = use_gt_poly
        self.use_gt_bbox = use_gt_bbox
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType, areaRng=areaRng) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
        # self.params = Params(iouType=iouType, areaRng=areaRng)
        self.areaType = areaType

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toPolyFromEllipse(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                img_info = coco.imgs[ann['image_id']]
                img_w, img_h = img_info['width'], img_info['height']
                area = get_area(ann, img_w, img_h, self.areaType)
                ann['segmentation'] = Ellipse(*ann["ellipse"]).get_polygon(32)
                ann['area'] = area
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
                

        def _toPoly(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                img_info = coco.imgs[ann['image_id']]
                img_w, img_h = img_info['width'], img_info['height']
                area = get_area(ann, img_w, img_h, self.areaType)
                # ann.pop("ellipse", None)
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        def _toBbox(anns, coco, real_bbox=False):
            # modify ann['segmentation'] by reference
            for ann in anns:
                img_info = coco.imgs[ann['image_id']]
                img_w, img_h = img_info['width'], img_info['height']
                area = get_area(ann, img_w, img_h, self.areaType)          
                # ann.pop("ellipse", None)
                if not real_bbox:
                    xmin, ymin, w, h, _ = ann['bbox_ellipse']
                else:
                    xmin, ymin, w, h = ann['bbox']
                xmax, ymax = xmin + w, ymin + h
                ann['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
                ann['area'] = area
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask
        if self.use_gt_poly:
            _toPoly(gts, self.cocoGt)
            _toPoly(dts, self.cocoDt)
        elif self.use_gt_bbox:
            _toBbox(gts, self.cocoGt, real_bbox=True)
            _toBbox(dts, self.cocoDt, real_bbox=False)
        else:
            _toPolyFromEllipse(gts, self.cocoGt)
            _toPolyFromEllipse(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
        
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[5], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((34,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.25, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, iouThr=.25, areaRng='small', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, iouThr=.75, areaRng='small', maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[9] = _summarize(1, iouThr=.25, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(1, iouThr=.75, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[12] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[13] = _summarize(1, iouThr=.25, areaRng='large', maxDets=self.params.maxDets[2])
            stats[14] = _summarize(1, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[2])
            stats[15] = _summarize(1, iouThr=.75, areaRng='large', maxDets=self.params.maxDets[2])
            stats[17] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[18] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[19] = _summarize(0, iouThr=.25, maxDets=self.params.maxDets[2])
            stats[20] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[21] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[22] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[23] = _summarize(0, iouThr=.25, areaRng='small', maxDets=self.params.maxDets[2])
            stats[24] = _summarize(0, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[2])
            stats[25] = _summarize(0, iouThr=.75, areaRng='small', maxDets=self.params.maxDets[2])
            stats[26] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[27] = _summarize(0, iouThr=.25, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[28] = _summarize(0, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[29] = _summarize(0, iouThr=.75, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[30] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            stats[31] = _summarize(0, iouThr=.25, areaRng='large', maxDets=self.params.maxDets[2])
            stats[32] = _summarize(0, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[2])
            stats[33] = _summarize(0, iouThr=.75, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    
class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.25, 0.95, int(np.round((0.95 - .25) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm', areaRng=None):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        if areaRng is not None:
            try:
                self.areaRng = [areaRng[key] for key in self.areaRngLbl]
            except:
                pass
        # useSegm is deprecated
        self.useSegm = None

