#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import numpy as np
import shapely.geometry

import labelme

from tools.polygon_to_ellipse.poly_to_xml import poly_to_ellipse
from tools.polygon_to_ellipse.rock_pascal_voc_to_coco import Ellipse


try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def points2poly(points):
    ctr = 0
    if len(points) == 1:
        poly = shapely.geometry.Polygon(np.array([points[0]]).reshape(-1, 2).tolist())
    else:
        poly = cascaded_union([shapely.geometry.Polygon(np.array([point_seg]).reshape(-1, 2).tolist()) for point_seg in points])

    return poly

def poly_iou_c(points1, points2):
    poly1 = points2poly(points1).buffer(0.01)
    poly2 = points2poly(points2).buffer(0.01)
    inter = poly1.intersection(poly2).area
    uni = poly1.area + poly2.area - inter
    del poly1
    del poly2
    if uni > 0:
        return inter/uni
    else:
        return 0


def ellipse_to_bbox(ellipse):

    xc, yc, a, b, ang = ellipse

    ang_rad = (ang % 180.) * np.pi / 180.
    cos_2 = np.cos(ang_rad) ** 2
    sin_2 = 1 - cos_2
    a_2 = a ** 2
    b_2 = b ** 2

    e_w = np.sqrt(a_2 * cos_2 + b_2 * sin_2)
    e_h = np.sqrt(b_2 * cos_2 + a_2 * sin_2)

    return [xc - e_w / 2, yc - e_h / 2, xc + e_w / 2, yc + e_h / 2, ang % 180.]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i # starts with 0
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )
    class_name_to_id["hard_rock"] = class_name_to_id["rock"]

    out_ann_file = osp.join(args.output_dir, f"{osp.basename(osp.abspath(args.input_dir))}.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    box_iou = []
    ellipse_iou = []
    bbox_iou = []
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        with open(filename, 'r') as f:
            json_file = json.load(f)
        img_h = json_file['imageHeight']
        img_w = json_file['imageWidth']

        base = osp.splitext(osp.basename(filename))[0]

        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=json_file['imagePath'],
                height=img_h,
                width=img_w,
                date_captured=None,
                id=image_id + 1,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        ellipses = dict()
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            ellipse = shape.get("other_data", {}).get("ellipse", None)
            mask = labelme.utils.shape_to_mask(
                [img_h, img_w], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            else:
                points = np.asarray(points)
            
            points = np.clip(points, np.array([0, 0]), 
                             np.array([img_w, img_h]) - 1)
            points = points.flatten().tolist()

            segmentations[instance].append(points)
            ellipses[instance] = ellipse
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask_encoded = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask_encoded))
            points_np_array = np.array(segmentations[instance]).reshape(-1, 2)
            xmin, ymin = np.min(points_np_array, axis=0)
            xmax, ymax = np.max(points_np_array, axis=0)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            
            if ellipses[instance] is None:
                ellipse = poly_to_ellipse(segmentations[instance], tolerance=1e-4)
            else:
                ellipse = ellipses[instance]
            
            bbox_ellipse = ellipse_to_bbox(ellipse)
            # Ellipse polygon
            ellipse_points = Ellipse(*ellipse).get_polygon(32)
            ellipse_points = np.clip(np.array(ellipse_points).reshape(-1, 2),
                                     np.array([0, 0]), 
                                     np.array([img_w, img_h]) - 1).reshape(1, -1).tolist()
            ellipse_iou.append(poly_iou_c(ellipse_points, segmentations[instance]))
            # Bbox vs polygon
            bbox_iou.append(poly_iou_c([[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
                        segmentations[instance]))
            bbox_ellipse[2] -= bbox_ellipse[0]
            bbox_ellipse[3] -= bbox_ellipse[1]

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id + 1,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    ellipse=ellipse,
                    bbox=bbox,
                    bbox_ellipse=bbox_ellipse,
                    iscrowd=0,
                )
            )

    
    np.save("ellipse_iou.npy", np.array(ellipse_iou))
    np.save("bbox_iou.npy", np.array(bbox_iou))
    print("Polygon vs Ellipse Stats")
    print("Min IoU:", np.min(ellipse_iou))
    print("Max IoU:", np.max(ellipse_iou))
    print("Mean IoU:", np.mean(ellipse_iou))
    print("50th percentile IoU:", np.quantile(ellipse_iou, 0.50))
    print("25th percentile IoU:", np.quantile(ellipse_iou, 0.25))
    print("75th percentile IoU:", np.quantile(ellipse_iou, 0.75))
    print("Polygon vs BBox Stats")
    print("Min IoU:", np.min(bbox_iou))
    print("Max IoU:", np.max(bbox_iou))
    print("Mean IoU:", np.mean(bbox_iou))
    print("50th percentile IoU:", np.quantile(bbox_iou, 0.50))
    print("25th percentile IoU:", np.quantile(bbox_iou, 0.25))
    print("75th percentile IoU:", np.quantile(bbox_iou, 0.75))
    with open(out_ann_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
