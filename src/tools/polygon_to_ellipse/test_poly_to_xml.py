
import argparse
import cv2
import glob
import json
import numpy as np
import os
import xmltodict

from utils.image import get_affine_transform, affine_transform_ellipse


def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Draw Polygons and their respective ellipses")

    parser.add_argument("path", type=str, default="", help="Sets the path where images, jsons, and xmls will be looked for")
    parser.add_argument("--test_train", action='store_true', default=False, help="Does augmentation operators as done in training")
    
    args = parser.parse_args()

    for path in glob.glob(os.path.join(args.path, "*.png")) + glob.glob(os.path.join(args.path, "*.jpg")):

        img = cv2.imread(path)

        try:        
            with open(os.path.splitext(path)[0] + ".json", 'r') as f:
                jf = json.load(f)

            overlay = np.zeros(img.shape, dtype=np.uint8)
            for arr in jf['shapes']:
                if len(arr['points']) > 0 and 'rock' in arr['label']:
                    pts = np.array(arr['points'], np.int32)
                    pts = pts.reshape(-1, 2)
                    overlay = cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                    xmin, ymin = pts.min(axis=0)
                    xmax, ymax = pts.max(axis=0)

            img = cv2.addWeighted(overlay, 0.3, img, 1 - 0.3, 1, img)

            for arr in jf['shapes']:
                if len(arr['points']) > 0 and 'rock' in arr['label']:
                    pts = np.array(arr['points'], np.int32)
                    pts = pts.reshape(-1, 1, 2)
                    img = cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                    pts = pts[:, 0, :]
                    xmin, ymin = pts.min(axis=0)
                    xmax, ymax = pts.max(axis=0)
                    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        except:
            pass

        with open(os.path.splitext(path)[0] + ".xml", 'r') as f:
            anndict = xmltodict.parse(f.read(), force_list=('object',))

        if 'object' in anndict['annotation']:
        
            if args.test_train:
                height, width = img.shape[0], img.shape[1]
                c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
                s = max(img.shape[0], img.shape[1]) * 1.0
                input_h, input_w = 512, 512
                flipped = False
                
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = _get_border(128, img.shape[1])
                h_border = _get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

                if np.random.random() < 0.5:
                    flipped = True
                    img = img[:, ::-1, :]
                    c[0] = width - c[0] - 1

                trans_input = get_affine_transform(
                    c, s, 0, [input_w, input_h])
                img = cv2.warpAffine(img, trans_input,
                                     (input_w, input_h),
                                     flags=cv2.INTER_LINEAR)
                                         
            for obj in anndict['annotation']['object']:
            
                xc = float(obj['bndbox']['xc'])
                yc = float(obj['bndbox']['yc'])
                a = float(obj['bndbox']['a'])
                b = float(obj['bndbox']['b'])
                ang = float(obj['bndbox']['angle'])
                
                if args.test_train:
                    
                    bbox = np.array([xc, yc, a, b, (ang % 180.) / 180.])
                    if flipped:
                        bbox[0] = width - bbox[0] - 1
                        bbox[4] = 1 - bbox[4]
                    bbox = affine_transform_ellipse(bbox, trans_input)
                    
                    xc, yc, a, b, ang_ = bbox
                    ang = 180. * ang_ 
                    
                img = cv2.ellipse(img, (int(xc), int(yc)), (int(a // 2), int(b // 2)), ang, 0, 360, (255, 0, 0), 2)

        print(path)
        cv2.imwrite("test.png", img)
        cv2.imshow(f'img: {path}', img)
        cv2.waitKey(0)
        cv2.destroyWindow(f'img: {path}')

