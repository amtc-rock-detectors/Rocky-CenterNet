# MODEL ZOO

### Common settings and notes

- The experiments are run with pytorch 1.4, CUDA 10.1, and CUDNN 7.1.
- The models can be downloaded directly from Google drive ([rockdet_coco_dla_2x](https://drive.google.com/file/d/15EkBUSIB_mDNhkXTHLzLJdrM_2azG1Am/view?usp=sharing) for detection with ellipses, or [ctdet_coco_dla_2x](https://drive.google.com/file/d/1t2OrQWuilNepO2LCcMH694Sc5-VhtyqO/view?usp=sharing) for detection with bounding boxes).
- Training for detection with bounding boxes will require weight initialization with CenterNet's pre-trained weights, which can be obtained in [Google Drive](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT).
