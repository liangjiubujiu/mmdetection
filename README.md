

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:
- [x] ResNet
- [x] ResNeXt
- [x] VGG
- [x] HRNet
- [x] RegNet
- [x] Res2Net
- [x] ResNeSt

Supported methods:
- [x] [RPN](configs/rpn)
- [x] [Fast R-CNN](configs/fast_rcnn)
- [x] [Faster R-CNN](configs/faster_rcnn)
- [x] [Mask R-CNN](configs/mask_rcnn)
- [x] [Cascade R-CNN](configs/cascade_rcnn)
- [x] [Cascade Mask R-CNN](configs/cascade_rcnn)
- [x] [SSD](configs/ssd)
- [x] [RetinaNet](configs/retinanet)
- [x] [GHM](configs/ghm)
- [x] [Mask Scoring R-CNN](configs/ms_rcnn)
- [x] [Double-Head R-CNN](configs/double_heads)
- [x] [Hybrid Task Cascade](configs/htc)
- [x] [Libra R-CNN](configs/libra_rcnn)
- [x] [Guided Anchoring](configs/guided_anchoring)
- [x] [FCOS](configs/fcos)
- [x] [RepPoints](configs/reppoints)
- [x] [Foveabox](configs/foveabox)
- [x] [FreeAnchor](configs/free_anchor)
- [x] [NAS-FPN](configs/nas_fpn)
- [x] [ATSS](configs/atss)
- [x] [FSAF](configs/fsaf)
- [x] [PAFPN](configs/pafpn)
- [x] [Dynamic R-CNN](configs/dynamic_rcnn)
- [x] [PointRend](configs/point_rend)
- [x] [CARAFE](configs/carafe/README.md)
- [x] [DCNv2](configs/dcn/README.md)
- [x] [Group Normalization](configs/gn/README.md)
- [x] [Weight Standardization](configs/gn+ws/README.md)
- [x] [OHEM](configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py)
- [x] [Soft-NMS](configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py)
- [x] [Generalized Attention](configs/empirical_attention/README.md)
- [x] [GCNet](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training](configs/fp16/README.md)
- [x] [InstaBoost](configs/instaboost/README.md)
- [x] [GRoIE](configs/groie/README.md)
- [x] [DetectoRS](configs/detectors/README.md)
- [x] [Generalized Focal Loss](configs/gfl/README.md)
- [x] [CornerNet](configs/cornernet/README.md)
- [x] [Side-Aware Boundary Localization](configs/sabl/README.md)
- [x] [YOLOv3](configs/yolo/README.md)
- [x] [PAA](configs/paa/README.md)
- [x] [YOLACT](configs/yolact/README.md)
- [x] [CentripetalNet](configs/centripetalnet/README.md)
- [x] [VFNet](configs/vfnet/README.md)

Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Traning your own dataset
0. prepare your dataset. boshi/boshi/yaqi/tooth
```
tooth
  |_train
    |_**********.png
    ...
    |_annotation_train.json
  |_val
    |_*********.png
    ...
    |_annotation_val.json
```
1. config configs/__base__/coco_instance.py
```
2      data_root = 'tooth/'

33     train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotation_coco.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotation_coco.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
```
2. configs/__base__/models/maskrcc_r50_fpn.py
```
num_classes=3# not include background
```
3. configs/__base__/schedules/schedule_1x.py
```
optimizer = dict(type='SGD', lr=0.006, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 500
```
4. configs/default_run_time.py, set saved interval.
```
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
```
5. mmdet/core/class_names.py
```
67 def coco_classes():
      # return [
      #     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
      #     'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
      #     'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
      #     'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
      #     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
      #     'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
      #     'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
      #     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
      #     'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
      #     'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
      #     'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
      #     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      #     'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
      # ]

      return ['R','G','B']
```
6. mmdet/datasets/coco.py
```
29 class CocoDataset(CustomDataset):

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    CLASSES=('R','G','B')
```
7. trainning your maskrcnn.

7.1 create  configs/tooth and copy .py from config/maskrcnn.
```
tooth  
  |_mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_tooth.py
  |_mask_rcnn_r50_fpn_1x_coco.py
```
7.2 run
```
python /tools/train.py  configs/tooth/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_tooth.py
```
8. testing the trained maskrcnn.
```
python /demo/mydemo.py configs/tooth/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_tooth.py    work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_tooth/epoch_500.pth   --show
```
9.change other parameters in model.
For example, add data augmentation alub in train_pipeline.
9.1 change config file in work_dirs/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco.py, rename work_dirs for now model.
```
190 albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=180,
        interpolation=1,
        p=0.5)
]
212 dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=180,
                interpolation=1,
                p=0.5)
        ],
336 work_dir = './work_dirs/cascade_rcnn_r50_sac_1x_coco_albu'
```
9.2 Train the new model with the modified config in 9.1
```
python tools/train.py work_dirs/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco.py
```
9.3 change demo/mydemo.py
```
51     parser.add_argument('--config', default='work_dirs/cascade_rcnn_r50_sac_1x_coco_albu/cascade_rcnn_r50_sac_1x_coco.py',help='test config file path')
    parser.add_argument('--checkpoint', default='work_dirs/cascade_rcnn_r50_sac_1x_coco_albu/epoch_60.pth',help='checkpoint file')
    parser.add_argument('--out', default='result/cascade_rcnn_r50_sac_1x_coco_albu.pkl', help='output result file in pickle format')
    
```
9.4 Test the new model
```
python demo/mydemo.py
```
9.5 Find the new test results in 'result/cascade_rcnn_r50_sac_1x_coco_albu.pkl', and change the config file 'work_dirs/cascade_rcnn_r50_sac_1x_coco_albu/cascade_rcnn_r50_sac_1x_coco.py in tools/eval_metric.py.
```
19  parser.add_argument('--config', default='work_dirs/cascade_rcnn_r50_sac_1x_coco_albu/cascade_rcnn_r50_sac_1x_coco.py',help='Config of the model')
    parser.add_argument('--pkl_results', default='result/cascade_rcnn_r50_sac_1x_coco_albu.pkl',help='Results in pickle format')
    
```
9.6 Open plot function and Visualize the Precision-Recall curve in mmdet/datasets/coco.py. 
```
548-572 open todo plot pr curve
549   change the third dimention into 0/1/2 for three classes.          
                pr_array1 = cocoEval.eval['precision'][0, :, 0, 0, 2]
                pr_array2 = cocoEval.eval['precision'][6, :, 0, 0, 2]
                pr_array3 = cocoEval.eval['precision'][7, :, 0, 0, 2]
                pr_array4 = cocoEval.eval['precision'][8, :, 0, 0, 2]
                pr_array5 = cocoEval.eval['precision'][9, :, 0, 0, 2]

```
9.7 Run tools/eval_metric.py.
```
python tools/eval_metric.py
```
10 Show mAP during trainning.
```
python tools/analyze_logs.py plot_curve /home/liangjiubujiu/Project/mmdetection-master/work_dirs/cascade_rcnn_r50_sac_1x_coco/20201214_144511.log.json /home/liangjiubujiu/Project/mmdetection-master/work_dirs/cascade_rcnn_r50_rfp_1x_coco/20201213_235433.log.json /home/liangjiubujiu/Project/mmdetection-master/work_dirs/detectors_cascade_rcnn_r50_1x_coco/20201215_033344.log.json /home/liangjiubujiu/Project/mmdetection-master/work_dirs/cascade_rcnn_r50_sac_1x_coco_albu/20201216_115552.log.json --keys bbox_mAP --legend sac rfp detectors sac+albu
```
## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Contact

This repo is currently maintained by Kai Chen ([@hellock](http://github.com/hellock)), Yuhang Cao ([@yhcao6](https://github.com/yhcao6)), Wenwei Zhang ([@ZwwWayne](https://github.com/ZwwWayne)),
Jiarui Xu ([@xvjiarui](https://github.com/xvjiarui)). Other core developers include Jiangmiao Pang ([@OceanPang](https://github.com/OceanPang)) and Jiaqi Wang ([@myownskyW7](https://github.com/myownskyW7)).
