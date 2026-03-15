# mmdet/models/detectors/yolo_Multinex.py
import torch
import torch.nn as nn
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone
from .yolo import YOLOV3

@DETECTORS.register_module()
class MultinexYOLO(YOLOV3):
    def __init__(self,
                 enhancer,
                 backbone,
                 neck,
                 bbox_head,
                 loss_factor=50, # New arg for consistency loss weight
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MultinexYOLO, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.enhancer = build_backbone(enhancer)

    def extract_feat(self, img):
        img_plus = self.enhancer(img)
        img_plus = torch.clamp(img_plus, 0.0, 1.0)

        # Detect
        x = self.backbone(img_plus)
        if self.with_neck:
            x = self.neck(x)

        if self.training and torch.rand(1) < 0.001:
            import torchvision
            debug_path = f'debug_epoch_{img_plus.device.index}.jpg'
            debug_img = torch.cat([img[0], img_plus[0]], dim=2)
            torchvision.utils.save_image(debug_img, debug_path)
            print('[DEBUG]: saved debug image')
            
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        
        x = self.extract_feat(img)

        # Standard YOLO Loss
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        # Discard loss during inference
        x = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results