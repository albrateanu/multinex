from ..builder import DETECTORS, build_backbone
from .yolo import YOLOV3

@DETECTORS.register_module()
class MultinexYOLO(YOLOV3):
    def __init__(self,
                 enhancer, # Config for Multinex
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLOV3, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        # Build the enhancer
        self.enhancer = build_backbone(enhancer)

    def extract_feat(self, img):
        # 1. Enhance
        img_enhanced = self.enhancer(img)
        
        # 2. Detect (Standard YOLO logic)
        x = self.backbone(img_enhanced)
        if self.with_neck:
            x = self.neck(x)
        return x