dataset_type = "UG2FaceDataset"
data_root = "/home/ubuntu/datasets/DarkFace_voc/"
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Expand",
        mean=img_norm_cfg["mean"],
        to_rgb=img_norm_cfg["to_rgb"],
        ratio_range=(1, 2),
    ),
    dict(
        type="MinIoURandomCrop",
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3,
    ),
    dict(type="Resize", img_scale=[(320, 320), (664, 664)], keep_ratio=True),
    # dict(type="Resize", img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    # dict(type='Resize', img_scale=[(360, 360), (720, 720)], keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        # img_scale=(608, 608),
        img_scale=(664, 664),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            # dict(type='Normalize', **img_norm_cfg),
            dict(
                type="Normalize", mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train.txt",
            img_prefix=data_root + "pairlie_train_images",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "test.txt",
        img_prefix=data_root + "pairlie_test_images",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.txt",
        img_prefix=data_root + "pairlie_test_images",
        pipeline=test_pipeline,
    ),
)
