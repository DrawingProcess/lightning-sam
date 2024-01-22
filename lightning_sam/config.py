from box import Box

config = {
    "num_devices": 2,
    "batch_size": 4,
    "num_workers": 4,
    "num_epochs": 200,
    "eval_interval": 5,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "/home/avs/lightning-sam/checkpoint/sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/home/avs/dataset/train_exd/EVT_20230715_112856_F_trim",
            "annotation_file": "/home/avs/dataset/train_exd/EVT_20230715_112856_F_trim.json"
        },
        "val": {
            "root_dir": "/home/avs/dataset/train_exd/EVT_20230715_112856_F_trim",
            "annotation_file": "/home/avs/dataset/train_exd/EVT_20230715_112856_F_trim.json"
        }
    }
}

cfg = Box(config)
