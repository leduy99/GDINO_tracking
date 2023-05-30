import argparse
import numpy as np
from pathlib import Path

import sys
import os
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'boxmot' / 'ImageBind') not in sys.path:
    sys.path.append(str(ROOT / 'boxmot' / 'ImageBind'))  # add ImageBind ROOT to PATH

from groundingdino.util.inference import Model

import boxmot.ImageBind.data
import torch
from boxmot.ImageBind.models import imagebind_model
from boxmot.ImageBind.models.imagebind_model import ModalityType


GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swinb_cogcoor.pth"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Instantiate model
bind_model = imagebind_model.imagebind_huge(pretrained=True)
bind_model.eval()
bind_model.to(device)

image = cv2.imread('/content/forest.jpg')
detections, phrases, feature = grounding_dino_model.predict_with_caption(
    image=image, 
    caption='trees', 
    box_threshold=0.2, 
    text_threshold=0.2
)
print(detections.xyxy)