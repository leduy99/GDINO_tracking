import argparse
import numpy as np
from pathlib import Path

import sys
import os
import cv2

from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

import supervision as sv
from traitlets.traitlets import _SimpleTest
from ultralytics.yolo.utils.checks import check_imgsz, print_args

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'boxmot' / 'ImageBind') not in sys.path:
    sys.path.append(str(ROOT / 'boxmot' / 'ImageBind'))  # add ImageBind ROOT to PATH

from groundingdino.util.inference import Model

import boxmot.ImageBind.data as data
import torch

from boxmot.tracker_zoo import create_tracker
from boxmot.ImageBind.models import imagebind_model
from boxmot.ImageBind.models.imagebind_model import ModalityType

GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swinb_cogcoor.pth"

# Init models
device = "cuda:0" if torch.cuda.is_available() else "cpu"

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

bind_model = imagebind_model.imagebind_huge(pretrained=True)
bind_model.eval()
bind_model.to(device)

# Define funcs
def post_process_detect(dets, sims):
    pass

def load_and_transform_vision_data_from_pil_image(img_list, device):
    if img_list is None:
        return None

    image_ouputs = []
    for image in img_list:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)

def retriev_vision_and_vision(elements, ref_pos, text_list=['']):
    inputs = {
        ModalityType.VISION: load_and_transform_vision_data_from_pil_image(elements, device),
        ModalityType.TEXT: data.load_and_transform_text(text_list, device)
    }
    with torch.no_grad():
        embeddings = bind_model(inputs)

    # cropped box region embeddings
    if text_list[0] == '':
        cropped_box_embeddings = embeddings[ModalityType.VISION][: , :]
        referring_embeddings = embeddings[ModalityType.VISION][ref_pos, :]
    else:
        cropped_box_embeddings = embeddings[ModalityType.VISION][: , :]
        referring_embeddings = embeddings[ModalityType.TEXT]

    # vision_referring_result = torch.softmax(cropped_box_embeddings @ referring_image_embeddings.T, dim=0),
    vision_referring_result = F.cosine_similarity(cropped_box_embeddings, referring_embeddings)
    return vision_referring_result, cropped_box_embeddings  # [113, 1]

@torch.no_grad()
def run(args):

    dest_folder = os.path.join(args.save_dir, args.source.split('/')[-3])
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Init Tracker
    tracking_config = \
        ROOT /\
        'boxmot' /\
        opt.tracking_method /\
        'configs' /\
        (opt.tracking_method + '.yaml')

    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        device,
      )

    detections = None
    ori_image = None
    image = None
    step = 0

    box_annotator = sv.BoxAnnotator()

    for img in sorted(os.listdir(args.source)):
    
        step += 1
        if step < args.start_frame:
            continue

        # load image
        img_dir = os.path.join(args.source, img)
        print('processing:', img_dir)

        image = cv2.imread(img_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect objects
        detections, phrases, feature = grounding_dino_model.predict_with_caption(
            image=image, 
            caption=args.text_prompt, 
            box_threshold=0.15, 
            text_threshold=0.2
        )

        rm_list = []
        for box_id in range(len(detections.xyxy)):
            xo1, yo1, xo2, yo2 = detections.xyxy[box_id] 
            cnt = 0
            for box in detections.xyxy:
                xi1, yi1, xi2, yi2 = box
                if cnt > 1:
                    break
                if xi1 >= xo1 and yi1 >= yo1 and xi2 <= xo2 and yi2 <= yo2:
                    cnt += 1
            if cnt > 1:
                rm_list.append(box_id)

        detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
        max_idx = detections.confidence.argmax()

        # Sim score theo ImageBind
        crops = []
        for det in detections.xyxy:
            crop = image[int(det[1]) : int(det[3]) , int(det[0]) : int(det[2])]
            crops.append(crop)

        sims, embs = retriev_vision_and_vision(crops, max_idx)
        # frame_idx += 1
        outputs = tracker.update(detections.xyxy, embs, image)

        # # Sim score theo GDino
        # result = feature_sim_from_gdino(detections.xyxy, feature, max_idx)

        # Sim score theo GDino nhưng average
        # result = roi_align_gdino(detections.xyxy.copy(), feature.tensors, max_idx)

        labels = []
        for i, det in enumerate(detections.xyxy):
            labels.append(f"{i} {detections.confidence[i]:0.2f} {sims[i]:0.04f}")

        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        img_path = os.path.join(dest_folder, img)
        cv2.imwrite(img_path, annotated_image)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--save-txt', action='store_true', help='save tracking results in a txt file')
    parser.add_argument('--save-dir', type=str, default='/content/drive/MyDrive/FPT-AI/GDinoBind')
    parser.add_argument('--text-prompt', type=str, default='red object')
    parser.add_argument('--start-frame', type=int, default=1)
    parser.add_argument('--end-frame', type=int, default=100)
    opt = parser.parse_args()
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)