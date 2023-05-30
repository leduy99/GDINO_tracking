import argparse
import numpy as np
from numpy import random
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

def delete_by_index(tensor, indices):
    mask = torch.ones(len(tensor), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

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

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3.5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

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
    color = [random.randint(0, 255) for _ in range(3)]

    for img in sorted(os.listdir(args.source)):
    
        step += 1
        if step < args.start_frame:
            continue

        # load image
        img_dir = os.path.join(args.source, img)
        print('processing:', img_dir)

        image = cv2.imread(img_dir)

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
        detections.confidence = np.delete(detections.confidence, rm_list, axis=0)
        max_idx = detections.confidence.argmax()

        # Sim score theo ImageBind
        crops = []
        for det in detections.xyxy:
            crop = image[int(det[1]) : int(det[3]) , int(det[0]) : int(det[2])]
            crops.append(crop)

        sims, embs = retriev_vision_and_vision(crops, max_idx)
        # frame_idx += 1

        # TODO: implement adaptive threshold
        target_conf = detections.confidence.max() * 0.7
        num_k = sum(map(lambda x : x >= target_conf, detections.confidence)) - 1
        target_sim = torch.mean(torch.sort(sims, descending=True)[0][1:num_k])

        rm_list = []
        for idx, conf in enumerate(detections.confidence):
            if conf < target_conf:
                if sims[idx] < target_sim:
                    rm_list.append(idx)
        
        detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
        detections.confidence = np.delete(detections.confidence, rm_list, axis=0)
        embs = delete_by_index(embs, rm_list)
        max_idx = detections.confidence.argmax()

        outputs = tracker.update(detections, embs.cpu(), image)

        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, detections.confidence)):
    
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                # if save_txt:
                #     # to MOT format
                #     bbox_left = output[0]
                #     bbox_top = output[1]
                #     bbox_w = output[2] - output[0]
                #     bbox_h = output[3] - output[1]
                #     # Write MOT compliant results to file
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                              #  bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                c = int(cls)  # integer class
                id = int(id)  # integer id
                label = (f'{id} {conf:.2f} {sims[j]:.2f}')
                plot_one_box(bboxes, image, label=label, color=color, line_thickness=2)

        # # Sim score theo GDino
        # result = feature_sim_from_gdino(detections.xyxy, feature, max_idx)

        # Sim score theo GDino nhưng average
        # result = roi_align_gdino(detections.xyxy.copy(), feature.tensors, max_idx)

        # labels = []
        # for i, det in enumerate(detections.xyxy):
        #     labels.append(f"{i} {detections.confidence[i]:0.2f} {sims[i]:0.04f}")

        # annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        img_path = os.path.join(dest_folder, img)
        cv2.imwrite(img_path, image)

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