import argparse
import numpy as np
from numpy import random
from numpy.linalg import norm
from pathlib import Path

import sys
import os
import re
import cv2

from torchvision import transforms, ops
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
from utils import FilterTools, nms, cal_iou, contains_bbox

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

# bind_model = imagebind_model.imagebind_huge(pretrained=True)
# bind_model.eval()
# bind_model.to(device)

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
    return vision_referring_result, cropped_box_embeddings

def clean_text(input_text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', input_text)
    return cleaned_text

def process_bboxes(detections, phrases, sub_parts, negative_parts):
    if sub_parts != '':
        rm_list = []
        for box_id in range(len(detections.xyxy)):
            #Check if detected box is the main object
            cls_name = clean_text(phrases[box_id])
            if (cls_name in sub_parts) or (sub_parts in cls_name):
                rm_list.append(box_id)
                continue
        phrases = np.delete(phrases, rm_list, axis=0)
        detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
        detections.confidence = np.delete(detections.confidence, rm_list, axis=0)

    rm_list = []
    for box_id in range(len(detections.xyxy)):
        if negative_parts != '' and negative_parts in phrases[box_id]:
            rm_list.append(box_id)
            continue

        # Remove overlapped boxes
        cnt = 0
        for id, box in enumerate(detections.xyxy):
            if box_id != id and cal_iou(detections.xyxy[box_id], box, 'min') > 0.85:
                if negative_parts != '' and negative_parts in phrases[id]:
                    cnt = 2
                else:
                    cnt += 1
                if cnt > 1:
                    break
        if cnt > 1:
            rm_list.append(box_id)

    phrases = np.delete(phrases, rm_list, axis=0)
    detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
    detections.confidence = np.delete(detections.confidence, rm_list, axis=0)
    detections.xyxy, detections.confidence = nms(detections.xyxy, detections.confidence, 0.45)

    return detections, phrases

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
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3.5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

@torch.no_grad()
def run(args):

  with open(args.source, "r") as f:
    for line in f:
        folder_set = line.split(': ')
        folder_name = folder_set[0]

        #construct each type of prompts
        prompts = folder_set[1].split('; ')
        sub_parts = ''
        negative_parts = ''
        if len(prompts) > 1:
            for i in range(1, len(prompts)):
                prompt_type = prompts[i].split(' -- ')[0]
                if prompt_type == 'sub_parts':
                   sub_parts = prompts[i].split(' -- ')[1]
                else:
                    negative_parts = prompts[i].split(' -- ')[1]

        print(f'start tracking {folder_name}')
        
        src_folder = os.path.join('/content/GenericMOT_JPEG_Sequence', folder_name, 'img1')
        dest_folder = os.path.join(args.save_dir, 'img', folder_name)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Create text path and clear content in text file
        parent_txt_path = os.path.join(args.save_dir, 'text', args.feature_mode)
        if not os.path.exists(parent_txt_path):
            os.makedirs(parent_txt_path)

        txt_path = os.path.join(parent_txt_path, folder_name + '.txt')

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
        image = None
        frame_idx = 0

        tools = FilterTools(args.short_mems, args.long_mems)

        box_annotator = sv.BoxAnnotator()
        color = [204, 0, 102]

        # Construct text prompt
        text_prompt = prompts[0]
        if sub_parts != '':
            text_prompt = f'{text_prompt} has {sub_parts}'
        if negative_parts != '':
            text_prompt = f'{text_prompt}. {negative_parts}'

        for img in sorted(os.listdir(src_folder)):
        
            #Option for tracking on a snippet
            if frame_idx < args.start_frame:
                frame_idx += 1
                continue

            if (frame_idx > args.end_frame) and (args.end_frame != 0):
                break

            # load image
            img_dir = os.path.join(src_folder, img)
            print('processing:', img_dir)

            image = cv2.imread(img_dir)

            # detect objects
            detections, phrases, feature = grounding_dino_model.predict_with_caption(
                image=image, 
                caption=text_prompt, 
                box_threshold=0.2, 
                text_threshold=0.2
            )
            detections, phrases = process_bboxes(detections, phrases, sub_parts, negative_parts)
            max_idx = detections.confidence.argmax()

            if args.feature_mode != "gdino":
                # Sim score theo ImageBind
                crops = []
                for det in detections.xyxy:
                    crop = image[int(det[1]) : int(det[3]) , int(det[0]) : int(det[2])]
                    crops.append(crop)

                sims, embs = retriev_vision_and_vision(crops, max_idx)
            else:
                # Sim score theo GDino
                sims, best_sims, cropped_sims, embs = tools.feature_sim_from_gdino(detections.xyxy, feature, max_idx,
                                                                      detections.confidence[max_idx])

            # Sim score theo GDino nhưng average
            # result = roi_align_gdino(detections.xyxy.copy(), feature.tensors, max_idx)


            # Adaptive Threshold
            if (args.short_mems > 0) or (args.long_mems > 0):
                target_conf = np.mean(detections.confidence) - 1.29*np.std(detections.confidence)
                num_k = sum(map(lambda x : x >= target_conf, detections.confidence)) - 1

                # Two-level filter
                rm_list = []
                for idx, conf in enumerate(detections.confidence):
                    if conf < target_conf:
                        # Level 2 is optional, sometimes it is better with only one level
                        if args.short_mems > 0:
                          target_sim_1 = torch.mean(torch.sort(sims.detach().clone(), descending=True)[0][1:num_k])
                          if sims[idx] < target_sim_1:
                              rm_list.append(idx)

                        if args.long_mems > 0:
                          target_sim_2 = torch.mean(torch.sort(best_sims.detach().clone(), 
                                                      descending=True)[0][1:num_k])
                          if best_sims[idx] < target_sim_2:
                          
                              if args.cropped_mems:
                                  target_sim_3 = torch.mean(torch.sort(cropped_sims.detach()
                                                      .clone(), descending=True)[0][1:num_k])
                                  if cropped_sims[idx] < target_sim_3:
                                      rm_list.append(idx)
                              else:
                                    rm_list.append(idx)
                          else:
                              if args.short_mems > 0:
                                  if idx in rm_list:
                                      rm_list.remove(idx)


                # Delete filtered objects
                detections.xyxy = np.delete(detections.xyxy, rm_list, axis=0)
                detections.confidence = np.delete(detections.confidence, rm_list, axis=0)
                embs = delete_by_index(embs, rm_list)
                sims = delete_by_index(sims, rm_list)
                max_idx = detections.confidence.argmax()

            #Feed data into tracker
            # sims = sims.cpu().detach().numpy()
            mean_vector = torch.mean(embs, dim=0)
            measures = []
            for idx, emb in enumerate(embs):
                sim = torch.nn.functional.cosine_similarity(mean_vector, emb, dim=-1).cpu()
                measures.append(sim)
              
            outputs = tracker.update(detections, measures, embs.cpu(), image)

            if len(outputs) > 0:
                for j, output in enumerate(outputs):
                
                    bboxes = output[0:4]
                    id = output[4]
                    conf = output[5]
                    cls = output[6]
                    sim = output[7]

                    if args.save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path, 'a') as f:
                            f.write(('%g,' * 9 + '%g' + '\n') % (frame_idx, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, conf, -1, -1, -1))

                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    label = (f'{id} {conf:.2f} {sim:.2f}')
                    plot_one_box(bboxes, image, label=label, color=color, line_thickness=2)

            # labels = []
            # for i, det in enumerate(detections.xyxy):
            #     labels.append(f"{i} {detections.confidence[i]:0.2f} {sims[i]:0.04f}")

            # annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

            img_path = os.path.join(dest_folder, img)
            cv2.imwrite(img_path, image)
            frame_idx += 1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-method', type=str, default='macsortv2', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--save-txt', action='store_false', help='save tracking results in a txt file')
    parser.add_argument('--save-dir', type=str, default='/content/drive/MyDrive/FPT-AI/GDinoBase')
    parser.add_argument('--feature-mode', type=str, default='gdino')
    parser.add_argument('--short-mems', type=int, default=0, help='re-filter with best embedding')
    parser.add_argument('--long-mems', type=int, default=0, help='re-filter with best embedding')
    parser.add_argument('--cropped-mems', action='store_true', help='re-filter for occluded objects')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=0)
    opt = parser.parse_args()
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)