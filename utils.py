from torchvision import transforms, ops
import torch.nn.functional as F
import torch

import numpy as np

class FilterTools():
    def __init__(self, num_target, two_filters):
        self.num_target = num_target
        self.two_filters = two_filters

        self.best_conf = None
        self.best_emb = None
        self.target_mem = None

    def feature_sim_from_gdino(self, dets, feature, ref_pos, ref_conf):
        rh,  rw = feature.tensors.shape[2] / 1080 , feature.tensors.shape[3] / 1920
        det_feats = []
        for det in dets:
            xc = int((det[0] + det[2])/2 * rw)
            yc = int((det[1] + det[3])/2 * rh)

            det_feats.append(feature.tensors[:, :, yc, xc])

            # # RoI Align
            # x, y, xx, yy = det
            # if xx - x == 0:
            #     xx += 1
            # if yy - y == 0:
            #     yy += 1

            # roi = feature.tensors[:, :, int(y * rh):int(yy * rh),
            #                             int(x * rw):int(xx * rw)]
            # roi = torch.nn.functional.interpolate(roi, size=(1, 1), mode='bicubic', align_corners=True)
            # roi.squeeze()
            # det_feats.append(roi)

        embs = torch.squeeze(torch.stack(det_feats, dim=0), 1)
        ref_emb = embs[ref_pos].unsqueeze(0)

        if self.target_mem == None:
            self.target_mem = ref_emb

            if self.two_filters:
                self.best_conf = np.array([ref_conf])
                self.best_emb = ref_emb
        else:
            if self.two_filters:
                if self.best_emb.size()[0] < 10:
                    np.append(self.best_conf , ref_conf)
                    self.best_emb = torch.cat((self.best_emb, ref_emb), dim=0)
                else:
                    if self.best_conf.min() < ref_conf:
                        min_idx = self.best_conf.argmin()
                        self.best_conf[min_idx] = ref_conf
                        self.best_emb[min_idx] = ref_emb

            if self.target_mem.size()[0] < self.num_target:
                self.target_mem = torch.cat((self.target_mem, ref_emb), dim=0)

            elif self.target_mem.size()[0] == self.num_target:
                self.target_mem = torch.cat((self.target_mem[1:, :], ref_emb), dim=0)

        t1_norm = F.normalize(embs, dim=1)
        t2_norm = F.normalize(self.target_mem, dim=1)
        result = torch.mean(torch.mm(t1_norm, t2_norm.t()), dim=1)

        if self.two_filters:
            t3_norm = F.normalize(self.best_emb, dim=1)
            best_res = torch.mean(torch.mm(t1_norm, t3_norm.t()), dim=1)
        else:
            best_res = None

        return result, best_res, embs

def nms(bounding_boxes, confidence_score, threshold=0.6):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = bounding_boxes

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes), np.array(picked_score)