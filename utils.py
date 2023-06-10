from torchvision import transforms, ops
import torch.nn.functional as F
import torch

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
                self.best_conf = ref_conf
                self.best_emb = ref_emb
        else:
            if self.two_filters and ref_conf >= self.best_conf:
                self.best_emb = ref_emb
                self.best_conf = ref_conf

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