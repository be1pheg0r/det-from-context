import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def generalized_box_iou(boxes1, boxes2):
    # Метрика GIoU для ограничивающих рамок
    area1, area2 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0), (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt, rb = torch.max(boxes1[:, None, :2], boxes2[None, :, :2]), torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    lt2, rb2 = torch.min(boxes1[:, None, :2], boxes2[None, :, :2]), torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    area_c = wh2[..., 0] * wh2[..., 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-6)

class HungarianMatcher(nn.Module):
    # Двудольное сопоставление (Hungarian algorithm) предсказаний и таргетов
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, logits, boxes, targets):
        indices = []
        for b in range(logits.shape[0]):
            tgt_labels, tgt_boxes = targets[b]["labels"], targets[b]["boxes"]
            if tgt_labels.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            C = (self.cost_bbox * torch.cdist(boxes[b], tgt_boxes, p=1) + 
                 self.cost_class * -logits.sigmoid()[b][:, tgt_labels] + 
                 self.cost_giou * -generalized_box_iou(box_cxcywh_to_xyxy(boxes[b]), box_cxcywh_to_xyxy(tgt_boxes))).cpu()
            row, col = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices

class SetCriterion(nn.Module):
    # Focal loss для классов, L1 + GIoU для боксов
    def __init__(self, num_classes, cls_weight=2.0, bbox_weight=5.0, giou_weight=2.0):
        super().__init__()
        self.matcher = HungarianMatcher(cls_weight, bbox_weight, giou_weight)
        self.weights = {"cls": cls_weight, "bbox": bbox_weight, "giou": giou_weight}

    def forward(self, logits, boxes, targets):
        total = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        for l in range(logits.shape[0]):
            indices = self.matcher(logits[l], boxes[l], targets)
            cls_target, bbox_l1, bbox_giou, num_pos = torch.zeros_like(logits[l]), 0.0, 0.0, 0
            for b, (pi, ti) in enumerate(indices):
                if pi.numel() == 0: continue
                cls_target[b, pi, targets[b]["labels"][ti].to(logits.device)] = 1.0
                pred_b, tgt_b = boxes[l][b, pi], targets[b]["boxes"][ti].to(logits.device)
                bbox_l1 += F.l1_loss(pred_b, tgt_b, reduction="sum")
                bbox_giou += (1 - generalized_box_iou(box_cxcywh_to_xyxy(pred_b), box_cxcywh_to_xyxy(tgt_b)).diag()).sum()
                num_pos += pi.numel()
            
            num_pos = max(num_pos, 1)
            prob, p_t = logits[l].sigmoid(), logits[l].sigmoid() * cls_target + (1 - logits[l].sigmoid()) * (1 - cls_target)
            loss_cls = (F.binary_cross_entropy_with_logits(logits[l], cls_target, reduction="none") * ((1 - p_t) ** 2) * (0.25 * cls_target + 0.75 * (1 - cls_target))).sum() / num_pos
            
            total["loss_cls"] += self.weights["cls"] * loss_cls
            total["loss_bbox"] += self.weights["bbox"] * (bbox_l1 / num_pos)
            total["loss_giou"] += self.weights["giou"] * (bbox_giou / num_pos)
            
        total["loss_total"] = total["loss_cls"] + total["loss_bbox"] + total["loss_giou"]
        return total