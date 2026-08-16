import math
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from dataclasses import dataclass

@dataclass
class DetectionBatch:
    images: torch.Tensor
    targets: Optional[List[Dict[str, torch.Tensor]]] = None
    sequence_id: Optional[str] = None
    frame_id: Optional[int] = None
    timestamp: Optional[float] = None

@dataclass
class DetectorOutput:
    logits: torch.Tensor 
    boxes: torch.Tensor   
    queries: Optional[torch.Tensor] = None
    reference_points: Optional[torch.Tensor] = None
    features: Optional[List[torch.Tensor]] = None
    decoder_layers: Optional[List[Dict[str, torch.Tensor]]] = None

class SinePositionEmbedding2D(nn.Module):
    # Синусное/косинусное позиционное кодирование
    def __init__(self, num_pos_feats: int, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        device = feat.device
        y = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).expand(H, W)
        x = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).expand(H, W)
        y, x = y / (H + 1e-6) * 2 * math.pi, x / (W + 1e-6) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x, pos_y = x[..., None] / dim_t, y[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        return pos.flatten(1, 2)

class MultiScaleProjector(nn.Module):
    # Преобразует фичи backbone в несколько масштабов
    def __init__(self, in_dim: int, hidden_dim: int, n_levels: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 1), nn.GroupNorm(32, hidden_dim))
        self.downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.GroupNorm(32, hidden_dim))
            for _ in range(n_levels - 1)
        ])

    def forward(self, feat):
        feats = [self.input_proj(feat)]
        for down in self.downsample:
            feats.append(down(feats[-1]))
        return feats

class DINOv3Backbone(nn.Module):
    # Обёртка над HF DINOv3
    def __init__(self, hf_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m", freeze: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_name)
        self.patch_size, self.embed_dim = self.model.config.patch_size, self.model.config.hidden_size
        self.num_special_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        if freeze:
            for p in self.model.parameters(): p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        h, w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        tokens = self.model(pixel_values=images).last_hidden_state
        return tokens[:, self.num_special_tokens:, :].transpose(1, 2).reshape(B, self.embed_dim, h, w)

#  Внимание и декодер
class MSDeformAttn(nn.Module):
    # Multi-scale deformable внимание через grid_sample
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.n_levels, self.n_heads, self.n_points = n_levels, n_heads, n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj, self.output_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        for p in [self.sampling_offsets, self.attention_weights]: nn.init.constant_(p.weight, 0.); nn.init.constant_(p.bias, 0.)

    def forward(self, query, reference_points, value, spatial_shapes):
        B, Q, C = query.shape
        H, P, D, L = self.n_heads, self.n_points, self.head_dim, self.n_levels
        value = self.value_proj(value).view(B, -1, H, D)
        value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
        offsets = self.sampling_offsets(query).view(B, Q, H, L, P, 2)
        attn = F.softmax(self.attention_weights(query).view(B, Q, H, L * P), dim=-1).view(B, Q, H, L, P)

        out = query.new_zeros(B, H, D, Q)
        for lvl, (h, w) in enumerate(spatial_shapes):
            v = value_list[lvl].permute(0, 2, 3, 1).reshape(B * H, D, h, w)
            loc = reference_points[:, :, None, None, :] + offsets[:, :, :, lvl] / query.new_tensor([w, h])
            loc = loc.clamp(0.0, 1.0) * 2 - 1
            sampled = F.grid_sample(v, loc.permute(0, 2, 1, 3, 4).reshape(B * H, Q, P, 2), mode="bilinear", align_corners=False)
            w_lvl = attn[:, :, :, lvl, :].permute(0, 2, 1, 3)
            out = out + torch.einsum("bhdqp,bhqp->bhdq", sampled.view(B, H, D, Q, P), w_lvl)
        return self.output_proj(out.permute(0, 3, 1, 2).reshape(B, Q, C))

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=3, n_points=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, reference_points, memory, spatial_shapes):
        q = k = tgt + query_pos
        sa_out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout(sa_out))
        ca_out = self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout(ca_out))
        return self.norm3(tgt + self.dropout(self.ffn(tgt)))

class DeformableDecoder(nn.Module):
    # Iterative box refinement
    def __init__(self, d_model=256, num_layers=6, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList([DeformableDecoderLayer(d_model) for _ in range(num_layers)])
        self.class_heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(num_layers)])
        self.bbox_heads = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4)) for _ in range(num_layers)])
        for bh in self.bbox_heads: nn.init.constant_(bh[-1].weight, 0.); nn.init.constant_(bh[-1].bias, 0.)

    def forward(self, tgt, query_pos, ref, memory, spatial_shapes):
        out_logits, out_boxes = [], []
        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, query_pos, ref, memory, spatial_shapes)
            delta = self.bbox_heads[i](tgt)
            box = (torch.cat([inverse_sigmoid(ref), torch.zeros_like(ref)], dim=-1) + delta).sigmoid()
            out_logits.append(self.class_heads[i](tgt))
            out_boxes.append(box)
            ref = box[..., :2].detach()
        return torch.stack(out_logits), torch.stack(out_boxes)
    
class RFDetrBaseline(nn.Module):
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        cfg = config or {}
        self.hidden_dim, self.num_classes, self.num_queries, self.n_levels = 256, cfg.get("num_classes", 10), 300, 3
        self.backbone = DINOv3Backbone(freeze=cfg.get("freeze_backbone", False))
        self.projector = MultiScaleProjector(self.backbone.embed_dim, self.hidden_dim)
        self.pos_embed = SinePositionEmbedding2D(self.hidden_dim)
        self.level_embed = nn.Parameter(torch.randn(self.n_levels, self.hidden_dim) * 0.02)
        self.decoder = DeformableDecoder(self.hidden_dim, num_classes=self.num_classes)
        self.enc_output, self.enc_output_norm = nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim)
        self.enc_class_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.enc_bbox_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, 4))
        self.query_pos_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, self.hidden_dim))

    @property
    def dim(self) -> int:
        return self.hidden_dim

    def initial_queries(self, batch: DetectionBatch) -> torch.Tensor:
        b = batch.images.shape[0] if batch.images is not None else 1
        return torch.zeros(b, self.num_queries, self.hidden_dim, device=batch.images.device)

    def encode_context_frames(self, context: Any) -> Optional[List[torch.Tensor]]:
        return None

    def freeze(self, backbone: bool = True, decoder: bool = False) -> None:
        pass

    def _flatten_levels(self, feats):
        srcs, poss, shapes = [], [], []
        for lvl, f in enumerate(feats):
            shapes.append(f.shape[-2:])
            srcs.append(f.flatten(2).transpose(1, 2))
            poss.append(self.pos_embed(f) + self.level_embed[lvl].view(1, 1, -1))
        return torch.cat(srcs, dim=1), torch.cat(poss, dim=1), shapes

    def forward(self, batch: DetectionBatch, context: Optional[Any] = None, query_init: Optional[torch.Tensor] = None) -> DetectorOutput:
        feats = self.projector(self.backbone(batch.images))
        # Здесь будет добавляться временной контекст 
        memory, pos_flat, shapes = self._flatten_levels(feats)
        B = memory.shape[0]
        enc = self.enc_output_norm(self.enc_output(memory + pos_flat))
        enc_scores = self.enc_class_head(enc).max(-1).values
        
        anchors = []
        for H, W in shapes:
            gy, gx = torch.meshgrid((torch.arange(H, device=memory.device) + 0.5) / H, (torch.arange(W, device=memory.device) + 0.5) / W, indexing="ij")
            anchors.append(torch.cat([torch.stack([gx, gy], -1).reshape(-1, 2), torch.full((H * W, 2), 1.0 / max(H, W), device=memory.device)], -1))
        anchors = torch.cat(anchors, dim=0).unsqueeze(0).expand(B, -1, -1)
        
        enc_boxes = (inverse_sigmoid(anchors) + self.enc_bbox_head(enc)).sigmoid()
        idx = enc_scores.topk(min(self.num_queries, memory.shape[1]), dim=1).indices
        idx_c = idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        
        tgt, ref_points = torch.gather(enc, 1, idx_c).detach(), torch.gather(enc_boxes, 1, idx.unsqueeze(-1).expand(-1, -1, 4))[..., :2].detach()
        logits, boxes = self.decoder(tgt, self.query_pos_mlp(torch.gather(pos_flat, 1, idx_c)), ref_points, memory, shapes)
        
        return DetectorOutput(
            logits=logits, 
            boxes=boxes,
            queries=tgt,
            reference_points=ref_points,
            features=feats,
            decoder_layers=[]
        )