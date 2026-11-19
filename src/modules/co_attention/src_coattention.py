"""Experimental box-supervised fire/smoke bidirectional spatial co-attention.
Ultralytics 8.4.24. Keep this file importable when loading custom .pt files.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


def grid_xy(h, w, device, dtype):
    y, x = torch.meshgrid((torch.arange(h, device=device, dtype=dtype)+.5)/h,
                          (torch.arange(w, device=device, dtype=dtype)+.5)/w, indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2)


class CrossDirection(nn.Module):
    """Q from target branch; K/V exclusively from the other branch."""
    def __init__(self, dim=32, heads=4, grid=8, chunk=1024):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.grid, self.chunk = heads, grid, chunk
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, bias=False)
        self.distance_scale = nn.Parameter(torch.zeros(heads))

    def forward(self, target, source, source_gate):
        b, d, h, w = target.shape
        gh, gw = min(self.grid, h), min(self.grid, w)
        source = F.adaptive_avg_pool2d(source, (gh, gw))
        gate = F.adaptive_avg_pool2d(source_gate, (gh, gw))
        def tokens(t):
            return t.flatten(2).transpose(1, 2).reshape(b, -1, self.heads, d//self.heads).transpose(1, 2)
        q, k, v = tokens(self.q(target)), tokens(self.k(source)), tokens(self.v(source))
        # Spatial proximity is a learned bias, not a hard assumption that smoke is above fire.
        xy_q = grid_xy(h, w, q.device, torch.float32)
        xy_k = grid_xy(gh, gw, q.device, torch.float32)
        log_gate = gate.float().flatten(2).clamp_min(1e-4).log().unsqueeze(1)
        chunks = []
        for start in range(0, h*w, self.chunk):
            dist = (xy_q[start:start+self.chunk, None] - xy_k[None]).square().sum(-1)
            bias = -F.softplus(self.distance_scale.float())[None, :, None, None] * dist[None, None]
            bias = (bias + log_gate).to(q.dtype)
            chunks.append(F.scaled_dot_product_attention(q[:, :, start:start+self.chunk], k, v,
                                                         attn_mask=bias, dropout_p=0.0))
        context = torch.cat(chunks, 2).transpose(1, 2).reshape(b, h*w, d).transpose(1, 2).reshape(b, d, h, w)
        # Predicted source confidence suppresses messages when the source class is absent.
        presence = source_gate.amax((2, 3), keepdim=True)
        return self.out(context) * presence


class SpatialRelationCoAttention(nn.Module):
    def __init__(self, channels, dim=32, heads=4, grid=8, chunk=1024):
        super().__init__()
        self.fire = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.GroupNorm(4, dim), nn.SiLU())
        self.smoke = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.GroupNorm(4, dim), nn.SiLU())
        self.fire_gate = nn.Conv2d(dim, 1, 1)
        self.smoke_gate = nn.Conv2d(dim, 1, 1)
        self.fire_from_smoke = CrossDirection(dim, heads, grid, chunk)
        self.smoke_from_fire = CrossDirection(dim, heads, grid, chunk)
        self.fuse = nn.Conv2d(dim*2, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(1e-3))
        self.collect_aux = False
        self.aux_logits = None

    def forward(self, x):
        fire, smoke = self.fire(x), self.smoke(x)
        lf, ls = self.fire_gate(fire), self.smoke_gate(smoke)
        gf, gs = lf.sigmoid(), ls.sigmoid()
        if self.collect_aux:
            self.aux_logits = torch.cat((lf, ls), 1)
        fire, smoke = fire*gf, smoke*gs
        # Simultaneous exchange: both directions read original branch features.
        cf = self.fire_from_smoke(fire, smoke, gs)
        cs = self.smoke_from_fire(smoke, fire, gf)
        return x + self.gamma * self.fuse(torch.cat((cf*gf, cs*gs), 1))


class P2WithSRC(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.src = SpatialRelationCoAttention(base.cv2.conv.out_channels, **config)
        # Preserve YOLO graph indices and routes exactly.
        for name in ('i', 'f', 'type', 'np'):
            setattr(self, name, getattr(base, name))
        self.type = 'P2WithSpatialRelationCoAttention'
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.src(self.base(x))


def box_targets(logits, batch):
    """Coarse occupancy from AUGMENTED normalized xywh boxes, not segmentation labels.
    Floor/ceil ensures sub-cell tiny boxes occupy at least one feature cell.
    """
    b, _, h, w = logits.shape
    target = torch.zeros((b, 2, h, w), device=logits.device, dtype=torch.float32)
    # Transfer small annotation arrays once, avoiding per-coordinate GPU syncs.
    ids = batch['batch_idx'].detach().flatten().long().cpu().tolist()
    cls = batch['cls'].detach().flatten().long().cpu().tolist()
    boxes = batch['bboxes'].detach().float().cpu().tolist()
    for bi, ci, (cx, cy, bw, bh) in zip(ids, cls, boxes):
        if ci not in (0, 1):
            raise ValueError('Required mapping: 0=fire, 1=smoke')
        x1 = max(0, min(w-1, math.floor((cx-bw/2)*w)))
        y1 = max(0, min(h-1, math.floor((cy-bh/2)*h)))
        x2 = min(w, max(x1+1, math.ceil((cx+bw/2)*w)))
        y2 = min(h, max(y1+1, math.ceil((cy+bh/2)*h)))
        target[bi, ci, y1:y2, x1:x2] = 1
    return target


def occupancy_loss(logits, target):
    # Equal average of positive/negative regions when both exist. Absent classes
    # receive negative BCE only. No exclusive softmax: fire/smoke can overlap.
    pixel = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
    pos, neg = target.sum((2, 3)), (1-target).sum((2, 3))
    lp = (pixel*target).sum((2, 3))/pos.clamp_min(1)
    ln = (pixel*(1-target)).sum((2, 3))/neg.clamp_min(1)
    return ((lp*(pos>0) + ln*(neg>0))/((pos>0).float()+(neg>0).float()).clamp_min(1)).mean()


class SRCDetectionModel(DetectionModel):
    def __init__(self, cfg='yolo26n-p2.yaml', ch=3, nc=2, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        assert self.yaml.get('scale') == 'n', 'This implementation is validated for nano only.'
        assert self.model[-1].f == [19, 22, 25, 28], 'Unexpected P2 graph in pinned runtime'
        self.aux_weight = float(self.yaml.get('src_aux_weight', 0.1))
        config = self.yaml.get('src_config', dict(dim=32, heads=4, grid=8, chunk=1024))
        # New module does not advance RNG of the shared architecture.
        with torch.random.fork_rng(devices=[]):
            self.model[19] = P2WithSRC(self.model[19], config)

    def loss(self, batch, preds=None):
        block = self.model[19].src
        block.collect_aux = preds is None and self.training
        try:
            loss, items = super().loss(batch, preds)
            aux = loss.new_zeros(())
            if block.aux_logits is not None:
                aux = occupancy_loss(block.aux_logits, box_targets(block.aux_logits, batch))*self.aux_weight
            # Detection losses are batch-scaled; preserve that convention for auxiliary loss.
            return torch.cat((loss.reshape(-1), (aux*batch['img'].shape[0]).reshape(1))), torch.cat((items, aux.detach().reshape(1)))
        finally:
            block.collect_aux = False
            block.aux_logits = None


def transfer_plain_state(model, plain_state):
    target = model.state_dict()
    mapped = {}
    for key, value in plain_state.items():
        out_key = key
        if isinstance(model, SRCDetectionModel) and key.startswith('model.19.'):
            out_key = key.replace('model.19.', 'model.19.base.', 1)
        if out_key not in target or target[out_key].shape != value.shape:
            raise ValueError(f'Paired initialization mismatch: {key}')
        mapped[out_key] = value
    model.load_state_dict(mapped, strict=False)
    for key, value in mapped.items():
        assert torch.equal(model.state_dict()[key].cpu(), value.cpu()), key
    print(f'Paired initialization: {len(mapped)} shared tensors copied and verified')


class SRCTrainer(DetectionTrainer):
    def __init__(self, *args, initial_state=None, experiment='src', **kwargs):
        self.initial_state, self.experiment = initial_state, experiment
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        cls = SRCDetectionModel if self.experiment == 'src' else DetectionModel
        model = cls(cfg, nc=self.data['nc'], ch=self.data['channels'], verbose=verbose)
        if weights is not None:
            model.load(weights)
        elif self.initial_state:
            state = torch.load(self.initial_state, map_location='cpu', weights_only=True)
            transfer_plain_state(model, state)
        return model

    def get_validator(self):
        validator = super().get_validator()
        if self.experiment == 'src':
            self.loss_names = ('box_loss', 'cls_loss', 'dfl_loss', 'src_aux_loss')
        return validator
