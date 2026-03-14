"""
train.py - RT-DETR → YOLOv8-n Knowledge Distillation
=====================================================
손실함수 구성:
    total_loss = det_loss + λ × attention_loss

    attention_loss = MSE(
        YOLO_heatmap,       ← model[21] C2f 출력 (Neck 끝, Detect 직전)
        RTDETR_heatmap      ← Decoder Cross-Attention weight 합산
    )

    RTDETR_heatmap : "전역 문맥으로 물체가 있을 위치"
    YOLO_heatmap   : "CNN이 탐지 직전 반응한 위치"

출처:
    [RT-DETR]    lyuwenyu/RT-DETR
    [Ultralytics] Ultralytics YOLOv8
    [AT2017]     Paying More Attention to Attention
    [DETR]       End-to-End Object Detection with Transformers
"""

import sys
import time
import argparse
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── RT-DETR ───────────────────────────────────────────────────────────────────
sys.path.insert(0, 'RT-DETR/rtdetr_pytorch')
from src.core import YAMLConfig                                      # [RT-DETR]

# ── Ultralytics ───────────────────────────────────────────────────────────────
from ultralytics.nn.tasks import DetectionModel                      # [Ultralytics]
from ultralytics.data.build import build_yolo_dataset, build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss


# ══════════════════════════════════════════════════════════════════════════════
# Hook
# ══════════════════════════════════════════════════════════════════════════════
class FeatureHook:
    """지정 모듈의 forward 출력을 캡처"""
    def __init__(self, module: nn.Module):
        self.feat = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        self.feat = out

    def remove(self):
        self._handle.remove()


class CrossAttnWeightHook:
    """
    RT-DETR Decoder Cross-Attention → 공간적 heatmap 생성
    위치 : model.decoder.decoder.layers[-1].cross_attn

    reference_points : (B, 300, 1, 4) - 각 쿼리의 기준 좌표 (cx, cy, w, h) 0~1
    attn_weights     : (B, 300, n_heads, n_levels*n_points) - 쿼리별 반응 강도

    → cx, cy를 20×20 격자에 투영 후 반응 강도 누적
    → 결과 : (B, 400) "RT-DETR이 전역 문맥으로 물체가 있다고 판단한 공간적 위치"
    """
    def __init__(self, module: nn.Module, grid_size: int = 20):
        self.weights = None
        self.grid_size = grid_size
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        query            = input[0]   # (B, 300, C)
        reference_points = input[1]   # (B, 300, 1, 4) cx,cy,w,h  0~1

        B      = query.shape[0]
        G      = self.grid_size
        device = query.device

        # 각 쿼리의 기준 좌표 추출
        cx = reference_points[:, :, 0, 0].detach()   # (B, 300)  0~1
        cy = reference_points[:, :, 0, 1].detach()   # (B, 300)  0~1

        # 각 쿼리의 반응 강도 : softmax 후 합산
        attn_w = module.attention_weights(query)      # (B, 300, n_heads*n_levels*n_points)
        attn_w = F.softmax(attn_w, dim=-1).sum(dim=-1).detach()  # (B, 300)

        # cx, cy → 격자 인덱스
        ix  = (cx * G).long().clamp(0, G - 1)        # (B, 300)
        iy  = (cy * G).long().clamp(0, G - 1)        # (B, 300)
        idx = iy * G + ix                             # (B, 300)  1D 인덱스

        # 20×20 격자에 반응 강도 누적
        heatmap = torch.zeros(B, G * G, device=device)
        for b in range(B):
            heatmap[b].scatter_add_(0, idx[b], attn_w[b])

        self.weights = heatmap  # (B, 400)

    def remove(self):
        self._handle.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap 생성
# ══════════════════════════════════════════════════════════════════════════════
def make_yolo_heatmap(feat: torch.Tensor) -> torch.Tensor:
    """
    YOLO model[21] C2f 출력 → heatmap
    feat : (B, C, H, W)
    return : (B, H*W) 최대값 정규화 (0~1)
    """
    B = feat.shape[0]
    heatmap = feat.pow(2).sum(dim=1)        # (B, H, W)
    heatmap = heatmap.view(B, -1)           # (B, H*W)
    # 최대값으로 정규화 → 0~1 범위, 값 크기 보존
    heatmap = heatmap / (heatmap.max(dim=1, keepdim=True).values + 1e-8)
    return heatmap


def make_rtdetr_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    CrossAttnWeightHook.weights → 최대값 정규화 (0~1)
    heatmap : (B, 400) 이미 공간적으로 투영된 값
    return  : (B, 400) 최대값 정규화
    """
    return heatmap / (heatmap.max(dim=1, keepdim=True).values + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# 검증
# ══════════════════════════════════════════════════════════════════════════════
def validate(student: nn.Module, args) -> dict:
    from ultralytics.models.yolo.detect import DetectionValidator
    student.eval()
    validator = DetectionValidator(
        args=SimpleNamespace(
            data=args.data, imgsz=640, batch=args.batch,
            workers=args.workers, device=args.device,
            verbose=False, save=False, save_json=False,
            conf=0.001, iou=0.6, half=False, plots=False,
        )
    )
    with torch.no_grad():
        results = validator(model=student)
    student.train()
    # results 타입에 따라 처리
    if hasattr(results, 'results_dict'):
        rd = results.results_dict
    elif isinstance(results, dict):
        rd = results
    else:
        rd = {}
    return {
        'map50'    : rd.get('metrics/mAP50(B)',    0.0),
        'map50_95' : rd.get('metrics/mAP50-95(B)', 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(save_dir, student, optimizer, epoch, metrics, is_best):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch'    : epoch,
        'metrics'  : metrics,
        'model'    : deepcopy(student).half().state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(ckpt, save_dir / 'last.pt')
    if is_best:
        torch.save(ckpt, save_dir / 'best.pt')
        print(f'  → best.pt 저장  mAP50={metrics["map50"]:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════════════════════════
def train(args):
    device   = torch.device(args.device)
    save_dir = Path(args.save_dir)
    best_map = 0.0
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(save_dir / 'train_log.csv', 'w')
    log_file.write('epoch,total_loss,det_loss,attn_loss,map50,map50_95\n')

    # ──────────────────────────────────────────────────────────────────────────
    # [1] RT-DETR-R101 불러오기
    # ──────────────────────────────────────────────────────────────────────────
    print('[1] RT-DETR-R101 불러오기')
    cfg     = YAMLConfig(args.teacher_cfg, resume=args.teacher_ckpt)
    teacher = cfg.model.deploy()

    ckpt = torch.load(args.teacher_ckpt, map_location='cpu')
    # ckpt['ema']['module'] 구조 우선 탐색  ★
    if isinstance(ckpt, dict):
        if 'ema' in ckpt and isinstance(ckpt['ema'], dict) and 'module' in ckpt['ema']:
            state = ckpt['ema']['module']
        else:
            state = ckpt.get('ema') or ckpt.get('model') or ckpt.get('state_dict') or ckpt
    else:
        state = ckpt
    teacher.load_state_dict(state, strict=False)

    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print('  완료')

    # ──────────────────────────────────────────────────────────────────────────
    # [2] RT-DETR Cross-Attention Hook 등록
    # 위치 : model.decoder.decoder.layers[-1].cross_attn
    # 마지막 Decoder layer의 Cross-Attention weight 캡처
    # ──────────────────────────────────────────────────────────────────────────
    print('[2] RT-DETR Cross-Attention Hook 등록')
    cross_attn_hook = CrossAttnWeightHook(
        teacher.decoder.decoder.layers[-1].cross_attn
    )
    print('  완료')

    # ──────────────────────────────────────────────────────────────────────────
    # [3] YOLOv8-n 불러오기
    # ──────────────────────────────────────────────────────────────────────────
    print('[3] YOLOv8-n 불러오기')
    student = DetectionModel('yolov8n.yaml', nc=args.nc)
    ckpt    = torch.load(args.student_ckpt, map_location='cpu')
    student.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    student = student.to(device).train()

    # YOLO Hook : model[21] = C2f (Neck 끝, Detect 직전)
    yolo_hook = FeatureHook(student.model[21])

    # v8DetectionLoss
    if not hasattr(student, 'args'):
        student.args = SimpleNamespace(
            box=7.5, cls=0.5, dfl=1.5,
            fl_gamma=0.0, label_smoothing=0.0,
        )
    det_loss_fn = v8DetectionLoss(student)
    print('  완료')

    # ──────────────────────────────────────────────────────────────────────────
    # [4] DataLoader
    # ──────────────────────────────────────────────────────────────────────────
    print('[4] DataLoader 구성')
    data_cfg = check_det_dataset(args.data)

    train_cfg = get_cfg()
    train_cfg.imgsz = 640; train_cfg.rect = False
    train_cfg.cache = False; train_cfg.single_cls = False
    train_cfg.fraction = 1.0; train_cfg.task = 'detect'; train_cfg.classes = None

    val_cfg = get_cfg()
    val_cfg.imgsz = 640; val_cfg.rect = True
    val_cfg.cache = False; val_cfg.single_cls = False
    val_cfg.fraction = 1.0; val_cfg.task = 'detect'; val_cfg.classes = None

    train_loader = build_dataloader(
        build_yolo_dataset(train_cfg, data_cfg['train'], args.batch, data_cfg, mode='train', stride=32),
        batch=args.batch, workers=args.workers, shuffle=True, rank=-1,
    )
    print('  완료')

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ──────────────────────────────────────────────────────────────────────────
    # [5] 학습 루프
    # ──────────────────────────────────────────────────────────────────────────
    print(f'\n[5] 학습 시작  (epochs={args.epochs})\n{"─"*70}')

    for epoch in range(args.epochs):
        student.train()
        teacher.eval()

        epoch_total = 0.0
        epoch_det   = 0.0
        epoch_attn  = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['img'].to(device).float() / 255.0
            B, C, H, W = images.shape

            # ── RT-DETR forward → Cross-Attention weight 캡처 ────────────────
            with torch.no_grad():
                teacher(images)
            # cross_attn_hook.weights : (B, 300)
            rtdetr_heatmap = make_rtdetr_heatmap(
                cross_attn_hook.weights,                             # (B, 400) 이미 공간 투영됨
            )                                                        # (B, 400) L2 정규화

            # ── YOLOv8 forward → Neck 끝 feature 캡처 ────────────────────────
            preds = student(images)
            # yolo_hook.feat : (B, 256, 20, 20)
            yolo_heatmap = make_yolo_heatmap(yolo_hook.feat)        # (B, 400)



            # ── Detection Loss ────────────────────────────────────────────────
            det_total, _ = det_loss_fn(preds, batch)
            if det_total.dim() > 0:
                det_total = det_total.sum()

            # ── Attention Loss ────────────────────────────────────────────────
            # RT-DETR가 전역 문맥으로 물체를 찾은 위치를
            # YOLO가 탐지 직전에 같은 위치에 집중하도록 학습  [AT2017][DETR]
            attn_loss = F.mse_loss(yolo_heatmap, rtdetr_heatmap.detach())

            # ── Total Loss ────────────────────────────────────────────────────
            total_loss = det_total + args.lambda_attn * attn_loss

            # ── Backward ─────────────────────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_det   += det_total.item()
            epoch_attn  += attn_loss.item()

            if batch_idx % 50 == 0:
                print(
                    f'  epoch {epoch+1:3d} | batch {batch_idx:4d} | '
                    f'total={total_loss.item():.4f}  '
                    f'det={det_total.item():.4f}  '
                    f'attn={attn_loss.item():.4f}'
                )

        scheduler.step()

        n = len(train_loader)
        epoch_total /= n
        epoch_det   /= n
        epoch_attn  /= n

        # ── 검증 ─────────────────────────────────────────────────────────────
        val_metrics = validate(student, args)
        elapsed = time.time() - t0
        print(
            f'\n[Epoch {epoch+1}/{args.epochs}] {elapsed:.1f}s | '
            f'mAP50={val_metrics["map50"]:.4f}  '
            f'mAP50-95={val_metrics["map50_95"]:.4f}\n'
        )

        log_file.write(
            f'{epoch+1},{epoch_total:.6f},{epoch_det:.6f},'
            f'{epoch_attn:.6f},{val_metrics["map50"]:.6f},'
            f'{val_metrics["map50_95"]:.6f}\n'
        )
        log_file.flush()

        is_best = val_metrics['map50'] > best_map
        if is_best:
            best_map = val_metrics['map50']

        save_checkpoint(
            save_dir  = save_dir,
            student   = student,
            optimizer = optimizer,
            epoch     = epoch,
            metrics   = {**{'total_loss': epoch_total, 'det_loss': epoch_det,
                            'attn_loss': epoch_attn}, **val_metrics},
            is_best   = is_best,
        )

    cross_attn_hook.remove()
    yolo_hook.remove()
    log_file.close()
    print(f'\n학습 완료.  best mAP50={best_map:.4f}  저장={save_dir}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher-cfg',   required=True)
    p.add_argument('--teacher-ckpt',  required=True)
    p.add_argument('--student-ckpt',  default='yolov8n.pt')
    p.add_argument('--data',          required=True)
    p.add_argument('--nc',            type=int,   default=80)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch',         type=int,   default=16)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight-decay',  type=float, default=1e-4)
    p.add_argument('--workers',       type=int,   default=8)
    p.add_argument('--device',        default='cuda:0')
    p.add_argument('--lambda-attn',   type=float, default=0.5)
    p.add_argument('--save-dir',      default='runs/distill')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)