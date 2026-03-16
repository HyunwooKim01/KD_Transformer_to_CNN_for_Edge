"""
train.py - RT-DETR → YOLOv8-n Knowledge Distillation
=====================================================
손실함수 구성:
    total_loss = det_loss + λ × attention_loss

    attention_loss = Pearson(P3_heatmap, rtdetr_small_heatmap)  ← 소형 객체
                   + Pearson(P4_heatmap, rtdetr_mid_heatmap)    ← 중형 객체
                   + Pearson(P5_heatmap, rtdetr_large_heatmap)  ← 대형 객체

    Pearson Correlation Loss 사용 이유 :
        값의 절대적 크기 무관, 패턴(상대적 반응 순위)만 비교
        RT-DETR(sparse) vs YOLO(dense) 스케일 차이 완전 해결
        loss = 1 - cosine_similarity(x - mean(x), y - mean(y))

    쿼리 크기 분류 (reference_points w*h 기준):
        소형 : w*h < 0.01  → 80×80 격자 → YOLO P3 (model[15])
        중형 : w*h < 0.10  → 40×40 격자 → YOLO P4 (model[18])
        대형 : w*h >= 0.10 → 20×20 격자 → YOLO P5 (model[21])

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
from ultralytics import YOLO
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


# ★ 변경 1 : CrossAttnWeightHook
# 단일 20×20 heatmap → 쿼리 크기별 3개 heatmap (P3/P4/P5)
class CrossAttnWeightHook:
    """
    RT-DETR Decoder Cross-Attention → 객체 크기별 공간 heatmap 생성
    위치 : model.decoder.decoder.layers[-1].cross_attn

    reference_points : (B, 300, 1, 4) - cx, cy, w, h  (0~1 정규화)
    attn_weights     : 쿼리별 반응 강도

    쿼리를 w*h 기준으로 3그룹으로 분류:
        소형 (w*h < 0.01)  → 80×80 격자 → heatmap_small  (B, 6400)
        중형 (w*h < 0.10)  → 40×40 격자 → heatmap_mid    (B, 1600)
        대형 (w*h >= 0.10) → 20×20 격자 → heatmap_large  (B,  400)
    """
    # 쿼리 크기 분류 임계값
    SMALL_THR = 0.01   # w*h < 0.01 → 소형
    MID_THR   = 0.10   # w*h < 0.10 → 중형, 이상 → 대형

    def __init__(self, module: nn.Module):
        self.heatmap_small = None   # (B, 6400) P3 대응
        self.heatmap_mid   = None   # (B, 1600) P4 대응
        self.heatmap_large = None   # (B,  400) P5 대응
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        query            = input[0]   # (B, 300, C)
        reference_points = input[1]   # (B, 300, 1, 4) cx,cy,w,h  0~1

        B      = query.shape[0]
        device = query.device

        cx = reference_points[:, :, 0, 0].detach()   # (B, 300)
        cy = reference_points[:, :, 0, 1].detach()   # (B, 300)
        qw = reference_points[:, :, 0, 2].detach()   # (B, 300) 쿼리 너비
        qh = reference_points[:, :, 0, 3].detach()   # (B, 300) 쿼리 높이
        area = qw * qh                                # (B, 300) 쿼리 면적

        # 쿼리별 반응 강도
        attn_w = module.attention_weights(query)
        attn_w = F.softmax(attn_w, dim=-1).sum(dim=-1).detach()  # (B, 300)

        # 크기별 마스크
        mask_small = area < self.SMALL_THR                          # 소형
        mask_mid   = (area >= self.SMALL_THR) & (area < self.MID_THR)  # 중형
        mask_large = area >= self.MID_THR                           # 대형

        self.heatmap_small = self._project(cx, cy, attn_w, mask_small, B, G=80, device=device)
        self.heatmap_mid   = self._project(cx, cy, attn_w, mask_mid,   B, G=40, device=device)
        self.heatmap_large = self._project(cx, cy, attn_w, mask_large, B, G=20, device=device)

    @staticmethod
    def _project(cx, cy, attn_w, mask, B, G, device):
        """해당 마스크의 쿼리들을 G×G 격자에 반응 강도 누적"""
        heatmap = torch.zeros(B, G * G, device=device)
        ix  = (cx * G).long().clamp(0, G - 1)
        iy  = (cy * G).long().clamp(0, G - 1)
        idx = iy * G + ix                              # (B, 300)
        for b in range(B):
            m = mask[b]                                # (300,) bool
            if m.any():
                heatmap[b].scatter_add_(0, idx[b][m], attn_w[b][m])
        return heatmap

    def remove(self):
        self._handle.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap 생성
# ══════════════════════════════════════════════════════════════════════════════
def make_yolo_heatmap(feat: torch.Tensor) -> torch.Tensor:
    """
    YOLO feature map → 평균 제거 heatmap (Pearson Loss용)
    feat   : (B, C, H, W)
    return : (B, H*W) 평균 제거
    """
    B = feat.shape[0]
    heatmap = feat.pow(2).sum(dim=1).view(B, -1)
    return heatmap - heatmap.mean(dim=1, keepdim=True)


def make_rtdetr_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    CrossAttnWeightHook 출력 → 평균 제거 heatmap (Pearson Loss용)
    heatmap : (B, G*G)
    return  : (B, G*G) 평균 제거
    """
    return heatmap - heatmap.mean(dim=1, keepdim=True)


def pearson_loss(yolo: torch.Tensor, rtdetr: torch.Tensor) -> torch.Tensor:
    """
    Pearson Correlation Loss
    yolo, rtdetr : (B, N) 평균 제거된 heatmap
    return       : scalar  1 - cosine_similarity
                   0 = 완전히 같은 패턴
                   2 = 완전히 반대 패턴
    """
    return (1.0 - F.cosine_similarity(yolo, rtdetr, dim=1)).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 검증
# ══════════════════════════════════════════════════════════════════════════════
def validate(student: nn.Module, save_dir: Path, args) -> dict:
    """
    student state_dict만 임시 저장 후
    별도 YOLO 인스턴스로 검증 → deepcopy 없이 student 상태 완전 보호
    """
    tmp_path = save_dir / '_val_tmp.pt'
    tmp_model = DetectionModel('yolov8n.yaml', nc=80)
    tmp_model.load_state_dict(student.state_dict())
    torch.save({'model': tmp_model.half()}, tmp_path)
    del tmp_model

    yolo = YOLO(tmp_path)
    results = yolo.val(
        data=args.data,
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        verbose=False,
        plots=False,
    )
    tmp_path.unlink(missing_ok=True)
    del yolo

    return {
        'map50'    : results.results_dict.get('metrics/mAP50(B)',    0.0),
        'map50_95' : results.results_dict.get('metrics/mAP50-95(B)', 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(save_dir, student, optimizer, epoch, metrics, is_best):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch'    : epoch,
        'metrics'  : metrics,
        'model'    : {k: v.half() for k, v in student.state_dict().items()},
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

    # ★ 변경 2 : YOLO Hook 3개 등록
    # model[15] = C2f  P3 80×80 (소형 객체)
    # model[18] = C2f  P4 40×40 (중형 객체)
    # model[21] = C2f  P5 20×20 (대형 객체)
    hook_p3 = FeatureHook(student.model[15])
    hook_p4 = FeatureHook(student.model[18])
    hook_p5 = FeatureHook(student.model[21])

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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            images = batch['img'].float() / 255.0

            # ── RT-DETR forward → 크기별 Cross-Attention heatmap 캡처 ─────────
            with torch.no_grad():
                teacher(images)

            # ★ 변경 3 : 크기별 RT-DETR heatmap 생성
            rtdetr_small = make_rtdetr_heatmap(cross_attn_hook.heatmap_small)  # (B, 6400)
            rtdetr_mid   = make_rtdetr_heatmap(cross_attn_hook.heatmap_mid)    # (B, 1600)
            rtdetr_large = make_rtdetr_heatmap(cross_attn_hook.heatmap_large)  # (B,  400)

            # ── YOLOv8 forward → 각 스케일 feature 캡처 ──────────────────────
            preds = student(images)

            # ★ 변경 4 : 크기별 YOLO heatmap 생성
            yolo_small = make_yolo_heatmap(hook_p3.feat)   # (B, 6400) P3 80×80
            yolo_mid   = make_yolo_heatmap(hook_p4.feat)   # (B, 1600) P4 40×40
            yolo_large = make_yolo_heatmap(hook_p5.feat)   # (B,  400) P5 20×20

            # ── Detection Loss ────────────────────────────────────────────────
            det_total, _ = det_loss_fn(preds, batch)
            if det_total.dim() > 0:
                det_total = det_total.sum()

            # ★ 변경 5 : 크기별 Pearson Correlation Loss 합산
            # 스케일 무관, 패턴(상대적 위치 반응)만 비교
            # 소형 쿼리 → P3, 중형 쿼리 → P4, 대형 쿼리 → P5
            attn_loss = (
                pearson_loss(yolo_small, rtdetr_small.detach()) +
                pearson_loss(yolo_mid,   rtdetr_mid.detach())   +
                pearson_loss(yolo_large, rtdetr_large.detach())
            )

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

        val_metrics = validate(student, save_dir, args)
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
    hook_p3.remove()
    hook_p4.remove()
    hook_p5.remove()
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
    p.add_argument('--save-dir',      default='runs/distill_v3')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)