"""
train.py  —  RT-DETR → YOLOv8-n Knowledge Distillation 학습 루프
================================================================
출처 및 새로 만드는 부분 표기:
  [RT-DETR]     lyuwenyu/RT-DETR 공식 코드
  [Ultralytics] Ultralytics YOLOv8 공식 코드
  [Hinton2014]  Distilling the Knowledge in a Neural Network
  [Romero2015]  FitNets: Hints for Thin Deep Nets
  [AT2017]      Paying More Attention to Attention
  [AdamW]       Decoupled Weight Decay Regularization (Loshchilov 2019)
  ★             새로 만드는 부분
"""

import sys
import os
import math
import time
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW                                        # [AdamW]
from torch.optim.lr_scheduler import CosineAnnealingLR              # PyTorch 공식

# ── RT-DETR ──────────────────────────────────────────────────────────────────
# [RT-DETR] lyuwenyu/RT-DETR 공식 코드
sys.path.insert(0, 'RT-DETR/rtdetr_pytorch')
from src.core import YAMLConfig                                      # [RT-DETR]

# ── Ultralytics ───────────────────────────────────────────────────────────────
# [Ultralytics] 공식 코드
from ultralytics.nn.tasks import DetectionModel                      # [Ultralytics]
from ultralytics.data.build import build_dataloader                  # [Ultralytics]
from ultralytics.utils.loss import v8DetectionLoss                   # [Ultralytics]
from ultralytics.utils.torch_utils import ModelEMA                   # [Ultralytics]

# de_parallel : Ultralytics 버전에 따라 없을 수 있어 직접 구현  ★
def de_parallel(model):
    """DDP/DataParallel 래퍼 벗기기. 없으면 그대로 반환."""
    return model.module if hasattr(model, 'module') else model

# ── 새로 만드는 부분 ★ ────────────────────────────────────────────────────────
from feature_distillation import FeatureDistillationLoss             # ★
import loss_patch  # noqa: F401  monkey-patch 적용                   # ★


# ════════════════════════════════════════════════════════════════════════════════
# ★ Hook 클래스  (새로 만드는 부분)
# ════════════════════════════════════════════════════════════════════════════════

class TeacherS5Hook:
    """
    RT-DETR HybridEncoder.encoder[0] (AIFI) 출력을 캡처.
    ★ 새로 만드는 부분 ★

    Hook 위치:
        teacher_model.encoder  = HybridEncoder          [RT-DETR]
        teacher_model.encoder.encoder[0] = AIFI (TransformerEncoder)  [RT-DETR]

    AIFI 출력 shape:
        (B, N, C)  시퀀스 형태 → (B, C, H, W) 로 reshape
    """
    def __init__(self, module: nn.Module):
        self.feat: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if out.dim() == 3:                    # (B, N, C) → (B, C, H, W)
            B, N, C = out.shape
            H = W = int(N ** 0.5)
            out = out.permute(0, 2, 1).reshape(B, C, H, W)
        self.feat = out.detach()              # Teacher는 grad 불필요

    def remove(self):
        self._handle.remove()


class StudentP5Hook:
    """
    YOLOv8-n backbone C2f (stride-32, model[9]) 출력을 캡처.
    ★ 새로 만드는 부분 ★

    Hook 위치:
        student_model.model[9]  = C2f  (P5, stride-32)  [Ultralytics]
    """
    def __init__(self, module: nn.Module):
        self.feat: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.feat = output                    # Student는 grad 유지

    def remove(self):
        self._handle.remove()


# ════════════════════════════════════════════════════════════════════════════════
# ★ 모델 로드 함수  (새로 만드는 부분)
# ════════════════════════════════════════════════════════════════════════════════

def load_teacher(cfg_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    """
    RT-DETR-R101 로드 및 추론 전용 설정.
    모델 로드: [RT-DETR] 공식 코드
    eval/freeze 설정: ★ 새로 만드는 부분 ★
    """
    # [RT-DETR] YAMLConfig로 모델 빌드
    cfg   = YAMLConfig(cfg_path, resume=ckpt_path)
    model = cfg.model.deploy()               # [RT-DETR]

    # ★ pth 파일 키 자동 감지 ★
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f'  체크포인트 키 목록: {list(ckpt.keys()) if isinstance(ckpt, dict) else "state_dict 직접"}')
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get('model') or
            ckpt.get('state_dict') or
            ckpt.get('ema') or
            ckpt.get('params') or
            ckpt
        )
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)

    # ★ 추론 전용 설정 ★
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


def load_student(ckpt_path: str, nc: int, device: torch.device) -> nn.Module:
    """
    YOLOv8-n 로드.
    모델 구조: [Ultralytics] DetectionModel
    가중치 로드: ★ 새로 만드는 부분 ★
    """
    # [Ultralytics] DetectionModel
    model = DetectionModel('yolov8n.yaml', nc=nc)
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    return model.to(device).train()


# ════════════════════════════════════════════════════════════════════════════════
# ★ 검증 함수  (새로 만드는 부분)
# ════════════════════════════════════════════════════════════════════════════════

def validate(student_model: nn.Module, data_cfg: dict, args) -> dict:
    """
    증류 없이 YOLOv8-n 단독으로 val set 평가.
    Ultralytics 공식 DetectionValidator 사용  [Ultralytics]
    ★ 새로 만드는 부분 ★

    Returns:
        {'map50': float, 'map50_95': float}
    """
    from ultralytics.models.yolo.detect.val import DetectionValidator   # [Ultralytics]

    validator = DetectionValidator(
        args=dict(
            data   = args.data,
            imgsz  = 640,
            batch  = args.batch,
            device = args.device,
            workers= args.workers,
            verbose= False,
        )
    )
    validator.training = False
    results = validator(model=student_model)

    return {
        'map50'    : float(results.results_dict.get('metrics/mAP50(B)',    0.0)),
        'map50_95' : float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
    }


# ════════════════════════════════════════════════════════════════════════════════
# ★ 체크포인트 저장  (새로 만드는 부분)
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    save_dir   : Path,
    student    : nn.Module,
    adapter    : nn.Module,
    optimizer  : torch.optim.Optimizer,
    epoch      : int,
    metrics    : dict,
    is_best    : bool = False,
) -> None:
    """
    Student 가중치 + Adapter 가중치 함께 저장.
    ★ 새로 만드는 부분 ★
    (Adapter를 저장하지 않으면 추론 시 P5 크기 불일치 발생)
    """
    ckpt = {
        'epoch'     : epoch,
        'metrics'   : metrics,
        'model'     : deepcopy(de_parallel(student)).half(),     # [Ultralytics]
        'adapter'   : deepcopy(adapter).half(),                  # ★
        'optimizer' : optimizer.state_dict(),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_dir / 'last.pt')
    if is_best:
        torch.save(ckpt, save_dir / 'best.pt')
        print(f'  → best.pt 저장  mAP50={metrics["map50"]:.4f}')


# ════════════════════════════════════════════════════════════════════════════════
# ★ 메인 학습 루프  (새로 만드는 부분)
# ════════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── 기본 설정 ──────────────────────────────────────────────────────────────
    device   = torch.device(args.device)
    save_dir = Path(args.save_dir)
    best_map = 0.0

    # ── 로그 파일 ──────────────────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(save_dir / 'train_log.csv', 'w')
    log_file.write('epoch,total_loss,det_loss,distill_loss,mse_loss,at_loss,map50,map50_95\n')

    # ──────────────────────────────────────────────────────────────────────────
    # [1] 모델 로드
    # ──────────────────────────────────────────────────────────────────────────
    print('[1] 모델 로드')

    # Teacher : RT-DETR-R101  [RT-DETR] + ★
    teacher = load_teacher(args.teacher_cfg, args.teacher_ckpt, device)
    print(f'  Teacher RT-DETR-R101 로드 완료')

    # Student : YOLOv8-n  [Ultralytics] + ★
    student = load_student(args.student_ckpt, args.nc, device)
    print(f'  Student YOLOv8-n 로드 완료')

    # ──────────────────────────────────────────────────────────────────────────
    # [2] Hook 등록  ★ 새로 만드는 부분 ★
    # ──────────────────────────────────────────────────────────────────────────
    print('[2] Hook 등록')

    # Teacher : HybridEncoder.encoder[0] = AIFI  [RT-DETR] 구조 기반 ★
    teacher_hook = TeacherS5Hook(teacher.encoder.encoder[0])

    # Student : model[9] = C2f (P5, stride-32)  [Ultralytics] 구조 기반 ★
    student_hook = StudentP5Hook(student.model[9])

    # ──────────────────────────────────────────────────────────────────────────
    # [3] Distillation Loss + Adapter 초기화  ★ 새로 만드는 부분 ★
    # ──────────────────────────────────────────────────────────────────────────
    print('[3] Distillation Loss 초기화')
    distill_loss_fn = FeatureDistillationLoss(
        student_ch = args.student_ch,   # YOLOv8-n P5 채널 (기본 512)
        teacher_ch = args.teacher_ch,   # RT-DETR   S5 채널 (기본 256)
        lambda_mse = args.lambda_mse,   # [Romero2015]
        lambda_at  = args.lambda_at,    # [AT2017]
    ).to(device)

    # Detection Loss  [Ultralytics] 원본 + ★ monkey-patch 적용
    # v8DetectionLoss는 model.args 하이퍼파라미터를 요구함 → 직접 주입  ★
    from types import SimpleNamespace
    if not hasattr(student, 'args'):
        student.args = SimpleNamespace(
            box=7.5, cls=0.5, dfl=1.5,
            fl_gamma=0.0, label_smoothing=0.0,
        )
    det_loss_fn = v8DetectionLoss(student)

    # ──────────────────────────────────────────────────────────────────────────
    # [4] Optimizer  [AdamW] + ★
    # Adapter 파라미터도 반드시 함께 등록
    # ──────────────────────────────────────────────────────────────────────────
    print('[4] Optimizer / Scheduler 설정')
    params = (
        list(student.parameters()) +
        list(distill_loss_fn.parameters())   # ★ Adapter 파라미터 포함
    )
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)   # [AdamW]

    # Cosine LR Scheduler  [PyTorch 공식]
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # EMA  [Ultralytics]
    ema = ModelEMA(student)

    # ──────────────────────────────────────────────────────────────────────────
    # [5] DataLoader  [Ultralytics]
    # build_yolo_dataset으로 Dataset 먼저 생성 후 build_dataloader에 전달  ★
    # ──────────────────────────────────────────────────────────────────────────
    print('[5] DataLoader 빌드')
    from ultralytics.data.build import build_yolo_dataset          # [Ultralytics]
    from ultralytics.utils import IterableSimpleNamespace          # [Ultralytics]
    from ultralytics.data.utils import check_det_dataset           # [Ultralytics]

    # data.yml 로드  [Ultralytics]
    data_cfg = check_det_dataset(args.data)

    # default.yaml 에서 전체 하이퍼파라미터 로드 후 필요한 값만 덮어씀  ★
    from ultralytics.cfg import get_cfg                             # [Ultralytics]
    train_cfg = get_cfg()          # default.yaml 전체 로드
    train_cfg.imgsz     = 640
    train_cfg.rect      = False
    train_cfg.cache     = False
    train_cfg.single_cls = False
    train_cfg.fraction  = 1.0
    train_cfg.task      = 'detect'
    train_cfg.classes   = None

    val_cfg = get_cfg()
    val_cfg.imgsz      = 640
    val_cfg.rect       = True
    val_cfg.cache      = False
    val_cfg.single_cls = False
    val_cfg.fraction   = 1.0
    val_cfg.task       = 'detect'
    val_cfg.classes    = None

    train_dataset = build_yolo_dataset(
        cfg      = train_cfg,
        img_path = data_cfg['train'],
        batch    = args.batch,
        data     = data_cfg,
        mode     = 'train',
        stride   = 32,
    )
    val_dataset = build_yolo_dataset(
        cfg      = val_cfg,
        img_path = data_cfg['val'],
        batch    = args.batch,
        data     = data_cfg,
        mode     = 'val',
        stride   = 32,
    )

    train_loader = build_dataloader(
        dataset    = train_dataset,
        batch      = args.batch,
        workers    = args.workers,
        shuffle    = True,
        rank       = -1,
    )                                                              # [Ultralytics]

    val_loader = build_dataloader(
        dataset    = val_dataset,
        batch      = args.batch,
        workers    = args.workers,
        shuffle    = False,
        rank       = -1,
    )                                                              # [Ultralytics]

    # ──────────────────────────────────────────────────────────────────────────
    # [6] Epoch 루프  ★ 새로 만드는 부분 ★
    # ──────────────────────────────────────────────────────────────────────────
    print(f'\n[6] 학습 시작  (epochs={args.epochs})\n{"─"*70}')

    for epoch in range(args.epochs):
        student.train()
        teacher.eval()      # Teacher는 항상 eval

        epoch_metrics = {k: 0.0 for k in
                         ('total_loss', 'det_loss', 'distill_loss', 'mse_loss', 'at_loss')}
        t0 = time.time()

        # ── Batch 루프 ────────────────────────────────────────────────────────
        for batch_idx, batch in enumerate(train_loader):
            images  = batch['img'].to(device).float() / 255.0
            targets = batch                                      # [Ultralytics] 형식

            # ── [4-1] Teacher Forward  [RT-DETR] + ★ ─────────────────────────
            with torch.no_grad():
                teacher(images)                # forward 실행 → hook이 S5 캡처
            s5 = teacher_hook.feat             # (B, 256, H_t, W_t)  ★

            # ── [4-2] Student Forward  [Ultralytics] + ★ ─────────────────────
            preds = student(images)            # forward → hook이 P5 캡처
            p5    = student_hook.feat          # (B, 512, H_s, W_s)  ★

            # ── [4-3] Detection Loss  [Ultralytics] ──────────────────────────
            det_total, det_items = det_loss_fn(preds, targets)
            # v8DetectionLoss 반환값이 vector일 수 있으므로 scalar로 변환  ★
            if det_total.dim() > 0:
                det_total = det_total.sum()

            # ── [4-4] Distillation Loss  ★ ([Romero2015] + [AT2017] 기반) ────
            distill_out = distill_loss_fn(student_p5=p5, teacher_s5=s5)
            distill_loss = distill_out['distill_loss']

            # ── [4-5] Total Loss  ★ ──────────────────────────────────────────
            total_loss = det_total + args.lambda_distill * distill_loss

            # ── [4-6] Backward  [PyTorch] ─────────────────────────────────────
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient Clipping  (RT-DETR 논문 설정 max_norm=0.1)  [RT-DETR]
            nn.utils.clip_grad_norm_(params, max_norm=0.1)

            optimizer.step()
            ema.update(student)                                  # [Ultralytics]

            # ── 배치 로그 ─────────────────────────────────────────────────────
            for k, v in (
                ('total_loss',   total_loss.item()),
                ('det_loss',     det_total.item()),
                ('distill_loss', distill_loss.item()),
                ('mse_loss',     distill_out['mse_loss'].item()),
                ('at_loss',      distill_out['at_loss'].item()),
            ):
                epoch_metrics[k] += v

            if batch_idx % 50 == 0:
                print(
                    f'  epoch {epoch+1:3d} | batch {batch_idx:4d} | '
                    f'total={total_loss.item():.4f}  '
                    f'det={det_total.item():.4f}  '
                    f'distill={distill_loss.item():.4f}  '
                    f'(mse={distill_out["mse_loss"].item():.4f}, '
                    f'at={distill_out["at_loss"].item():.4f})'
                )

        # ── LR Scheduler Step  [PyTorch] ─────────────────────────────────────
        scheduler.step()

        # ── Epoch 평균 ───────────────────────────────────────────────────────
        n = len(train_loader)
        for k in epoch_metrics:
            epoch_metrics[k] /= n

        # ──────────────────────────────────────────────────────────────────────
        # [7] Validation  ★ 새로 만드는 부분 ★
        # 증류 없이 YOLO 단독 평가 → 공정한 성능 측정
        # ──────────────────────────────────────────────────────────────────────
        val_metrics = validate(ema.ema, data_cfg, args)

        elapsed = time.time() - t0
        print(
            f'\n[Epoch {epoch+1}/{args.epochs}] '
            f'{elapsed:.1f}s | '
            f'mAP50={val_metrics["map50"]:.4f}  '
            f'mAP50-95={val_metrics["map50_95"]:.4f}\n'
        )

        # ── CSV 로그 ─────────────────────────────────────────────────────────
        log_file.write(
            f'{epoch+1},'
            f'{epoch_metrics["total_loss"]:.6f},'
            f'{epoch_metrics["det_loss"]:.6f},'
            f'{epoch_metrics["distill_loss"]:.6f},'
            f'{epoch_metrics["mse_loss"]:.6f},'
            f'{epoch_metrics["at_loss"]:.6f},'
            f'{val_metrics["map50"]:.6f},'
            f'{val_metrics["map50_95"]:.6f}\n'
        )
        log_file.flush()

        # ──────────────────────────────────────────────────────────────────────
        # [8] 체크포인트 저장  ★ 새로 만드는 부분 ★
        # Student + Adapter 함께 저장
        # ──────────────────────────────────────────────────────────────────────
        is_best = val_metrics['map50'] > best_map
        if is_best:
            best_map = val_metrics['map50']

        save_checkpoint(
            save_dir  = save_dir,
            student   = ema.ema,
            adapter   = distill_loss_fn.adapter,   # ★ Adapter 별도 저장
            optimizer = optimizer,
            epoch     = epoch,
            metrics   = {**epoch_metrics, **val_metrics},
            is_best   = is_best,
        )

    # ── Hook 해제  ★ ──────────────────────────────────────────────────────────
    teacher_hook.remove()
    student_hook.remove()
    log_file.close()

    print(f'\n학습 완료. best mAP50={best_map:.4f}  저장 위치={save_dir}')


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='RT-DETR → YOLOv8-n Knowledge Distillation')

    # 모델 경로
    p.add_argument('--teacher-cfg',   type=str, required=True,
                   help='RT-DETR-R101 config yaml 경로  (RT-DETR/rtdetr_pytorch/configs/)')
    p.add_argument('--teacher-ckpt',  type=str, required=True,
                   help='RT-DETR-R101 사전학습 가중치 .pth 경로')
    p.add_argument('--student-ckpt',  type=str, default='yolov8n.pt',
                   help='YOLOv8-n 사전학습 가중치 경로')

    # 데이터
    p.add_argument('--data',     type=str, required=True,   help='데이터셋 yaml 경로')
    p.add_argument('--nc',       type=int, default=80,      help='클래스 수')

    # 채널 수  (실측값 기준)
    p.add_argument('--student-ch', type=int, default=256,   help='YOLOv8-n P5 채널 (SPPF 출력 실측 256)')
    p.add_argument('--teacher-ch', type=int, default=384,   help='RT-DETR   S5 채널 (AIFI 출력 실측 384)')

    # 학습 하이퍼파라미터
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch',          type=int,   default=16)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--weight-decay',   type=float, default=1e-4)
    p.add_argument('--workers',        type=int,   default=8)
    p.add_argument('--device',         type=str,   default='cuda:0')

    # 증류 가중치
    p.add_argument('--lambda-distill', type=float, default=0.5,
                   help='전체 distill loss 가중치')
    p.add_argument('--lambda-mse',     type=float, default=1.0,
                   help='MSE loss 가중치  [Romero2015]')
    p.add_argument('--lambda-at',      type=float, default=0.5,
                   help='AT  loss 가중치  [AT2017]')

    # 저장
    p.add_argument('--save-dir', type=str, default='runs/distill')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)