"""
Total Training Loss
= YOLOv8 Detection Loss  (box + cls + dfl)
+ Feature Distillation Loss (MSE + AT)

사용법:
    criterion = TotalDistillLoss(student_channels=512, teacher_channels=256)
    losses = criterion(
        student_preds=...,   # YOLOv8 head 출력
        targets=...,         # GT boxes/labels
        student_p5=...,      # YOLOv8 P5 feature
        teacher_s5=...,      # RT-DETR AIFI 출력 S5 (no_grad)
    )
"""

import torch
import torch.nn as nn
from feature_distillation import FeatureDistillationLoss


class TotalDistillLoss(nn.Module):
    """
    YOLOv8 Detection Loss + Feature Distillation Loss 통합.

    Args:
        student_channels (int)  : YOLOv8-n P5 채널 수  (기본 512)
        teacher_channels (int)  : RT-DETR S5 채널 수   (기본 256)
        lambda_mse       (float): MSE 증류 가중치
        lambda_at        (float): Attention Transfer 증류 가중치
        lambda_distill   (float): 전체 증류 손실 가중치 (detection loss 대비 스케일)
        yolo_loss_fn            : YOLOv8 공식 손실함수 인스턴스
                                  (ultralytics v8DetectionLoss 등 외부에서 주입)
    """

    def __init__(
        self,
        student_channels: int = 512,
        teacher_channels: int = 256,
        lambda_mse: float     = 1.0,
        lambda_at: float      = 0.5,
        lambda_distill: float = 0.5,
        yolo_loss_fn          = None,
    ):
        super().__init__()
        self.distill_loss_fn = FeatureDistillationLoss(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            lambda_mse=lambda_mse,
            lambda_at=lambda_at,
        )
        self.lambda_distill = lambda_distill
        self.yolo_loss_fn   = yolo_loss_fn   # 외부 주입 (None이면 detection 스킵)

    def forward(
        self,
        student_p5: torch.Tensor,            # YOLOv8 Backbone P5
        teacher_s5: torch.Tensor,            # RT-DETR AIFI 출력 (detach 필요)
        student_preds = None,                # YOLOv8 Head 출력 (detection loss용)
        targets       = None,                # GT labels (detection loss용)
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                'total_loss'   : 최종 역전파 손실
                'det_loss'     : YOLOv8 detection 손실 (없으면 0)
                'distill_loss' : 증류 손실 합계
                'mse_loss'     : Feature MSE
                'at_loss'      : Attention Transfer
        """
        # ── 1) Detection Loss ──────────────────────────────────────────────
        if self.yolo_loss_fn is not None and student_preds is not None:
            det_loss = self.yolo_loss_fn(student_preds, targets)
            # ultralytics 반환값이 tuple인 경우 첫 번째 원소 (sum)
            if isinstance(det_loss, (tuple, list)):
                det_loss = det_loss[0]
        else:
            det_loss = torch.tensor(0.0, device=student_p5.device)

        # ── 2) Feature Distillation Loss ───────────────────────────────────
        # Teacher feature는 gradient 불필요 → detach 보장
        distill_out = self.distill_loss_fn(
            student_p5=student_p5,
            teacher_s5=teacher_s5.detach(),
        )

        # ── 3) Total Loss ──────────────────────────────────────────────────
        total_loss = det_loss + self.lambda_distill * distill_out["distill_loss"]

        return {
            "total_loss"   : total_loss,
            "det_loss"     : det_loss,
            "distill_loss" : distill_out["distill_loss"],
            "mse_loss"     : distill_out["mse_loss"],
            "at_loss"      : distill_out["at_loss"],
        }