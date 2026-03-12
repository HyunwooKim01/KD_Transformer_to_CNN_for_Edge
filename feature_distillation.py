"""
feature_distillation.py

출처:
  - MSE Feature Loss  : FitNets (Romero et al., ICLR 2015)
  - AT Loss           : Paying More Attention to Attention (Zagoruyko & Komodakis, ICLR 2017)
  - FeatureAdapter    : ★ 새로 만드는 부분 ★ (RT-DETR S5 ↔ YOLOv8 P5 정렬)
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# ★ 새로 만드는 부분 ★
# FeatureAdapter : Student P5 → Teacher S5 채널/공간 크기 정렬
#   - 1x1 Conv + BN + ReLU : 채널 변환 (YOLOv8-n P5=256 → RT-DETR S5=384)
#   - bilinear interpolate  : 공간 크기 변환 (H_s,W_s → H_t,W_t)
# ────────────────────────────────────────────────────────────────
class FeatureAdapter(nn.Module):
    """
    Args:
        student_ch (int): YOLOv8-n P5 채널 수  (실측 256 : SPPF 출력)
        teacher_ch (int): RT-DETR   S5 채널 수  (실측 384 : AIFI 출력)
    """
    def __init__(self, student_ch: int = 256, teacher_ch: int = 384):
        super().__init__()
        # 1x1 Conv : 채널 변환  ← ★ 새로 만드는 부분
        self.conv = nn.Sequential(
            nn.Conv2d(student_ch, teacher_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, student_feat: torch.Tensor,
                target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            student_feat : (B, C_s, H_s, W_s)
            target_hw    : (H_t, W_t)  Teacher S5의 공간 크기
        Returns:
            adapted      : (B, C_t, H_t, W_t)
        """
        x = self.conv(student_feat)
        if target_hw is not None and x.shape[2:] != torch.Size(target_hw):
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x


# ────────────────────────────────────────────────────────────────
# AT Loss helper
# 출처: Zagoruyko & Komodakis, ICLR 2017
#   attention(F) = L2_normalize( sum_c( F_c^2 ) )
# ────────────────────────────────────────────────────────────────
def attention_map(feat: torch.Tensor) -> torch.Tensor:
    """
    feat : (B, C, H, W)
    return: (B, H*W)  L2-normalized spatial attention
    """
    # sum of squared activations over channel dim → (B, H, W)
    attn = feat.pow(2).sum(dim=1)               # (B, H, W)
    attn = attn.view(attn.size(0), -1)           # (B, H*W)
    return F.normalize(attn, p=2, dim=1)         # L2 normalize


# ────────────────────────────────────────────────────────────────
# Feature Distillation Loss
#   출처(프레임워크): Hinton et al., NeurIPS 2014
#   출처(MSE)      : Romero et al., ICLR 2015
#   출처(AT)       : Zagoruyko & Komodakis, ICLR 2017
# ────────────────────────────────────────────────────────────────
class FeatureDistillationLoss(nn.Module):
    """
    Loss = λ_mse * MSE(P5_adapted, S5)
         + λ_at  * MSE(AT(P5_adapted), AT(S5))

    Args:
        student_ch  (int)  : YOLOv8-n P5 채널
        teacher_ch  (int)  : RT-DETR   S5 채널
        lambda_mse  (float): MSE loss 가중치
        lambda_at   (float): AT  loss 가중치
    """
    def __init__(
        self,
        student_ch : int   = 256,
        teacher_ch : int   = 384,
        lambda_mse : float = 1.0,
        lambda_at  : float = 0.5,
    ):
        super().__init__()
        self.adapter    = FeatureAdapter(student_ch, teacher_ch)  # ★ 새로 만드는 부분
        self.lambda_mse = lambda_mse
        self.lambda_at  = lambda_at

    def forward(
        self,
        student_p5 : torch.Tensor,   # (B, 256, H_s, W_s)  YOLOv8 P5  (SPPF 출력)
        teacher_s5 : torch.Tensor,   # (B, 384, H_t, W_t)  RT-DETR AIFI 출력
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            distill_loss    : 총 증류 손실 (역전파용)
            mse_loss        : Feature MSE    (FitNets)
            at_loss         : Attention Transfer (AT)
            student_adapted : Adapter 통과 feature (시각화용)
        """
        target_hw = (teacher_s5.shape[2], teacher_s5.shape[3])

        # 1) Adapter : P5 → P5_adapted  (S5와 동일 shape)
        p5_adapted = self.adapter(student_p5, target_hw=target_hw)

        # 2) MSE Loss  (FitNets, Romero et al. 2015)
        mse_loss = F.mse_loss(p5_adapted, teacher_s5.detach())

        # 3) AT Loss  (Zagoruyko & Komodakis, 2017)
        at_s = attention_map(p5_adapted)
        at_t = attention_map(teacher_s5.detach())
        at_loss = F.mse_loss(at_s, at_t)

        distill_loss = self.lambda_mse * mse_loss + self.lambda_at * at_loss

        return {
            'distill_loss'    : distill_loss,
            'mse_loss'        : mse_loss,
            'at_loss'         : at_loss,
            'student_adapted' : p5_adapted,
        }