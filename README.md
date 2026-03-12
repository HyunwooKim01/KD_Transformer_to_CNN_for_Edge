# KD_Transformer_to_CNN_for_Edge
Edge 환경에서 객체 탐지를 위한 Knowledge Distillation 기반 경량 모델 연구

---

## 1. 연구 배경 및 목적

최근 객체 탐지 모델들은 높은 정확도를 가지고 있지만, 대부분의 고성능 모델들은 계산량이 많아 한정된 자원을 가진 Edge Device에서의 실시간 적용이 어렵다.

본 연구에서는 Knowledge Distillation 기법을 활용하여 대형 객체 탐지 모델의 지식을 경량 모델로 전달하고, Edge 환경에서도 높은 정확도를 유지하면서 실시간 객체 탐지가 가능하도록 하는 것을 목표로 한다.

---

## 2. 연구 목표

```
High-capacity Detection Model --(KD)--> Light-weight Detection Model --> Edge Device
      RT-DETR-R101                           YOLOv8-n                    Jetson
```

고성능 객체 탐지 모델에서 지식 증류로 경량 모델에 전달한 뒤, Edge에서의 실시간 적용을 목표로 한다.

---

## 3. 데이터셋

초기 단계에서는 대규모 객체 탐지 데이터셋인 COCO Dataset을 사용한다.

- 학습 : train2017 (118,287장)
- 검증 : val2017 (5,000장)
- 클래스 수 : 80

---

## 4. 모델 구성

### Teacher Model : RT-DETR-R101

선정 이유 : Transformer 기반 객체 탐지 모델로, Self-Attention을 통해 이미지의 전역 문맥(Global Context)을 파악하는 능력이 뛰어나다. 특히 AIFI(Attention-based Intra-scale Feature Interaction) 모듈을 통해 생성된 S5 Feature Map은 풍부한 의미론적 정보를 담고 있어 Student Model에 효과적인 지식을 제공할 수 있다고 판단하였다.

### Student Model : YOLOv8-n

선정 이유 : Edge 환경에서 중요한 추론 속도가 빠르고, Edge에서 많이 사용되는 경량 모델이다. 지식 증류 적용 전후의 성능 변화를 통해 Edge 환경에서의 효용성을 검증하기에 적합하다.

---

## 5. 연구 진행 방법

```
COCO Dataset
     │
     ├──→ RT-DETR-R101 (Teacher, 고정)
     │         │
     │       AIFI
     │         │
     │      S5 Feature Map (384ch)
     │         │
     │         ▼
     └──→ YOLOv8-n (Student, 학습)
               │
             SPPF
               │
            P5 Feature Map (256ch)
               │
           FeatureAdapter (1×1 Conv)
               │
          P5_adapted (384ch)
               │
    ┌──────────┴──────────┐
    │                     │
MSE Loss              AT Loss
(Feature 정렬)     (Attention 정렬)
    │                     │
    └──────────┬──────────┘
               │
         distill_loss
               │
          + det_loss (box + cls + dfl)
               │
          total_loss
               │
          backward()
               │
     YOLOv8-n 가중치 업데이트
```

---

## 6. 지식 증류 방법

### Feature-level Distillation

RT-DETR의 AIFI를 통과한 S5 Feature Map을 YOLOv8-n의 P5에 증류한다.

- Teacher : `encoder.encoder[0]` (AIFI) 출력 S5 → shape : `(B, 384, H, W)`
- Student : `model[9]` (SPPF) 출력 P5 → shape : `(B, 256, H, W)`
- FeatureAdapter : 1×1 Conv로 Student P5의 채널을 Teacher S5에 맞게 변환 (256 → 384)

## 7. 손실함수 구성

```
total_loss   = det_loss + λ_distill × distill_loss

det_loss     = box_loss + cls_loss + dfl_loss        # YOLOv8 기존 손실함수

distill_loss = λ_mse × MSE(P5_adapted, S5)          # Feature MSE (FitNets, Romero 2015)
             + λ_at  × AT(P5_adapted, S5)            # Attention Transfer (Zagoruyko 2017)
```

| 하이퍼파라미터 | 기본값 | 설명 |
|---|---|---|
| λ_distill | 0.5 | 전체 증류 손실 가중치 |
| λ_mse | 1.0 | Feature MSE 가중치 |
| λ_at | 0.5 | Attention Transfer 가중치 |

---

## 8. 구현 구조

```
Cross_Arch_Distillation/
├── RT-DETR/                          # lyuwenyu/RT-DETR 공식 코드 (수정 없음)
├── ultralytics/                      # Ultralytics YOLOv8 공식 코드 (수정 없음)
├── feature_distillation.py           # FeatureAdapter + MSE/AT Loss
├── loss_patch.py                     # v8DetectionLoss monkey-patch
├── train.py                          # 통합 학습 루프
├── data.yml                          # COCO 데이터셋 설정
├── yolov8n.pt                        # Student 초기 가중치
└── rtdetr_r101vd_6x_coco_from_paddle.pth  # Teacher 사전학습 가중치
```

---

## 9. 학습 실행

```bash
python train.py \
    --teacher-cfg  RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml \
    --teacher-ckpt rtdetr_r101vd_6x_coco_from_paddle.pth \
    --student-ckpt yolov8n.pt \
    --data         data.yml \
    --nc           80 \
    --epochs       100 \
    --batch        16 \
    --device       cuda:0 \
    --lambda-distill 0.5 \
    --lambda-mse     1.0 \
    --lambda-at      0.5 \
    --save-dir     runs/distill
```

---

## 10. 참고 논문

```bibtex
@article{zhao2023rtdetr,
  title={DETRs Beat YOLOs on Real-time Object Detection},
  author={Zhao et al.},
  year={2023}
}
@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton et al.},
  year={2015}
}
@inproceedings{romero2015fitnets,
  title={FitNets: Hints for Thin Deep Nets},
  author={Romero et al.},
  booktitle={ICLR},
  year={2015}
}
@inproceedings{zagoruyko2017at,
  title={Paying More Attention to Attention},
  author={Zagoruyko and Komodakis},
  booktitle={ICLR},
  year={2017}
}
```











