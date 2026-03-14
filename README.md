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

선정 이유 : Transformer 기반 객체 탐지 모델로, Decoder의 Cross-Attention을 통해 이미지의 전역 문맥(Global Context)을 파악하는 능력이 뛰어나다. 특히 Decoder 마지막 레이어의 Cross-Attention weight는 300개의 쿼리가 이미지 내 어느 위치를 참조했는가를 나타내며, 이를 공간적 heatmap으로 변환하면 "전역 문맥으로 판단한 물체가 있을 위치"를 효과적으로 표현할 수 있다.

### Student Model : YOLOv8-n

선정 이유 : Edge 환경에서 중요한 추론 속도가 빠르고, Edge에서 많이 사용되는 경량 모델이다. 지식 증류 적용 전후의 성능 변화를 통해 Edge 환경에서의 효용성을 검증하기에 적합하다.

---

## 5. 연구 진행 방법

```
COCO Dataset
     │
     ├──→ RT-DETR-R101 (Teacher, 고정)
     │         │
     │    Decoder Cross-Attention
     │    (layers[-1].cross_attn)
     │         │
     │    reference_points (B, 300, 1, 4)  ← 각 쿼리의 기준 좌표 (cx, cy)
     │    attention_weights (B, 300, heads, levels*points)  ← 각 쿼리의 반응 강도
     │         │
     │    20×20 격자에 투영 (scatter_add)
     │         │
     │    RT-DETR Heatmap (B, 400)
     │    "전역 문맥으로 물체가 있을 공간적 위치"
     │         │
     │         ▼
     └──→ YOLOv8-n (Student, 학습)
               │
           model[21] C2f
           (Neck 끝, Detect 직전)
               │
           채널 방향 제곱합 + 최대값 정규화
               │
           YOLO Heatmap (B, 400)
           "CNN이 탐지 직전 반응한 공간적 위치"
               │
         Attention Loss (MSE)
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

### Attention Map Distillation

RT-DETR Decoder의 Cross-Attention weight를 공간적 heatmap으로 변환하여 YOLOv8-n의 Neck 출력과 정렬한다.

**RT-DETR heatmap 생성**
- Hook 위치 : `model.decoder.decoder.layers[-1].cross_attn`
- `reference_points` (B, 300, 1, 4) 에서 cx, cy 추출
- `attention_weights` Linear로 각 쿼리의 반응 강도 계산
- cx, cy를 20×20 격자에 투영 후 반응 강도 누적 → (B, 400)
- 최대값 정규화 → 0~1 범위

**YOLO heatmap 생성**
- Hook 위치 : `student.model[21]` (C2f, Neck 끝 Detect 직전)
- Feature Map (B, 256, 20, 20) → 채널 방향 제곱합 → (B, 400)
- 최대값 정규화 → 0~1 범위

---

## 7. 손실함수 구성

```
total_loss = det_loss + λ × attention_loss

det_loss      = box_loss + cls_loss + dfl_loss   # YOLOv8 기존 손실함수

attention_loss = MSE(YOLO_heatmap, RTDETR_heatmap)
                 "RT-DETR이 전역 문맥으로 물체를 찾은 위치를
                  YOLO가 탐지 직전에 같은 위치에 집중하도록 학습"
```

| 하이퍼파라미터 | 값 | 설명 |
|---|---|---|
| λ | 50.0 | attention_loss 가중치 |

λ=50은 AT Loss의 스케일이 det_loss 대비 매우 작기 때문에 균형을 맞추기 위해 설정하였다. 이는 AT 논문(Zagoruyko 2017)에서 β=1000을 사용한 것과 동일한 이유이며, attention_loss가 total_loss의 약 3~5%를 기여하도록 조정하였다.

---

## 8. 구현 구조

```
Cross_Arch_Distillation/
├── RT-DETR/                          # lyuwenyu/RT-DETR 공식 코드 (수정 없음)
├── ultralytics/                      # Ultralytics YOLOv8 공식 코드 (수정 없음)
├── train_v2.py                       # 통합 학습 루프 (단일 파일)
├── data.yml                          # COCO 데이터셋 설정
├── yolov8n.pt                        # Student 초기 가중치
└── rtdetr_r101vd_6x_coco_from_paddle.pth  # Teacher 사전학습 가중치
```

---

## 9. 학습 실행

```bash
python train_v2.py \
    --teacher-cfg  RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml \
    --teacher-ckpt rtdetr_r101vd_6x_coco_from_paddle.pth \
    --student-ckpt yolov8n.pt \
    --data         /home/user/git/Cross_Arch_Distillation/data.yml \
    --nc           80 \
    --epochs       100 \
    --batch        16 \
    --device       cuda:0 \
    --lambda-attn  50.0 \
    --save-dir     runs/distill_final
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
@inproceedings{carion2020detr,
  title={End-to-End Object Detection with Transformers},
  author={Carion et al.},
  booktitle={ECCV},
  year={2020}
}
```











