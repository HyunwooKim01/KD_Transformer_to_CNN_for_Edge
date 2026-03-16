# RT-DETR → YOLOv8-n Knowledge Distillation

## 버전 구조

| 버전 | 파일 | 증류 방식 | 손실함수 | Hook 위치 |
|---|---|---|---|---|
| v2 | `train_v2.py` | 단일 스케일 (P5 20×20) | Pearson | model[21] |
| v3 | `train_v3.py` | 멀티 스케일 (P3/P4/P5) | MSE | model[15/18/21] |
| v4 | `train_v4.py` | 멀티 스케일 (P3/P4/P5) | Pearson | model[15/18/21] |

---

## 디렉토리 구조

```
프로젝트 루트/
│
├── RT-DETR/                          ← lyuwenyu/RT-DETR 클론 (수정 없음)
│     └── rtdetr_pytorch/
│           └── src/zoo/rtdetr/
│                 └── rtdetr_decoder.py   ← Hook 위치
│
├── ultralytics/                      ← Ultralytics YOLOv8 클론 (수정 없음)
│
├── yolov8n.pt                        ← YOLOv8-n 사전학습 가중치
├── rtdetr_r101vd_6x_coco_from_paddle.pth  ← RT-DETR 사전학습 가중치
│
├── train_v2.py                       ★ 단일 스케일
├── train_v3.py                       ★ 멀티 스케일 / MSE
├── train_v4.py                       ★ 멀티 스케일 / Pearson
└── data.yml                          ← COCO 데이터셋 설정
```

---

## 증류 구조

### Teacher : RT-DETR Decoder Cross-Attention

```
Decoder layers[-1].cross_attn
        ↓
reference_points (B, 300, 1, 4)  cx, cy, w, h  (0~1 정규화)
attention_weights (B, 300)        쿼리별 반응 강도

쿼리 크기 분류 (w × h 기준):
  소형 (w*h < 0.001)  → 80×80 격자 투영  → heatmap_small (B, 6400)
  중형 (0.001~0.005)  → 40×40 격자 투영  → heatmap_mid   (B, 1600)
  대형 (>= 0.005)     → 20×20 격자 투영  → heatmap_large (B,  400)
```

### Student : YOLOv8-n Neck Feature Map

```
model[15] C2f → P3 (B,  64, 80, 80) → 채널 제곱합 → heatmap (B, 6400)
model[18] C2f → P4 (B, 128, 40, 40) → 채널 제곱합 → heatmap (B, 1600)
model[21] C2f → P5 (B, 256, 20, 20) → 채널 제곱합 → heatmap (B,  400)
```

---

## 손실함수

### v2 : 단일 스케일 Pearson

```
total_loss = det_loss + λ × pearson_loss(P5_heatmap, rtdetr_large_heatmap)
```

### v3 : 멀티 스케일 MSE

```
total_loss = det_loss + λ × (
    MSE(P3_heatmap, rtdetr_small_heatmap)
  + MSE(P4_heatmap, rtdetr_mid_heatmap)
  + MSE(P5_heatmap, rtdetr_large_heatmap)
)
```

### v4 : 멀티 스케일 Pearson

```
total_loss = det_loss + λ × (
    pearson_loss(P3_heatmap, rtdetr_small_heatmap)
  + pearson_loss(P4_heatmap, rtdetr_mid_heatmap)
  + pearson_loss(P5_heatmap, rtdetr_large_heatmap)
)

pearson_loss(x, y) = 1 - cosine_similarity(x - mean(x), y - mean(y))
```

**Pearson 선택 이유 :**
RT-DETR heatmap(sparse)과 YOLO heatmap(dense)의 스케일 차이를 평균 제거 후
패턴(상대적 반응 순위)만 비교하여 해결. MSE는 값 자체를 맞추려 해서
feature가 죽는 현상(under-activation) 발생 가능.

---

## 실행 방법

### v2 (단일 스케일 / Pearson)

```bash
python train_v2.py \
    --teacher-cfg  RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml \
    --teacher-ckpt rtdetr_r101vd_6x_coco_from_paddle.pth \
    --student-ckpt yolov8n.pt \
    --data         /home/user/git/Cross_Arch_Distillation/data.yml \
    --nc           80 \
    --epochs       100 \
    --batch        24 \
    --device       cuda:0 \
    --lambda-attn  50.0 \
    --save-dir     runs/distill_v2
```

### v3 (멀티 스케일 / MSE)

```bash
python train_v3.py \
    --teacher-cfg  RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml \
    --teacher-ckpt rtdetr_r101vd_6x_coco_from_paddle.pth \
    --student-ckpt yolov8n.pt \
    --data         /home/user/git/Cross_Arch_Distillation/data.yml \
    --nc           80 \
    --epochs       100 \
    --batch        24 \
    --device       cuda:0 \
    --lambda-attn  50.0 \
    --save-dir     runs/distill_v3
```

### v4 (멀티 스케일 / Pearson)

```bash
python train_v4.py \
    --teacher-cfg  RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml \
    --teacher-ckpt rtdetr_r101vd_6x_coco_from_paddle.pth \
    --student-ckpt yolov8n.pt \
    --data         /home/user/git/Cross_Arch_Distillation/data.yml \
    --nc           80 \
    --epochs       100 \
    --batch        24 \
    --device       cuda:0 \
    --lambda-attn  5.0 \
    --save-dir     runs/distill_v4
```

---

## 하이퍼파라미터

| 항목 | v2 | v3 | v4 |
|---|---|---|---|
| lambda-attn | 50.0 | 50.0 | 5.0 |
| loss | Pearson | MSE | Pearson |
| 스케일 | 단일 (P5) | 멀티 (P3/P4/P5) | 멀티 (P3/P4/P5) |
| batch | 24 | 24 | 24 |
| epochs | 100 | 100 | 100 |
| optimizer | AdamW | AdamW | AdamW |
| scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |

---

## 저장 파일

```
runs/distill_vN/
├── best.pt        ← best mAP 기준 Student 가중치
├── last.pt        ← 마지막 epoch 가중치
└── train_log.csv  ← epoch별 loss / mAP 기록
```

---

## 논문 인용

```bibtex
@inproceedings{zhao2024rtdetr,
  title={DETRs Beat YOLOs on Real-time Object Detection},
  author={Zhao, Yian and Lv, Wenyu and Xu, Shangliang and Wei, Jinman
          and Wang, Guanzhong and Dang, Qingqing and Liu, Yi and Chen, Jie},
  booktitle={CVPR},
  year={2024}
}
@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  year={2015}
}
@inproceedings{romero2015fitnets,
  title={FitNets: Hints for Thin Deep Nets},
  author={Romero, Adriana and Ballas, Nicolas and Kahou, Samira Ebrahimi
          and Chassang, Antoine and Gatta, Carlo and Bengio, Yoshua},
  booktitle={ICLR},
  year={2015}
}
@inproceedings{zagoruyko2017at,
  title={Paying More Attention to Attention},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle={ICLR},
  year={2017}
}
@inproceedings{carion2020detr,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel
          and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={ECCV},
  year={2020}
}
@inproceedings{zhu2021deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin
          and Wang, Xiaogang and Dai, Jifeng},
  booktitle={ICLR},
  year={2021}
}
@inproceedings{lin2014coco,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge
          and Hays, James and Perona, Pietro and Ramanan, Deva
          and Dollár, Piotr and Zitnick, C. Lawrence},
  booktitle={ECCV},
  year={2014}
}
```
