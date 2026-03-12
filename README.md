# KD_Transformer_to_CNN_Object_Detection

Edge 환경에서 객체 탐지를 위한 Knowledge Distillation 기반 경량 모델 연구

1. 연구 배경 및 목적
최근 객체 탐지 모델들은 높은 정확도를 가지고 있지만, 대부분의 고성능 모델들을 계산량이 많아 한정된 자원을 가진 Edge device에서의 실시간 적용이 어렵다
본 연구에서는 Knowledge Distillation 기법을 활용하여 대형 객체 탐지 모델의 지식을 경량 모델로 전달하고, Edge 환경에서도 높은 정확도를 유지하면서 실시간 객체 탐지가 가능하도록 하는 것을 목표로 한다.

2. 연구 목표
High-capacity Detection Model --(KD)--> Light-weight Detection Model --> Edge Device Model
고성능 객체 탐지 모델에서 지식 증류로 경량모델에 전달한 뒤, Edge에서의 적용을 목표로 한다.

3. 데이터셋
초기단계에서는 대규모 객체 탐지 데이터 셋인 COCO Dataset을 사용한다.

4. 모델 구성
Teacher Model : RT-DETR
선정 이유 : 요즘 핫한 Transformer기반 객체 탐지 모델이고, Attention으로 이미지의 전역 문맥을 잘 파악할 수 있기 때문에 Student Model에 풍부한 지식을 증류할 수 있을거라고 생각.
Student Model : YOLOv8-n
선정이유 : Edge 환경에서 중요한 추론 속도가 빠르고, Edge에서 많이 사용되는 경량 모델이므로, 지식 증류를 했을때에 Edge에서 얼마나 더 효과적으로 발전하는지 보여주기에 좋다고 생각했다.
