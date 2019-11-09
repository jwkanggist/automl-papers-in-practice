# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensemble

Paper Authors: Balaji Lakshminarayanan et al.

Affiliation: Google DeepMind

Publication: NIPS2017 ([`NIPS video`](https://www.facebook.com/nipsfoundation/videos/1554654864625747/), [`ArXiv URL`](https://arxiv.org/abs/1612.01474), [`Github URL`](https://github.com/Kyushik/Predictive-Uncertainty-Estimation-using-Deep-Ensemble))
- ref blog: https://tech.instacart.com/3-nips-papers-we-loved-befb39a75ec2

Update date: Marco @ 191103

---


1. Key Question and Hypothesis of Present Paper
- 딥러닝 모델은 아래에 너무 민감 
    - 데이터 셋의 통계량 차이 (valid. data가 train set의 distribution에서 약간이라도 벗어나는 경우 불확실성 급증!)
    - 초기값
    - 하이퍼 파라미터 변경

- 우리의 딥러닝 예측이 얼마나 불확실 한지 측정 할 수 있는가?
- "Can we know when our deep learning models are uncertain about their predictions?"
- 불확실성을 측정하고 제어 할 수 있는가?
- 데이터셋의 도메인 시프트가 있는 경우에도 불확실 성을 제거 할 수 있는가?


2. Main Contributions
- classification model 출력의 불확실성을 측정하는 방법을 제시 
- 측정한 불확실성 + 모델 앙상블을 이용해서 안정된 예측을 할 수 있는 방법 제시



3. Method Summary
- Explanation with figures

4. Dataset Specification


5. Experimental Result


