# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensemble

Paper Authors: Balaji Lakshminarayanan et al.

Affiliation: Google DeepMind

Publication: NIPS2017 ([`NIPS video`](https://www.facebook.com/nipsfoundation/videos/1554654864625747/), [`ArXiv URL`](https://arxiv.org/abs/1612.01474), [`Github URL`](https://github.com/Kyushik/Predictive-Uncertainty-Estimation-using-Deep-Ensemble))
- ref blog: https://tech.instacart.com/3-nips-papers-we-loved-befb39a75ec2

Update date: Marco @ 191112

---
> 오타주의 (발표자에게는 관대합시다) 

### 1. Key Question and Hypothesis of Present Paper
- 딥러닝 모델은 아래에 너무 민감 하고 같은 모델 + 데이터라 하더라도 예측이 달라질 수 있음
    - seen data 와 unseen data
    - weight 초기값 
    - 하이퍼 파라미터 변경

- Predictive uncertainty estimation
```
Simple and Scalable Predictive Uncertainty Estimation using Deep Ensemble
```
    - 딥러닝 예측이 얼마나 확실한지 측정 할 수 있는가?
    - 불확실성 (predictive uncertainty)을 측정하고 우리가 제어 할 수 있는가?
    - 데이터셋의 도메인 시프트가 있는 경우에도 불확실성이 어떻게 나타나는가? 그것을 알고 제어 할 수 있는가?


### 2. Main Contributions
- 1) `predictive uncertainty estimation` 이라는 개념을 처음으로 제시
    - 적용 할 수 있는 simple pipeline 제시
        - proper scoring rule: 특정 조건을 만족하는 loss함수
        - adversarial training    

- 2) 위 두 가지를 이용한 안정된 예측(smooth prediction)을 할 수 있는 방법론 제시
    - uniformly-weighted mixture ensemble
    - 결과를 앙상블 하지말고 distribution을 앙상블 하자
        - distribution을 잘 모델링 해야함 : 흔하게 gaussian / mixture gaussian 사용
    - classification / regression 모두 적용가능

- 3) low-computation and simple modification to classical NN training pipeline
    - variational inference / MCMC 기반의 Bayesian NN 보다 매우 simple

- 아래는 저자의 NIPS2017 발표자료로 부터 발췌
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/contribution.png" title="contrib">
</p>

### 3. Related works

#### Bayesian NNs
- 파라미터 들에 대한 prior을 가정하고 posterior 을 구해서 regulation 하는 방식
- prior 을 정확하게 주는 것이 중요
- Bayesian network 를 복잡하게 설계할 수록 계산량이 크다 
- 일반적으로 computational relaxation 을 적용하여 구현
    - model approximation (parameteric 방식 - variational inferencen 계열)
    - sampling (nonparametric 방식 - MCMC 계열)
-  적절한 computational relaxation 방법을 못찾는 경우 실용화 어려움   

#### Monte carlo dropout (baseline)
- dropout이 기본적으로 model ensemble combination 이라는 것에 주목해서 하는건데 
- dropout 하듯이  model을 sampling해서 훈련 후 ensemble. model sampling에 MC 적용
- 아직 안읽어 봄


### 4. Method Summary

#### Deep ensemble 

##### 1) scoring rule
- S(p_theta,(y,x))로 표기; predictive distribution, p_theta, 의 함수
- measure the quality of predictive uncertainty (높을 수록 uncertainty 낮음)
- scoring rule을 true distribution로 expectation 한 것을 아래와 같이 정의

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/scoring-rule.png" title="scoring-rule">
</p>
    - p_theta : predictive distribution
    - q: true distribution (given by training set)

- **A proper scoring rule** : 아래 조건을 만족하는 경우
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/proper-scoring-rule.png" title="proper-scoring-rule">
</p>

- (-)를 붙여서 loss로 사용

- 1) proper scoring rule for classification :
    - `softmax loss`: S(p_theta,(y,x)) = log p(y|x) 이고 k-multi classification 문제 인 경우
    - `Brier score`: one-hot과 pred dist사이의 cross entropy가 아닌 MSE loss함수를 구성

- 2) proper scoring rule for regression:
    - `negative log likelihood (NLL)`: estimating **mean** and **variance** both
    - `MSE` seeks to only estimate the mean 

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/equ1-nll.png" title="equ1-nll">
</p>

- tips:
    - gaussian은 자유도가 낮음: the other mixture gaussian is better
    - MAP with proper prior can be better than prior

- `remark`: 요지는 model output ensemble 하지말고 predictive distribution을 parametric으로 잘 모델링해서 distribution ensemble해야한다


##### 2) adversarial training (AT)
- 목적: smoothing predictive distribution
    - model output만 출력하는 것은 uncertainty 측정을 전혀 할 수 없음
    - 약간의 loss가 증가하는 방향의 노이즈를 주어서 모델을 강인하게 만들고 잘 학습시킴
- training data x 로 부터 loss가 증가하는 방향으로 perturbation을 더해서 augmentation을 하는 방법
- ex) **fast gradient sign** 
    
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fast-gradient-sign.png" title="fast-gradient-sign">
</p>

- AT는 항상 loss가 증가하는 augmentation을 보장
- 주어진 데이터 주위 /epsilon 반경으로 likelihood를 확대 
    - encourage p(y|x) to be similar to p(y| x + /epsilon)
- model prediction coverage를 주어진 데이터를 중심으로 확대
```
Interestingly, adversarial training can be interpreted as a computationally efficient solution to smooth
the predictive distributions by increasing the likelihood of the target around an /epsilon-neighborhood of
the observed training example
```
 
 - random direction: x' = x + random 으로 augmentation을 할 수 도 있으나 loss의 증가를 보장하지 않음

##### 3) uniformly-weighted mixture ensemble
- ensemble은 크게 random forest계열과 boosting계열로 나뉨
    - parallelization이 쉽다는 측면에서 random forest 선호; boosting계열은 multiple optima가 존재하는 경우 동작을 잘 안해서 deep learning에 맞지 않음
    - random forest의 약점은 branch마다의 correlation이 커지는것
    - 곳곳에 randomization을 하여 de-correlation 하는 것이 중요
        - random init
        - random suffling before data batch builiding
        
- parallel하게 학습해서 **uniformly-weighted mixture ensemble** 사용
    
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/uniformly-weight-mixture.png" title="uniformly-weight-mixture">
</p>


    
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/algo.png" title="algo">
</p>

- 각 ensemble branch를 위한 data random sampliing
- **fast gradient sign** 로 adversarial augmentation
- loss forr adversarial learning
- uniformly-weighted mixture ensemble for M models

### 5. Experimental Result

```
- batch size 100
- adam opt
- lr 0.1 fixed
- \epsilon =0.01 for AT
```
#### 1) toy example : y = x^3 + noise 

- deep ensemble 방법이 얼마나 predictive uncertainty estimation 을 잘하는지 보여주기 위함
    - NLL + AT + ensemble 이 잘한다

    
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fig1.png" title="fig1">
</p>

- left most: MSE pred + ensemble M=5
- second left : NLL pred (est mean and var) + ensemble M=1
- second right : NLL pred (est mean and var) + ensemble M=1 + Adversarial training
- right most : NLL pred (est mean and var) + ensemble M=5 + Adversarial training

    - gray : mean + 3 sigma
    - red : noisy observation
    - blue : true


#### 2) Classification deon MNIST, SVHN, and ImageNet


##### MNIST
```
- model 1: MLP with 3-hidden layers with 200 hidden units
```

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fig2-a.png" title="fig2-a">
</p>

- M 증가에 따라서 성능 개선 
- MC dropout 보다 잘함
- AT 가 random aug(R) 보다 잘함
- 2 layer MLP / CNN에서도 비슷한 동작 확인


##### SVHN

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fig2-b.png" title="fig2-b">
</p>

- MNIST 에 비해서 효과가 없음 
- class 간 data의 특징이 뚜렷히 구분되는 데이터 셋에서는 효과가 없다고 함

##### ImageNet
- M 증가에 따라서 성능 개선 


<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fig4.png" title="fig4">
</p>


#### 3) Uncertainty evaluion: test example from know vs unknow classes

- unseen data (out-of-distribution data)의 predictive uncertainty 측정 목적
- training data에 포함되지 않은 또는 완전히 다른 데이터 셋은 높은 uncertainty를 가지는게 desirable 함


##### 1 MNIST train + NotMNIST test (alphabat in MNIST format)
    
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fig3-a.png" title="fig3-a">
</p>

- entropy histogram
- known class (MNIST test data) 는 entropy가 매우 낮음 (0에 몰림)
- unknown class (NotMIST test data)에서는 entropy  가 크다
    - M이 커질 수록 entropy가 큼 (불확실성이 크다)
    - MC-dropout 도 커지기는 하나 mode는 여전히 0이다
    - AT가 제일 빨리 entropy 증

##### 2 SVHN train + CIFAR test (SVHN not incluidng digit images)
- 비슷한 경향

    
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fig3-b.png" title="fig3-b">
</p>

##### 3 ImageNet dog image train + non-=dog image test
- spliting train set by categories
- train by the dog set
- test by non-dog set 


<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/deep-ensemble/fig5.png" title="fig5">
</p>


### 6. Discussion


### 건질만한 것들

- 

