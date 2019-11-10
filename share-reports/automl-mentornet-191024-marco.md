# MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels

Paper Authors: Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei

Affiliation: Google and Stanford Univ.

Publication: ICML2018 ([`ArXiv URL`](https://arxiv.org/abs/1712.05055), [`Github URL`](https://github.com/google/mentornet))

Update date: Marco @ 191024
---


### 1. Key Question and Hypothesis of Present Paper
- Corrupted Lable 데이터 셋이 주어졌을때,  데이터를 제공하는 순서를 잘 오더링하면 (데이터 커리큘럼을 잘 구성하면) 모델의 학습을 더 잘할 수 있지 않을까?

### 2. Motivation
- 학습 방식 중 에 빈지오 교수가 창안한 `커리큘럼 러닝`(CL) 이라는 방식이 있음.
    - `CL`: Predefined 커리큘럼이 존재하고 결과한 커리큘럼을 이용해서 모델 학습을 하는 방식
- 기존 방식은 모델 학습 피드백을 커리큘럼 구성에 반영하지 않는 한계 존재 
- 기존 방식은 "predefined 커리큘럼"을 구하는 과정이 alternating minimization 으로 구하는데 이 부분이 깊은 모델 학습이 적용하는 경우 잘 안된다고 함

### 2. Main Contributions
- label corruption 문제를 다룸
- 두 모델이 데이터 커리큘럼과 피드백을 주고 받는 방식으로 deep CNN 학습 프레임워크를 구성하고 성능개선
    - MentorNet : 커리큘럼을 만드는 모델
    - StudentNet: 실제 학습을 하는 모델
    
- data-driven  MentorNet 훈력 (기존은 pre-defined 방식)
- 4가지 벤치마크에 대해서 성능 개선을 보임
    - CIFAR / ImageNet / WebVision 데이터 셋


### 3. Method Summary

<!--![equ1](https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/equ1.png)-->
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/equ1.png" title="equ1">
</p>

- equ1 가 overall objective fn
    - w는 student model의 weight
    - L(y,g_v,w)이 student model 학습의 loss

- 본 논문에서는 G(v,\lambda)를 "커리큘럼"이라고 부른다. 
    - 즉 데이터의 corruption 여부를 판별하기 위한 loss
    
- v_i는 해당 데이터 D={x_i,y_i}의 corruption confidence 를 보여준다.; 
    - when v \in {0,1}, the data is clean if v == 1, otherwise corrupted.
    
- \lambda는 커리큘럼 학습에서의 하이퍼 파라미터
- equ1을 최적화 하는 방식은 L()와 G()을 alternative 하게 최소화하는 방식 (alternating minimization)
    - student model의 학습은 find w given D={x_i,y_i},{v} 이다.
    - mentor model 의 학습은 find v given  D={x_i,y_i},w 

<!--![equ3](https://oss.navercorp.com/companyai/louie/blob/master/AutoML/share-reports/figs/mentornet/equ3.png)-->
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/equ3.png" title="equ3">
</p>

- equ3은 mentor model 학습을 위한 objective fn
    - equ3 의 최적화 문제를 푸는 또는 approximation하는 것이 mentor model이다. 
    - z = \phi(x,y,w)를 논문에서 mentor model의 input feature라고 정의한다. 이것은 student model 부터 주어진다. 
    - mentor model은 input feature로 부터 "corruption confidence" v를 추론해야 한다. 

<!--![equ4](https://oss.navercorp.com/companyai/louie/blob/master/AutoML/share-reports/figs/mentornet/equ4.png)-->
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/equ4.png"  title="equ4">
</p>

- equ4는 equ3를 풀어 쓴것이다. 
    - 여기서 v_hat = gm(z_i;\Theta) (mentor model)
    - l_i=L(y_i,gs()) (student model loss)로 보면 된다.
    - theta는 mentor model weight

- equ4를 놓고 mentor model 학습은 두 방식 진행 가능: 
    - pre-defined 방식
    - data-driven 방식


#### 1) approximate **predefined** curriculums

<!--![equ7](https://oss.navercorp.com/companyai/louie/blob/master/AutoML/share-reports/figs/mentornet/equ7.png)-->
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/equ7.png"  title="equ7">
</p>

- 이 경우 mentor model은 rule (equ7)로 주어진다. 
    - 여기서 equ7는 G()가 equ5로 주어졌을 경우이고 
    - [focal loss](https://wordbe.tistory.com/entry/ML-Cross-entropyCategorical-Binary%EC%9D%98-%EC%9D%B4%ED%95%B4)등 다른 loss로 모델링 할수 있다. (focal loss는 틀린경우에 더 weight를 주는 loss로, 이 경우 equ8로 결과한다.)
    - 학습이 필요없음
        
    
#### 2) learning **data-driven** curriculums: v를 추론해줄 모델 학습하기

- 모델을 학습해서 이용해서 v를 추론하는 경우
- equ4를 loss로 두고 mentor model weight \Theta를 학습하기 
- 학습을 위해서 데이터 셋을 아래 와 같이 구성한다.
```
1) input feature로 student model의 loss, 이전 epoch와의 loss difference, and label (y_i), 
2) model output로 해당 label annotation v
3) 추가로 현재 student model의 training epoch percent도 입력으로 들어감
```  
  
- 따라서 mentor model을 학습하기 위해서 전체 데이터 셋 중 일부의 레이블 y_i이 사람에 검증되어 label annotated 되어야함
- 또한 훈련시 mentor model 학습전에 burn-in period 필요
    - burn-in period 란 student model에서 제공하는 input feature의 신뢰성이 낮은 초기 train epoch 구간에서는 mentor model을 학습하지 않고 v ~Bernoulli(p)로 제공한다
    
### 4. Architecture
#### 1) Training Architecture

<!--![train-arch](https://oss.navercorp.com/companyai/louie/blob/master/AutoML/share-reports/figs/mentornet/train_arch.png)-->
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/train_arch.png"  title="train-arch">
</p>

**Training Algorithm**
```
1) student model을 커리큘럼 없이 훈련 (burn-in)
2) if update_curr == true, then mentor model 훈련 (curriculum update,equ3,4)
3) v 값 업데이트 by trained mentor model (equ12)
4) student model 훈련 (equ1)
5) goto 2)
```

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/algo.png"  title="algo">
</p>


- 커리큘럼은 훈련에 따라서 업데이트 된다
- 훈련에서는 student model의 lr이 변하면 업데이트 하는 방식: 훈련이 21%, 75% 진행 됐을때 한번식 업데이트 함 

#### 2) Mentor Model Architecture
- input features
    - data label: one-hot label features
    - training epoch percentage: 0 - 99 사이의 int로 student model의 훈련 진행율을 보이는 features
    - 과거 K개의 mini-batch에 대해서 loss 와 loss diff가 BiLSTM을 타고 feature로 들어감
        - loss: student model sample loss 
        - loss diff: 하나의 minimatch에 안에서 "sample loss"와 "p-th [percentile](https://en.wikipedia.org/wiki/Percentile) loss의 exponential moving average"의 diff

> "p-th percentile" 는 하위 p%의 값을 의미 

- model 
    - 2-layer MLP
    - 2-layer CNN + mean pooling
    - LSTM
    
- workflow
<!--![mentornet-arch](https://oss.navercorp.com/companyai/louie/blob/master/AutoML/share-reports/figs/mentornet/mentornet-arch.png)-->
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/mentornet-arch.png"  title="mentornet-arch">
</p>


- 잘되는 이유에 대해서 설명하기 위해서 커리큘럼 approximator 가 dec fn of student loss (l_i) 인 경우 equ10으로 결과하여 이게 M-estimator를 푸는 문제와 같다라고 하는데 잘 모르겠음


### 5. Experiment


#### Setups
- CIFAR-10/100
```
- 32x32
- 50k training set, 10k valid set
- 10/100 classes
```    

- ImageNet
```
- 299x299x3
- 1.2 million training set, 50k valid set
- 1000 classes
```

- training set 만 오염시키고 validset은 깨끗하게 유지

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/table1.png"  title="table1">
</p>

#### Methods    
- `FullModel`: wihtout CL
- `Forgetting`: dropout param opt
- `Self-paced`: a pre-defined CL
- `Focal Loss`: a pre-defined CL
- `Reed Soft`: a weakly-supervised learning 
- `MentorNet PD`: Mentornet pre-defined (equ5)
- `MentorNet DD`: Mentornet data-driven 

#### 1) CIFAR ex
```
- 20% burn-in
- MentorNet is trained by 5000 images randomly sampled from CIFAR-10 dataset
- noise ratio : 0.2,0.4,0.8 (인위적으로 label corruption)
- batchsize=128
- momentum SGD opt (momentum=0.9)
- l2 weight decay 4E-3 
- data augmentation
- dropout 0.5

- Student models:
    - Resnet101
    - inception
```
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/table2.png"  title="table2">
</p>

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/fig3.png"  title="fig3">
</p>

- remarks
    - MentorNet DD 모든 면에서 월등
    - MentorNet의 training loss는 0가까이로 수
    - 20000 epoches 에서 mentornet이 corrupted label를 골라내기 시작

#### 1) ImageNet ex
```
- 20% burn-in
- MentorNet is trained by 5000 images randomly sampled from CIFAR-10 dataset
- noise ratio : 0.4 (인위적으로 label corruption)
- batchsize=32
- momentum SGD opt (momentum=0.9)
- l2 weight decay 4E-5 
- data augmentation
- dropout 0.8

- Student models:
    - Inception-resnet-v2
``` 
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/mentornet/table3.png"  title="table3">
</p>

### 6. Discussion

####  Comparison to PICO framework 
| - | PICO          | Mentonet        |
|:----:| :--------:    | :--------: |
| 방식 | 데이터 셋을 직접 리레이블링, 두 데이터 셋(train and valid)를 한 모델이 보는 방식  | 모델 통해서 좋은 데이터 선별해서 모델 학습에 feeding, 한 데이터 셋을 두 모델이 보는 방식  |
| 장점 | 데이터 정제와 모델 훈련 분리(training 비용 유지), 오염 데이터 재활용가능 | 데이터 셋을 따로 만질 필요없음, 모델 맞춤형 데이터 커리큐럼 제공  | 
| 단점 | 모델 맞춤형 X, 성능개선 제한적,  | 큰 데이터 셋에 대해서만 적용 가능, training 복잡 (확인필요), MentorNet을 훈련시키기 위한 추가 데이터 셋 필요|
