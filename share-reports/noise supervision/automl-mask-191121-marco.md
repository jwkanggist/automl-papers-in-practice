# Paper Masking: A new perspective of noisy supervision

Paper Authors:  Bo Han et al. 

Affiliation: University of technology sydney

Publication: NIPS 2018 ([`ArXiv URL`](https://arxiv.org/abs/1805.08193))

github repo : https://github.com/bhanML/Masking

Update date: Marco @ 200420

---


### 1. Key Question and Hypothesis of Present Paper
- mislabeling에 대한 구조를 사전지식을 주고 어떤 레이블 노이즈를 estimation하면 더 잘 할 수 있는가?


### 2. Main Contributions
- noise estimation을 위한 정보를 사람이 구조적으로 주어서 noisy label이 존재하는 경우 모델 훈련을 더 잘하게 끔 한다. 

- 방법
- 1) human-assisted masking (structure extraction): 오염될 수 있는 class간에 class transition matrix 을 정의하고 human-assisted masking 를 부여 --> finite dataset으로 부족한 정보량 보완 
- 2) learning with masking (structure aligment): class transition matrix가 포함되는  "a structure aware probabilistic model"을 정의 하고 noise estimation 

### 3. 배경 지식

##### label noise 문제 해결 방향 3가지로 정리 :

- 1) Curriculum learning 방식 (sample selectio bias): 훈련 데이터 샘플을 선택하는 방식
    - 샘플 선택 criteria 가 휴리스틱이라서 성능 보장이 어려움
    - 이론적인 성능 보장 어려움
    - mentorNet 이 대표적
    
- 2) regularization 방식
    - Explicit regularization: objective loss를 만지는 방식 (virtual adversarial training 등)
    - Implicit regularization: training 알고리즘 안에서 regularization (temporal ensemble 등)
    - optimal performance에 근접하지 못하는 한계 (왜?)
    
- 3) classifier 위에 block 하나를 추가 더 올려서 label noisying 함수 자체를 하는 방식
    - 1) Training convolutioinal network with noise labels - ICLR workshop 2015 (add linear layer)
    - 2) [benchmark1: Goldberger et al.] Training deep neural-networks using a noise adaptatioin layer - ICLR 2017 (add nonlinear softmax layer)
    - 3) [benchmark2: Patrini et al.] Making deep neural netowrks robust to label noise: A loss correction approach - CVPR 2017
    - 한계:  기존 방법들은 finite dataset이기 때문에 잘 안됨
    - 질문: finite dataset을 가지고돌 noise label을 잘 추정할 수 있는 기법을 만들 수 있는가?    


### 4. A new perspective of noise supervision (how to mask)
- class transition matrix : 노이즈에 의해서 class transition 일어날 확률 행렬
- 사람의 인사이트를 class transition이 일어날 만한 영역만 남겨두고 모두 masking한다
   --> estimation 불확실성을 줄여서 noise estimation 정확도를 높일 수 있다
   --> 팀에서 논의되어 오던 negative labeling의 개념와 일치
   
   
- 아래 그림 중 (a)는 사람이 고양이와 개는 서로 헤갈리기 쉽다고 판단하여 masking하여 relabeling될 수 있게 한다
- 아래 그림 중 (b)는 비슷한 개 3종류가 데이터 셋에 있는 경우 그 class만을 masking하여 relabeling될 수 있게 한다
- 아래 그림 중 (c)는 비슷한 class 끼리 클러스터링 (super-class) 하여 masking한 하여 relabeling될 수 있게 한다  

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/masking/fig1.png" title="fig1">
</p>


### 5. 두 벤치마크 모델에 대한 의견

##### [benchmark1, S-adaptation] Training deep neural-networks using a noise adaptatioin layer - ICLR 2017 (add nonlinear softmax layer)
##### [benchmark2, F-correction] Makinig deep neural netowrks robust to label noise: A loss correction approach - CVPR 2017

- 기존 벤치마크 들은 아래 (a) 과 같이 훈련한다
     - step1) noise transition matrix를 prediction한다 ( y -> \tilde y )
     - step2) noise transition matrix을 이용하여 loss  correction하여  x -> y를 추정한다.
     - where x denote model inputs, y is clean model output (labels) and \tilde y is corresponding noisy label
     - 자세히 읽어보지는 않았지만 [F-correction]은 step1 --> step2로 하고 (noise estimation and model training alternatively)  [ S-adaptation]는 joint 하게 한다. (noise estimation은 모델 훈련 안에서 같이) 
- 이 방법은 두가지 약점이 있다 (내 의견)
    - 훈련 데이터에 noisy label 과 clean label이 동시에 필요하다 (x, y , \tilde y)
    - 훈련 데이터 분포에서 벗어난 패턴을 보이는 label noise를 전혀 correction할 수 없다 (이 부분이 사실 매우 취약)
               

### 6. Learning with Masking

##### 기본 컨셉
핵심주장: 자잘한 label noise는 놔주고 사람이 보기에 잘 틀리것 같은 label끼리만 masking을 씌워서 label noise correction하자
- 그림 (b) 에서 보면 사람 (h)가 개입하여 masking structure (s)를 준다

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/masking/fig2.png" title="fig2">
</p>

##### method detail: when structure meet generative model
- \tilde y : noisy prediction
- y : clean prediction
- x : model input
- s : class transition prob (prob matrix) --> model 에 의해서  class간 transition prob가 multi-Dirac form 으로 구해진다. 
- s0 : prior structure (given by human cognition) where s0 = f(s)
- f() : mapping function from s to s0 : "class transition prob" 을 "prior structure" 변환하는 함수

- x -> y -> s -> s0 -> \tilde y
- 위의 관계를 기반으로 log likelihood 을 설계하고 approximate posterior Q(s0) 을 variational inference하는  equ 1을 세운다. 
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/masking/equ1.png" title="equ1">
</p>

- 여기서  ln Q/P가 아니고 ln P/Q인 것을 의심중
- P(s0) := p(s0 | \tilde y, x) which is posterior of s0
- equ1으로 부터 3가지 를 수행한다. 

1) structure extraction (M-step)
- obtain model predictive distribution p(y | x) (Categorical dist)
- estimate "class transition prob" p(\tilde y | y, s) given prediction p(y | x)
- calculate structure s0 given f() and "class transition prob" S

2) structure alignment (learning structure, Q(s0), E-step) 
- run variational inference wrt  Q(s) to approximate P(s)
- mapping Q(s) --f()--> Q(s0)
- obtain difference of learned structure and prior structure s0_hat : M(s0 s0_hat) 


4. Dataset Specification

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/masking/fig3.png" title="fig3">
</p>

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/masking/table1.png" title="table1">
</p>

5. Experimental Result

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/masking/fig4.png" title="fig4">
</p>

##### note

-  pico에서 class transition matrix에서 
    - 특정 컬럼이 하이라이트 되는 경우 그 컬럼에 해당하는 레이블은 mislabeling하기 쉬운 레이블이다. 
    - 특정 레이블만 relabeling 하기 위해서는 fig-1-a 와 같이 특정 레이블에 강하게 prior를 주고 나머지는 음수값을 주고 pico를 하면된다
    - 특정 레이블을 relabeling 하지 않기 위해서는  해당  transition에 음수값을 준다. 
        - 이것은 네가티브 레이블링을 활용할 수 있다. 
-  row에서 밝게 들어온 레이블들은 비슷환 특징을 가지는 레이블이라고 볼수 있다. 
- 즉 transition을 가지고 valid class transition / invalid class transition 을 성형할 수 있다. 

- class transition matrix 를 row col permutation해서 block diagonal matrix를 만들어서 클래스를 뭉치게 할 수 있다

- 이 논문에서 말하는 h￿uman congnition을 AI가 하기 위해서 어떻게 해야하는가?
    - 관련된 class를 iteration 애 따라서 줄인다
    - human based masking을 어떻게 ai가 하도록 할것인가
  