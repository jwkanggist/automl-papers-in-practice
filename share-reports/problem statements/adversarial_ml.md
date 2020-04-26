# Adversarial ML
Summarized by Jaewook Kang @ 2020 Feb



## 질문: 우리는 어떤 OOD를 잡아내야하는가?
 - 나의 정의: OOD란 : 모델 출력의 어느 클래스에서도 높은 확률 영역에 속하지 못하는 샘플 
 - mislabel -->  현재 label에서 낮은 확률이고 다른 label에서 높은 확률 --> relabeling 
 - trash data --> 모든 클래스에서 낮은 확률 + input vector space에서 비슷한 벡터가 없는 데이터  --> 삭제 (trash filter)
 - adversarial example 데이터 셋 강화: --> 모든 클래스에서 낮은 확률 + input vector space에서 가까운 데이터 존재 -->  adversarial example 생성으로 데이터 셋 강화
 
 
## Related paper footprint
> PI : C. Szegedy(Google), I. J. Goodfellow(OpenAI),  and A. Kurakin(MIT)

- 2014 | Intriguing properties of neural networks | C. Szegedy, et al. | ICLR |  [`PDF`](https://arxiv.org/pdf/1312.6199.pdf)
- 2015 | Explaining and harnessing adversarial examples | I. J. Goodfellow et al.| ICLR |  [`PDF`](https://arxiv.org/pdf/1412.6572.pdf)
< GAN 이 이쯤 나왔습니다>
<!--- 2017 | Adversarial machine learning at scale | A. Kurakin et al. | ICLR |   [`PDF`](https://arxiv.org/pdf/1611.01236.pdf)-->
- 2017 | Adversarial examples in the physical world | A. Kurakin et al. | ICLR |   [`PDF`](https://arxiv.org/pdf/1607.02533.pdf)
- 2017 | Towards Deep Learning Models Resistant to Adversarial Attacks | Aleksander Madry etal | ICLR | [`PDF`](https://arxiv.org/abs/1706.06083)
- 2017 | Towards Evaluating the Robustness of Neural Networks |https://arxiv.org/abs/1608.04644

<!--- 2017 | Ensemble Adversarial Training: Attacks and Defenses  https://arxiv.org/abs/1705.07204-->
<!--- 2017 | Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks |　https://arxiv.org/abs/1704.01155-->
<!--- 2018 | Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples | Anish Athalye et al. | ICML |  [`PDF`](https://arxiv.org/pdf/1802.00420.pdf)-->
<!--- 2018 | Synthesizing Robust Adversarial Examples | https://arxiv.org/abs/1707.07397-->
<!--- 2018 |-->

## 기타 참고 자료 

- https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-8-d9507cf20352
- http://www.cleverhans.io/security/privacy/ml/2016/12/16/breaking-things-is-easy.html
- https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html

## Paper Note 1: Intriguing properties of neural networks 
> 논문에서 `adversarial examples` 에 관련 부분만 관심을 가졌습니다. 

##### Key Question and Hypothesis of Present Paper
- 입력에 small pertubation(=사람이 인지 하지 못할 정도를 변화)를 주어서 의도적으로 모델의 예측을 왜곡할 수 있는가? 
- 그런 입력 샘플 (adversarial examples or hacked sample)을 인위적으로 생성할 수 있는가?
- 입력의 어떤 part가 핵심적으로 모델 예측 결정에 영향을 주는지 알수 있는가?

#####  Main Contributions
- DNN에서 입력의 작은 변화(small pertubation)가 출력의 큰 변화를 야기 할 수 있고 이러한 사실은 잘못된 예측으로 이어질 수 있다는 사실을 발견
- 예측을 왜곡할 수 있는 small pertubation을 포함하는 입력 samples (adversarial example)을 인위적으로 생성할 수 있는 최적화 문제 정립
- 

##### Key Statement or experimental results

###### DNN의 non-local generalization에 주목
- 머신러닝 모델은 input-output mapping을 하는 비선형 함수이다.
- 간단한 머신러닝에서는 input sample의 `local generalization` 이 성립한다. 
- 다시 말해서 매우 작은 \epsilon과 input x에 대해서 x+r ( ||r|| < \epsilon)은 모델에 의해서 x와 같은 클래스로 분류가 된다
- 하지만 DNN은 여러 비선형 층을 통해서 input-output mapping이 매우매우 복잡하다. 따라서 `local generalization`이 더이상 성립하지 않는다.
- input space에서의 작은 epsilon은 output space에서는 더이상 작지 않을 수 있다
- input space에서 euclidean distance 맥락에서 촘촘/균등하게 분포되어 있는 sample들도 output space에서는 촘촘/균등한 sample이 아닐 수 있다.
- input space에서 train sample들이 uniform하게 분포 되어 있어도  결과하는 output sample은 uniform하지 않다.
- input space에서 근접 sample들도 output space에서 멀리 떨어질 수있다.
- input space에서 먼 sample도 output space에서 가까울 수 있다.

###### Adversarial example
- 위와 같은 이유 때문에 
- **output distribution은 불연속적이고 smooth하지 않다**
- input space에서  매우 근접한 sample ( in euclidean distance) 들도 다른 클래스롤 분류 될 수 있다. 
- 같은 클래스로 labeled된 input sample이 아무리 많아도 그 클래승의 output space를 촘촘하게 표현하지 못할지도 모른다
- `Adversarial example`이란 input space에서 원본 sample과 매우 흡사하고 
거리적으로도 (in euclidean distance) 가까운 sample이지만 결과적으로 다른 클래스로 분류되는 sample을 말한다. 
- 원본 sample x에 `small pertubation`이 가해져도 모델이 다른 클래스로 분류할 수 있다.
- 이런  `Adversarial example`을 생성하는 방법을 아는 것은 잘 훈련된 모델을 해킹하는 방법을 아는 것이다.

<블로그에서 그림가져와서 추가>

###### optimization formulation to generate adversarial example
- 즉 adversarial example는 찾는다는 것은 원래 분류되어야 할 class에서 낮은 확률을 갖게 되  input sample 을 찾는다는 것이다
- 즉 모델이 다른 클래스로 분류하게 될 가장 비슷한 input sample
- pertubation크기 ||r|| 을 크게하면 x는 많이 변하기만 더 확실한 adversarial example이 만들어진다. --> 
- adversarial example의 존재는 마지막 층의 output 큰 변화를  일으키는 small pertubation on input이 존재함을 보여준다

<최적화 문제 설정수식 참고>

###### 실험 결과

- `cross-model-generalizatioin` : 비교적 많은 수의 adversarial example이 서로 다른 구조를 가지는 모델을 hacking할수 있다. 
--> see table 1,2
- `cross-dataset-generalization`: 비교적 많은 수의 adversarial example이 서로 다른 training set으로 훈련된  모델을 hacking할 수 있다. 
--> see tablee 3,4


###### adversarial example에 강인한 모델을 만든는 한가지 방법
Lipschitz(립쉬츠) 상수를 구하는 것으로 각 layer 변화의 upper bound를 구함으로써 해당 레이어에서 adversarial example 발생가능성 여부를 판단할 수 있다.
- 큰 립쉬츠 상수 M가 adversarial example의 존재를 보장하지는 않지만 낮은 M은 adversarial example 존재하지 않도록 할 수 있다. but 표현력을 떨어뜨린다
- 따라서 립쉬츠 상수를 제한하는 regularizer를 추가하면 adversarial example에 강인한 모델을 만드는 한가지 방법이 될 수 있다. 



## Paper Note 2: Explaining and harnessing adversarial examples
> https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8-FGSM-%EB%A6%AC%EB%B7%B0-EXPLAINING-AND-HARNESSING-ADVERSARIAL-EXAMPLES
> https://www.slideshare.net/JungHoonSeo2/explaining-and-harnessing-adversarial-examples-2015

기존 논문으로 부터 관점의 변화: 
- 기존:  DNN의 `non-local generalization`이 adversarial example의 원인
- 본 논문:  high dimensionality에서의 linearity가 adversarial example의 원인

- 새로운 trade off : 훈련을 쉽게 하기 위한 선형적 모델 설계 <--> adversarial pertubartion에 강인한 비선형적 모델 설계
- 기존의 regularization은 adversarial pertubartion에 대한 취약성을 줄이지 못함



### paper Note 3 : adversarial examples in the physical world

##### Key Question and Hypothesis of Present Paper
- 만들어진 adversarial example들이 `photo transformation`(카메라 / 마이크와 같은 모델의 입력 인터페이스를) 거쳐서 로 들어왔을때도 모델예측을 왜곡할 수 있을까?

- 사람이 듣기에는 이상한 소리가 머신러닝 시스템을 작동시킨 사례있음
```
An adversarial example for the voice command domain would consist of a recording
that seems to be innocuous to a human observer (such as a song) but contains voice commands recognized by a machine learning algorithm
```

- 사람 얼굴에 미묘한 마킹을 하여 모델로 하여금 다른 사람으로 착각하게 하는 사례
```
An adversarial example for the face recognition domain
might consist of very subtle markings applied to a person’s face, so that a human observer would
recognize their identity correctly, but a machine learning system would recognize them as being a
different person
```

#####  Main Contributions
- 3가지 adversarial example 을 생성하는 방법을 제안
- 1) fast method: 가장 뺘른  adversarial example 생성 방식, 어느 클래스로 바뀔지는 모름, 가장 원본이 L-infinity constraint에서 작게 변하는 방향으로 감
- 2) basic iterative method: fast method가 한방에 adversarial example을 찾는 것이라고 하면 정교하게 iterative하게  구하는 것
- 3) iterative least-likeliy class method: 잘못 판단되도록 하는 class를 지정하는 수 있는 생성 방법
<see equations in page 4>

<see fig2 in page 5>

- adversarial/clean example  `photo transformation` 영향을 보기 위한 실험
- imagenet 데이터 셋으로 1) 원본==clean set 과 2) adversarial example set을 만듦 (clean set은 대조군)
- `photo transformation` 수행: an clean/adversirial image printing out --> 1) photo taking -> 2) cropping --> feed to the model
- 평가 모델 :inception v3
- 아래 두가지 경에 대한 실험을 수행
-- 1) average case :  `photo transformation`된 102 image를 랜덤하게 샘플하여 성능 측정 --> table1
-- 2) prefiltered case: `photo transformation`된 102 image중 confidence score 가 0.8이상인 102개 선별해서 성능 측정 --> table2

< see fig3 in page6>

- 실험결과
- `photo transformation`을 하면 clean example에 대한 성능은 떨어짐
- `photo transformation`을 하면  adversarial example 의 misclassification rate도 떨어짐 (즉 adversarial effect가 mitigated 됨)
- 'basic iterative method' ' iterative least-likeliy class method'으로 만들어진 adversarial example은 `photo transformation`을 거치게 되면 효과를 발휘하지 못한다. 
 
 
< table1 / table2  >


### paper Note 4 : adversarial examples at scale

##### Key Question and Hypothesis of Present Paper
-  Imagenet dataset 처럼 큰 데이터 셋에서도 adversarial training의 효과 가 있는가?


#####  Main Contributions

