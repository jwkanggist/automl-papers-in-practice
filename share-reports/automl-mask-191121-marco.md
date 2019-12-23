# Paper Masking: A new perspective of noisy supervision

Paper Authors:  Bo Han et al.

Affiliation: University of technology sydney

Publication: NIPS 2018 ([`ArXiv URL`](https://arxiv.org/abs/1805.08193))

Update date: Marco @ 191216

---


### 1. Key Question and Hypothesis of Present Paper
- mislabeling에 대한 구조를 사전지식을 주고 어떤 레이블 노이즈를 estimation하면 더 잘 할 수 있는가?


### 2. Main Contributions
- one
- two
- three 


### 3. 배경 지식
- label noise 문제 해결 방향 3가지로 정리 :

- 1) Curriculum learnnig 방식: 데이터 샘플을 선택하는 방식
    - 샘플 선택 criteria 가 휴리스틱이라서 성능 보장이 어려움
- 2) regularization 방식
    - regularization bias 때문에 최적화 되기 어려움 (이해X)
- 3) classifier 위에 block 하나를 추가 더 올려서 noise estimation하는 방식




3. Method Summary
- Explanation with figures

4. Dataset Specification


5. Experimental Result


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
  