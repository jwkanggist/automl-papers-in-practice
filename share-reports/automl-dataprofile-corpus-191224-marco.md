# The Influence of Corpus Quality on Statistical Measurements on Language
Resources

Paper Authors:  Thomas Eckart, Uwe Quasthoff, Dirk Goldhahn

Affiliation: Natural Language Processing Group, University of Leipzig, Germany

Publication:  International conference on Language Resources and Evaluatio (LREC) 2012 

([`LREC 2012 proceeding URL`](http://www.lrec-conf.org/proceedings/lrec2012/pdf/476_Paper.pdf))

Update date: Marco @ 191224

---


### Key Question and Hypothesis of Present Paper
- 코퍼스의 품질을 측정하기 위한 통계적 지표는 어떤 것들이 있는가?
- 코퍼스 품질에 따라서 각 통계적 지표는 어떻게 나타나는가?


### Main Contributions
- 전처리에 따른 코퍼스 데이터량 변 경향을 보여줌
- 코퍼스의 통계적 지표의 정리


### 전제
- 좋은 코퍼스의 지표는 bell-shaped distribution을 가진다. (본인 논문 인용)
- 논문에서 사용한 데이터 소스:  newspaper + Wikipedia articles + Web pages

### 전처리에 따른 데이터 감소 

#### 문장 수준 전처리 
```
(a) Boilerplate removal
(b) Removal of foreign language parts (whole texts
or sentences)
(c) (Optional) Removal of parts which are not wellformed sentences, using pattern matching methods
(d) (Optional) Removal of duplicate sentences
(e) (Optional)  Removal of near duplicate sentences
```

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table1.png" title="table1">
</p>

- after pattern based cleaning : (a)-(c)
- after duplicate removal: (a)-(e)

> Remark1: 문장 수준 전처리와 단어 수준 전처리르 구분해야 한다. 

> Remark2: 문장 수준 전처리는 코퍼스 데이터 감소량이 크다 


#### 단어 수준 전처리
```
(a) Boilerplate removal
(b) Removal of foreign language parts (whole texts
or sentences)
(c) Removal of all types
that contained characters not in a specified set of letters,
numbers and some special characters.
(d) removal of all words containing numbers and other ill-formed terms based on set of patterns.
```

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table2.png" title="table2">
</p>

- Number of types with only valid characters : (a)-(c)
- Additional cleaning procedures : (a)-(d)

> Remark1: 영어는 다양한 특수문자와 함께 사용되기 때문에 다른 언어에 비해서 전처리에 따른 감소량이 크다

### 통계적 지표

```
• Number of tokens: 코퍼스를 이루는 단어의 개수
• Type-Token-Ratio: | Number of Types | / | Number of tokens | 로 정의
• Average word length: 코퍼스 안에 존재하는 워드의 평균 길이 (워드를 구성하는 char수로 측정)
• Average sentence length: 코퍼스 안에 존재하는 문장 평균 길이 (문장을 구성하는 토수로 측정)
```

> Remark1: 토큰은 공백 또는 `punctuation marks` (. ? "" ! , )로 구분한다. 

> Remark2: Types- uniques words in corpus (유니크한  토큰 수)

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table3.png" title="table3">
</p>

> remark3: 전처리에 따른 전체 토큰 수 감소는 크지 않음

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table4.png" title="table4">
</p>

> 전처리에 따른 특수문자 / 유효하지 않은 문자 제거에 따른 type수의 큰 감소 가 TTR 감소로 이어짐


<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table5.png" title="table5">
</p>

> 첫번째 전처리에서 HTML/JavaScript Markup, Email, URL 형식이 포함되는 토큰 제거가  word length 감소에 매우 큰 영향을 미침

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table6.png" title="table6">
</p>



#### exemplary distribution :  sentence length
- near duplicated sentence가 two peaks in the distribution의 원인


<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/fig1.png" title="fig1">
</p>

> 1) The first peak is due to a large set of nearly identical
sentences that were not removed by duplicate detection:
```
• Taun ka-1118 Masehi dina Kal  ender Gr ´ egorian. 
• Taun ka-1119 Masehi dina Kal  ender Gr ´ egorian. 
• Taun ka-1120 Masehi dina Kal  ender Gr ´ egorian. 
```

> 2) The second peak is due to the following kind of sentences:
```
• Ancol nyaeta salasahiji d ´ esa di kacamatan Cineam, ´Kabupaten Tasikmalaya, Propinsi Jawa Barat, In-donesia. 
• Babakan nyaeta salasahiji d ´ esa di kacamatan Wanayasa, Kabupaten Purwakarta, Propinsi Jawa Barat, Indonesia. 
• Bakung Lor nyaeta salasahiji d ´ esa di kacamatan Klan-genan, Kabupaten Cirebon, Propinsi Jawa Barat, In-donesia. 
```

> remark1: near-dup가 코퍼스의 품질을 주요한게 여해함

#### Discussion
> Khan Dataprofiling feature에 아래 추가 예정
- Number of types
- TTR
- Average word length 

> 수나 특수기호 다름에 따른 near-dup 문장 제거를 어떻게 해야하는가 고민 필요 아래는 논문에서 얘기하는 near-dup 문장들의 특징
- 공유하는 단어의 수
- 문장의 길이 
- 비슷한 문장의 시작과 끝
- puntuation marks에 만 다른 경우 


> 본 논문애서는  통계지표의 히스토그램 분포가 `bell-shaped`로 나타나는 경우 코퍼스의 품질이 좋다고 주장하고 있으나 그런 경우 모델의 추론성능도 함께 좋을지는 확인 되지 않음
- 따라서 관련 하여 각 통계분포가 얼마나 `bell-shaped` 한지의 지표를 같이 보면 좋을 것 같음
- 본 논문에 따른면 빡센 전처리가 코퍼스 품질을 개선하고 있음을 주장함
