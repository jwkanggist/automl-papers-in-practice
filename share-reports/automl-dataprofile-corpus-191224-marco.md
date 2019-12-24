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


### 통계적 지표

• Number of tokens: 코퍼스를 이루는 단어의 개수
• Type-Token-Ratio: | Number of Types | / | Number of tokens | 로 정의
- Types: uniques words in corpus
• Average word length: 코퍼스 안에 존재하는 워드의 평균 길이 
• Average sentence length: 코퍼스 안에 존재하는 문장 평균 길이 (문장을 구성하는 토큰수)



<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table3.png" title="table3">
</p>

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table4.png" title="table4">
</p>

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table5.png" title="table5">
</p>

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/table6.png" title="table6">
</p>


#### exemplary distribution :  sentence length
- near duplicated sentence가 two peaks in the distribution의 원인


<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/lrec2012/fig1.png" title="fig1">
</p>

1) The first peak is due to a large set of nearly identical
sentences that were not removed by duplicate detection:
```
• Taun ka-1118 Masehi dina Kal ´ ender Gr ´ egorian. ´
• Taun ka-1119 Masehi dina Kal ´ ender Gr ´ egorian. ´
• Taun ka-1120 Masehi dina Kal ´ ender Gr ´ egorian. 
```

2) The second peak is due to the following kind of sentences:
```
• Ancol nyaeta salasahiji d ´ esa di kacamatan Cin ´ eam, ´
Kabupaten Tasikmalaya, Propinsi Jawa Barat, In- ´
donesia. ´
• Babakan nyaeta salasahiji d ´ esa di kacamatan ´
Wanayasa, Kabupaten Purwakarta, Propinsi Jawa ´
Barat, Indonesia. ´
• Bakung Lor nyaeta salasahiji d ´ esa di kacamatan Klan- ´
genan, Kabupaten Cirebon, Propinsi Jawa Barat, In- ´
donesia. ´
```

#### remarks

