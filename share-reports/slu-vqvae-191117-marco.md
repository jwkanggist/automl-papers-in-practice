# Neural Discrete Representation Learning
Paper Authors: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu

Affiliation: Google DeepMind

Publication: NIPS2017 ([`ArXiv URL`](https://arxiv.org/abs/1711.00937), [`Github URL`](https://github.com/hiwonjoon/tf-vqvae)) [`blog`](https://avdnoord.github.io/homepage/vqvae/)›

Update date: Marco @ 191117

---

### 1. Key Question and Hypothesis of Present Paper
- variational encoder with discrete latent variable을 사용하면 더 압축효율 feature embedding을 얻을 수 있는가?
- VAE에서 입력에 노이즈가 큰 상황에서 일어나는 `posterior collapse` 현상을 discrete latent variable을 적용하면 해결할 수 있는가?



###  2. Main Contributions
- vq-vae 모델 제안
    - discrete latent variable 을 사용하는 vae 모델
    - posterior collapse 현상 극복 ?
-   다양한 도메인에서 적용 사례를 보임
     - image sampling/ audio reconstruction

```
posterior collapse 현상
- VAE에서 decoder 가 latent를 무시하고 신호를 복원하는 현상
- latent가 gaussian 이라고 했을떄 
    - latent의 variance가 너무 크게 estimation되는 경우 mean값이 의미를 가지지 못하는 맥락
    -  example: Z = X + W where X = 5, W ~ N(0, \sigma_w^2)
        - ==> Z ~ N(X, \sigma_w^2) ~= N(whatever, \sigma_w^2) when \sigma_w --> \infinity
```
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/posterial-collapse.png"  title="posterial-collapse">
</p>

###  3. Proposed VQ-VAE

#####  VAE: 
- input x --> latent posterial q(z|x)  -->  reconstruction x_hat from p(x|z)
    - encoder:
        - 1) find posterial q(z|x) given x
        - 2) latent parameter estimation: find parameter of the posterior q(z|x)
            - approximate the latent z to `Gaussian distribution` and let the VAE  estimate mean and var of z
    - decoder : reconstruct x_hat by sampled z_hat from the posterior q(z|x)
    
    
##### VQ-VAE:

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/fig1.png"  title="fig1">
</p>

- encoder: 
    - obtain output of encoder the z_e(x)
        - 논문에는 나와있지 않지만  z_e(x)도 확률 분포이어야 말이
    - run `one-hot encoding` given dictionary matrix E:
        - where the dictionary E works as centroid in K-means clustering 
        - where the latent z is not continuous but a discrete latent variable 
            - p(z) is K - categorical distribution (multinomial) and set the multinomial rate to uniform  in this work. 
        
 <p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/equ1.png"  title="equ1">
</p>                

    
- decoder 
    - 1) nearest-neighbour look-up z_q(x): z_e(x)와 가장 가까운  dictionary vector e_k \in {e_0,...,e_K-1} 를 사용
        - which is the decoder input
    - 2) obtain the decoder outputs p(x|z) 

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/equ2.png"  title="equ2">
</p>   
          

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/equ3.png"  title="equ3">
</p> 
   
- about training loss: 
    - where the sg[]는 stop gradient operator
        - backpropd의 forward pass에서는 identity
        - backward pass에서는  zero patial derivative
        - 즉 sg[] 안에 있는 함수는 gradient 업데이트 대상이 아니라는 것
        
- three loss terms
    - reconstruction loss:  
        - to train the encoder and decoder
        - no effect on the dictionary E
    - dictionary learning loss:
        - for the `vector quantization` : dictionary E를 학습하기 위함
        - E는 encoder output을 닮아가는 방향으로 (l2 norm기반) 학습됨
        - no effect on the enc and  dec
        -  E의 초기값은 intent의 중심벡터라고 생각하고 주면 OK
    - commitment loss:
        -  encoder output 과 dictionary가 보조를 맞춰서 학습되도록 강제하는 regularization loss
        
        
- reconstruction:
    - given the decoder output and the latent prior p(z), we have the reconstruction as
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/decoder-output.png"  title="decoder-output">
</p>    


```
- [Marco's Remark] noise reduction 효과 ?
    - 사실 이것도 noise가 quantization stepsize 충분히 보다 작을때 의미 있음. 노이즈가 너무 크면 효과 없음
        - where stepsize related to 카테고리의 개수 K
        - 즉 효과가 크기 위해서는 dictionary size가 작던지 노이즈가 작던지
```       


### 4. Experimental Results

##### image reconstruction
- CIFAR10 dataset
- model:   
    - The encoder: 2 strided convolutional layers with stride 2 and window size 4 × 4, followed by two residual 3 × 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units. 
    - The decoder: two residual 3 × 3 blocks, followed by two transposed convolutions with stride 2 and window size 4 × 4. 
- 결과   
    - left - orig: 128 x 128 x 3
    - right - reconst from  : 32 x 32 x 1 (K=512)
- remark: (128 x 128 x 3 x 8) / (32 x 32 x 9) = 42.6 in bits로 압축가능

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/image-ex.png"  title="image-ex">
</p>    

##### audio reconstruction 
- VCTK dataset
    - 109 different speakers
- model :
    - The encoder : 6 strided convolutions with stride 2 and window-size 4
    - The decoder :  a dilated convolutional architecture similar to WaveNet decoder
        - conditioned on the latents and one-hot vector to indicate speaker
            
```            
- 결과1: the original and reconstruction    
    - left - orig: 256-quantized (8bits) audio samples
    - right - reconst from  : 256-quantized (9bits) 64x downsampled (K=512), the same speaker id
    - 컨텍스트는 완벽하게 복원하나 억양이 조금 달라짐
```
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/audio-reconst.png"  title="audio-reconst">
</p> 

``` 
- 결과2: the voice style-transfer
    - left - orig: 256-quantized (8bits) audio samples
    - right - reconst from  : 256-quantized (9bits) 64x downsampled (K=512), the diff speaker id
    - 컨텍스는은 완벽하게 복원, 스피커 목소리 바뀜
```

<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/audio-reconst-diff.png"  title="audio-reconst-diff">
</p> 

```
- 결과3: wavenet을 훈련하여 encoder 대신에 사용가능
```
<p align="center">
  <img src="https://github.com/jwkanggist/automl-papers-in-practice/blob/master/share-reports/figs/vq-vae/audio-wavenet.png"  title="audio-wavenet">
</p> 

- demo: the author's blog (https://avdnoord.github.io/homepage/vqvae/)
    
##### 아무나 하는 생각
- 한국어 voice vq-vae 임베딩 만듬 (일단 공용데이터)
- 필요한 음성 데이터 셋을 넣어서 다양한 스피커 id로 음성 합성가능
 