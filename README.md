# CycleGAN GTA5-to-Real Image

# [1]. 실험목표

- 참고 논문에서 Github속 사진에는 비슷한 시간대의 GTA영상과 실제 운전모습을 적용 했을 때 결과가 나옴.
- 실제로 게임 실행하여, GTA영상을 녹화함
- 참고 논문 결과와 달리 이번 실험에서는 GTA영상과 실제 운전 이미지의 시간대와 배경환경을 달리하여, 다음 같은 가능성을 확인하고자 함
- GAN을 통해 시간대 변경 학습에 대한 가능성
- 부분적인 채도, 명도, 색상이 잘 조절이 아닌 전체 이미지에 대해서 적용의 가능성
- GTA는 미국이나 타국을 배경이며, 실제영상은 한국으로 매칭이 되는지에 대한 가능성

실제 학습에 사용한 데이터 대표 예시

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled.png)

<GTA 온라인 버전 석양 질 시간대 > <YouTube을 통해 다운받은 낮 운전 영상>



# [2]. CycleGAN?란

## PIX2PIX + GAN = CycleGAN

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%202.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%202.png)

1. Pix2pix모델은 input과 ground truth가 맵핑됨 

>> 맵핑 된 데이터를 실제 이미지를 대입하기엔 어려움

2. GAN모델은 image translation에서는 input과 output이
학습이 잘되는 방향으로 연결된다는 보장이 전혀 없음

>> Generator는 discriminator를 속이기만 하면 되기 때문 >> mode collapse 발생

## CycleGAN의 간략요약

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%203.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%203.png)

- X에서 Y로의 변환 모델을 학습시키고자 할 때, G: X→Y 뿐 아니라 F: Y→X를 동시에 학습시킴

즉, G로 생성된 Y를 다시 F에 적용했을 때 기존의 이미지 X로 돌아올 수 있어야 함

- Pix2Pix모델 처럼 Input과 Ground Truth가 짝(pair)을 이루지 않고, unpair한 이미지로 스타일만 가져 올 수있음

## Cycle consistency

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%204.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%204.png)

1. G가 input x와 학습방향에 연관되어 있는 output을 만들었다는 것
2. output이 외관은 바뀌었지만 input x와 같은 concept을 공유한다는 의미로 해석함
3. G가 만든 output이 또 다른 함수 Y를 통해 다시 x로 돌아갈 수 있다면, 그 output에는 x로 돌아갈 수 있는 충분한 정보가 남아 있었다고 봄
4. 의미있게 보면 input x와 output에 대해서 unpair을 넣어서, pair적인 개념이 된다고 개인적으로 생각함

## CycleGAN의 LOSS

사진을 translation하는데, 다시 원래 그림으로 복구가능한 정도로만 변환하는 의미

- **Adversarial Loss**

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%205.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%205.png)

A>>B로 가는 모델과 B>>A 로 가는 모델의 맵핑함수는 adversarial loss를 활용함

G(A>>B)는 위의 함수를 최소화시키는 것이, 반대로 D(B>>A)는 위의 함수를 최대화하는 것

- **Cycle Consistency Loss**

adversarial loss에 consistency개념이 더해져서, 다시 A>>B 로 만든 것을 A로 / B>>A 것을 다시 B로 가는 모델을 생성하여, 그 모델에 대한 loss가 추가됨

Adversarial losses 단독으로는 매핑 함수를 제대로 된 학습을 보장하기 어려워 한번 더 하는 것이라고 이해함

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%206.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%206.png)

# [3]. 구현 조건

- GTA GAME을 저녁 6시대에 실행하여, 석양 있는 국도로 운행하여 5분영상을 60프레임으로 1인칭시점으로 저장
- GPU 메모리 부족 현상으로 1920*1080이미지를 196*196으로 Resize해서 학습을 진행하여 GT보다 해상도가 낮을 수 있음
- 실제 영상은 6분영상에 시내에서 국도로 운행하는 1인칭 시점 자동차 영상을 저장
- 저녁 6시 시간대 GTA영상이 실제 낮 운행 영상으로 변환이 되는지 확인함
- 다양한 Discriminator 의 다양한 Loss을 활용해 GTA영상이 예측한 이미지를 확인
- [X >> Y] Generator와 [Y >> X] Generator Loss는 MSELoss를 사용, Cycle, Identity Loss는 L1Loss를 사용
- Tatal Loss = Cycle Loss와 Identity Loss 에는 10/5로 가중치를 둠

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%207.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled11.png)

- Epoch이 낮을 때는, 영상의 화질 낮게 출력되는 현상 있고, 아직 GTA게임에 가까움
1. 전반적으로, 창을 통해 보이는
구름과 하늘 색은 저녁 > 구름 낀 하늘로 변경됨
2. 사람의 살색 경우 아직까지 노란빛이 돔
전면에 보이는 도로와 노란색 자동차가 진한 노란색인데 뿌옇게 노란색이 됨

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%207.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled12.png)
- 50 Epoch이상부터는 전반적으로, 픽셀이 번지는 현상이 발생하지만, 전반적으로
색상이 진해지며, 잉크 번지듯이 차량 내부 틀 색이 망가짐
1. GTA차량은 차량내부는 검정색이지만,예측은 기아차의 내부 회색으로  변경됨
사람의 살색 경우, 실제 GT의 사람색과 비슷해짐
2. GT이미지와 GTA이미지가 섞여지기 시작함
3. GTA의 전반적 틀은 맞지만, 핸들의 손위치가 GTA와 살짝 다르고 오히려 GT에서 비슷한 이미지를 찾는 느낌이 강하게 듬 >> mode collapse 발생

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%207.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled13.png)
- 100 Epoch이상 부터, 영상의 화질 낮게 출력되는 현상
1. 전반적으로, 창을 통해 보이는
구름과 하늘 색은 저녁 > 구름 낀 하늘로 변경됨
2. 차량내부와 국도 색이 구분되기 시작함 또한 국도에 노란색 있는게 흐릿하게 생김
전면에 보이는 도로와 노란색 자동차가 진한 노란색인데 뿌옇게 노란색으로 보임

# [4]. 실험 방향

- 개선할 점
1. Epoch이 진행될수록 점점 학습이 무뎌지는 현상이 발생하면서 mode collapse현상이 발생
2. 학습되면서, 영상이 뿌옇게 되는 현상
3. 차량 내부가 잘 Translation되면, 차량 창 밖의 풍경의 채도와 색상이 엉망이됨
4. 이미지가 전반적으로 뭉게지는 현상

- 개선 아이디어
1. Discriminator Model 안에 Relu함수를 Weakly Relu함수로 변경하여, 학습이 안되는 현상을 개선해보자
2. 학습되면서, 영상이 뿌옇게 되는 현상은 Total Loss에서 cycle과 identity L1Loss의 가중치를 조절하자
3. 차량 창 밖의 풍경의 채도와 색상이 엉망이 되는 현상과 뭉게지는 이미지에 학습전에 Crop하여 Patch기능을 없애 보자

## 1. 개선방향 구현 내용1 및 결과

### Epoch이 진행될수록 점점 학습이 무뎌지는 현상

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%208.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%208.png)

110epoch의 fps의 한 장면을 본 결과, 같은 Epoch이지만 미세하게 계속 학습되는 경향으로 개선 시킴
게다가 이미지가 뿌옇게 되는 현상도 감소하며, 차량 거울에 반사된 풍경도 더욱 선명해짐
오히려 이미지가 뭉게지는 현상도 크게 감소 하면서 mode collapse현상도 감소함

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%209.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%209.png)


## 2. 학습이되면서, 영상이 뿌옇게 되는 현상
<img width="80%" src="https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/media1.gif"/>

![CycleGAN%20GTA5-to-Real%20Image%20aff4293d222143ffb6b6d82029785cce/Untitled%207.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled20.png)
- 개선 방향 1일 때는 Default값으로, W1과 W2에 각10/5로 설정
- 개선 방향 2일 때는 W1과 W2에 각 1/1로 설정
- TEST할 때, 이미지 해상도를 384*384 로 변경
- Epoch 76 일 때 영상
- 창 밖에 보이는 풍경과 도로, 나무의 색상조절과 뿌옇게되는 현상이 사라져, 좀 더 현실적인 느낌이 강하게 나옴
- 예상치 못하게 차량내부가 어둡게 나옴
- 그러면서 Total Loss의 히스토그램이 낮게 나와 기뻐했지만, 당연한 걸 깨닫게 됨 >> 수치가 낮은 것은 의미가 없고 그래프가 모양을 봐야 한다는 것을 파악함
- 현실성이 높아지고 뭉게지는 현상이 많이 사라졌지만, 아직 실제로 GT에 해당되는 이미지와 많이 다름

## 3. 학습이되면서, 영상이 뿌옇게 되는 현상
<img width="80%" src="https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/media2.gif"/>
이미지가 DataLoader를 통해 CycleGAN에서도 처음 구현할 때 pix2pix처럼 이미지를 patch하여 넣기 때문에, patch된 이미지에서 채도와 색상이 변환된다고 생각되어, crop하지않고, 이미지 전체를 한 번에 넣어 학습했음

- 개선 방향 2일 때는W1과 W2에 각1/1로 설정해서 차량내부가 어두워 재조정
- 개선 방향 3일 때는 W1과 W2에 각 5/5로 설정하여, 색상은 낮아져 Input으로 넣은 GT이미지에 가까워졌으며, 차량내부와 계기판이 뚜렷하게 보임
- TEST할 때, 이미지 해상도를 208*208 로 변경하여 fps60일때로 자연스럽게 동영상 출력
- 개선 방향을 진행할수록 현실성이 높아지며, 색감이 Ground Truth에 가까워지지만,  운전대와 사람손을 구분하지 못하는 문제점과 mode collapse 조금씩 다시 발생

## 4. A>>B / B>>A 결과

![CycleGAN2%2018f9179936fc41ea972447935a054f68/Untitled%201.png](https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/Untitled%2021.png)

## 5. 실험 결론

GTA Image-To-Real Image Translation의 개선

- 논문을 구현하여, 모델을 적용한 결과 시간대가 다른 GTA영상과 실제영상 간에 변환이 되는지에 대해서 실험함
1. GAN을 통해 시간대 변경 학습에 대한 가능성 

    >> 낮은 Epoch에서부터 낮과 저녁은 바꿀 수 있음

2. 시간대가 달라도 채도, 명도, 색상이 잘 조절되는지에 대한 가능성 

    >> 전체적으로 조절하기 쉽진 않지만, 차량 실내와 차량 밖의 풍경의 채도, 명도, 색상을 가중치와 이미지 loss함수 변경 등 학습 방법을 통해 이번 실험을 통해 가능성을 성공적으로 봤다고 봄

3. GTA는 미국이나 타국을 배경이며, 실제영상은 한국으로 매칭이 되는지에 대한 가능성      

    >> 대표적은 도로의 색상과 나무의 색상은 바뀌는 것을 확인했으나, 전체적인 분위기는 아직 이국적인 부분이 많음

4. 학습하면서 기존에 epoch이 높아질수록 80을 넘어가는 순간부터 오히려 mode collapse현상이 발생했었는데 , 이번 실험을 통해 줄이는 가능성을 보았음

## 5-1. 시행 착오 영상들
<img width="80%" src="https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/media3_Trim.gif"/>
<img width="80%" src="https://github.com/justin95214/CycleGAN-GTA-to-Real-Image/blob/main/src/media5_Trim.gif"/>


## 6. 아쉬운 점

- GPU 메모리가 좀 더 확보했으면, 저 높은 해상도의 Translation결과를 기대함
- 눈으로 기준으로 하는 것보다 Object Detection을 적용하여, 실제로 Class별 정확도와 Pix별 정확도를 기준으로 실험을 개선했으면 하는 바램 >> 데이터셋을 조금 더 확보하고 , 이번 실험을 바탕으로 모델을 개선하여 조금 더 실제 영상처럼 기여하고 싶음


# [6]참고 자료
- https://mikigom.github.io/jekyll/update/2017/07/11/cyclegan.html
- https://engineer-mole.tistory.com/50
- https://chloe-ki.tistory.com/entry/cyclegan
- https://arxiv.org/abs/1703.10593
- http://www.kwangsiklee.com/2018/03/cyclegan%EC%9D%B4-%EB%AC%B4%EC%97%87%EC%9D%B8%EC%A7%80-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90/
