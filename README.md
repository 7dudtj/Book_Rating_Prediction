# :book:Book Rating Prediction

<p align="center">
  <img src="https://user-images.githubusercontent.com/67851701/201507411-822d107f-41f5-4252-9d6c-b766e75812b5.JPG">  
</p>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white"> <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white"> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white"> <img src="https://img.shields.io/badge/W&B-FFBE00?style=for-the-badge&logo=WeightsandBiases&logoColor=white"> <img src="https://img.shields.io/badge/Scikit_learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">

## :heart:Competition Introduction

일반적으로 책 한 권은 원고지 기준 800~1000매 정도 되는 분량을 가지고 있습니다.  

뉴스기사나 짧은 러닝 타임의 동영상처럼 간결하게 콘텐츠를 즐길 수 있는 ‘숏폼 콘텐츠’는 소비자들이 부담 없이 쉽게 선택할 수 있지만, 책 한권을 모두 읽기 위해서는 보다 긴 물리적인 시간이 필요합니다. 또한 소비자 입장에서는 제목, 저자, 표지, 카테고리 등 한정된 정보로 각자가 콘텐츠를 유추하고 구매 유무를 결정해야 하기 때문에 상대적으로 선택에 더욱 신중을 가하게 됩니다.  

해당 경진대회는 이러한 소비자들의 책 구매 결정에 대한 도움을 주기 위한 개인화된 상품 추천 대회입니다.  

책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점, 총 3가지의 데이터 셋(users.csv, books.csv, train_ratings.csv)을 활용하여 이번 대회에서는 각 사용자가 주어진 책에 대해 얼마나 평점을 부여할지에 대해 예측하게 됩니다.  

## :raising_hand:Team Members

Naver Boostcamp AI Tech 4기 Recsys 13조, Team 취향존중  

| [<img src="https://github.com/snuff12.png" width="100px">](https://github.com/snuff12) | [<img src="https://github.com/GT0122.png" width="100px">](https://github.com/GT0122) | [<img src="https://github.com/mbaek01.png" width="100px">](https://github.com/mbaek01) | [<img src="https://github.com/7dudtj.png?v=4" width="100px">](https://github.com/7dudtj) | [<img src="https://github.com/sj970806.png?v=4" width="100px">](https://github.com/sj970806) |  
| :---: | :---: | :---: | :---: | :---: |  
| [김선도](https://github.com/snuff12) | [박경태](https://github.com/GT0122) | [백승렬](https://github.com/mbaek01) | [유영서](https://github.com/7dudtj) | [유상준](https://github.com/sj970806) |

## :computer:How to train prediction model?

1. Repository를 Local 환경으로 clone 합니다.
```shell
git clone https://github.com/boostcampaitech4recsys2/level1_bookratingprediction_recsys-level1-recsys-13.git
cd level1_bookratingprediction_recsys-level1-recsys-13
pip install -r requirements.txt
```

2. 원하는 용도에 맞는 파일을 선택합니다.
```text
main.py: 모델 학습
kfold.py: K-Fold Cross Validation을 이용한 모델 학습
```

3. 원하는 모델에 대하여, 원하는 인자를 제공하여 학습을 진행합니다. 자세한 내용은 main.py 코드를 참고해주세요.
```python
# Available Models: FM, FFM, NCF, WDN
python main.py --MODEL=FM --FM_EMBED_DIM=4 --WEIGHT_DECAY=0.0001 --DATA_PATH='data/data2/' --EPOCHS=10
```

4. 혹시 다음과 같은 에러가 발생하실 경우,
```shell
AssertionError: Torch not compiled with CUDA enabled
```

5. 다음 명령어를 입력하면 에러가 해결됩니다.
```shell
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```

6. 학습된 모델은 다음 경로에 저장됩니다.
```text
./models/
```

## :smiley:Experience our prediction model

Book Rating Prediction Model을 직접 경험하실 수 있도록, 프로토타입을 제작하여 배포하였습니다.  
[책 평점 예측 모델 프로토타입](https://7dudtj-book-rating-prediction-streamlit-o3hwlu.streamlit.app/) 에서 직접 예측 모델을 경험할 수 있습니다.  

## :star:To see more

저희 Project에 대한 더 자세한 정보를 알고싶으시다면, [Project Wiki](https://github.com/boostcampaitech4recsys2/level1_bookratingprediction_recsys-level1-recsys-13/wiki) 를 확인하세요!

## :earth_asia:Result

| 리더보드 | Score|
| :---: | :---: |
| private | 2.1476 |
| public | 2.1454 |
