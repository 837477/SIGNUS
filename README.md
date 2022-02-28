# SIGNUS - 대학생들을 위한 추천 뉴스피드
<br><br>

<div align=center>
    <strong># HTML / CSS</strong> &nbsp;
    <strong># Javascript</strong> &nbsp;
    <strong># Flask</strong> &nbsp;
    <strong># uWsgi</strong> &nbsp;
    <strong># NginX</strong> &nbsp;
    <strong># MongoDB</strong> &nbsp;
    <strong># NLP</strong> &nbsp;
    <strong># AWS</strong> &nbsp;
    <br><br>
    <p><img src="https://837477.github.io/portfolio/signus/img/logo.png"></p>
</div>
<br>

## What is this?
> 본 프로젝트는, "SOOJLE" 프로젝트의 후속 작품입니다.

대학교는 재학생의 편리한 학교생활을 위해 여러 커뮤니티 및 공지 등을 통해 유용한 정보를 게시하거나 공지합니다.<br>
그 안에는 학교의 메인 홈페이지 외에도 학과 및 동아리에서 각각 개설되는 웹 사이트, 학생들이 자발 적으로 만든 커뮤니티, 플랫폼까지 포함하면 한 학교에서만 관련 사이트가 수백 개에 다다르게 되어있습니다.

그러나 이러한 환경에 반해 실제 대학교 관련 정보를 탐색하는 상당히 불편하다는 평을 받고 있고, 서비스 자체의 접근성, 알려지지 않아 가치 면에 비해 사용량이 저조한 서비스, 여러 곳에 흩어져 있고 중복으로 존재하는 정보 등을 통해 구글을 비롯한 통합 검색 엔진으로도 찾을 수 없는 정보가 점점 증가하고 있습니다.

본 프로젝트는 이러한 문제점을 해결하기 위해 해당 학교 관련된 웹상의 모든 정보를 수집하여 통합한 접근성이 높은 서비스를 제공함으로 학생들의 정보 접근성을 높여 더 편리한 학교생활에 임할 수 있도록 하는 서비스 제공을 목표로 합니다.

더불어, 학생 개개인 마다의 관심사를 측정하여 마치 페이스북 및 유튜브와 같은 추천 뉴스피드를 제공합니다.

### Tokenizer
입력된 문자열의 명사 추출 및 리스트의 형태로 반환합니다.
```python
from modules.tokenizer import Tokenizer

msg = "사람은 밥을 먹는다"

obj = Tokenizer()
result = obj.get_tk(msg)
print(result)

>>> ['사람', '밥']
```

### Trainer
```python
from modules.recommender.fasttext.trainer import Trainer

# 학습용 데이터
sent_1 = [
    ['computer', 'aided', 'design'],
    ['computer', 'science'],
    ['computational', 'complexity'],
    ['military', 'supercomputer'],
    ['central', 'processing', 'unit'],
    ['onboard', 'car', 'computer'],
]

# 전이 학습용 데이터
sent_2 = [
    ['computer', 'design', 'aided'],
    ['computer', 'scienc'],
    ['I', 'love', 'him'],
    ['military', 'supercomputer'],
    ['I', 'love', 'you'],
]

# 트레이너 로드 및 하이퍼파라미터 세팅
trainer = Trainer() 
trainer.set_params(
    vec_size=10,
    windows=3,
    min_count=1,
    iteration=1,
    workers=1
)

# 학습용 코포라 세팅 및 학습
trainer.set_corpora(sent_1) 
trainer.train()

# 전이학습용 코포라 세팅 및 학습
trainer.set_corpora(sent_2)
trainer.update()

# 모델 저장 및 불러오기 메소드
trainer.save_model(path="./test_model")
trainer.load_model(path="./test_model")

```

### Recommender
Recommender 클래스를 사용하기 위해서는 사전에 학습된 모델이 필요합니다.<br>
해당 모델의 경로를 클래스 선언시의 인자로 넘겨 모델을 임포트할 수 있습니다.
```python
from modules.recommender.fasttext import Recommender
# Path to FastText model
recommender = Recommender("./ft/soojle_ft_model")


# 입력된 토큰, 토큰 리스트에 대하여 가장 의미상 가까운 단어 num 개를 반환
>>> recommender.doc2words("python")
[('sql', 0.900479257106781), ('tensorflow', 0.8909381628036499), ('javascript', 0.8879169821739197), ('ript', 0.8836684823036194), ('opengl', 0.8771063089370728), ('프로그래밍', 0.8725562691688538), ('nosql', 0.8630207777023315), ('자바스크립트', 0.861074686050415), ('framework', 0.859409749507904), ('nodejs', 0.8563205003738403)]


# 입력된 토큰, 토큰 리스트에 대한 임베딩 벡터 반환
>>> vec1 = recommender.doc2vec("python")
>>> vec1
array([ 0.02856478, -0.11470377,  0.01810528,  0.06545372, ...], dtype=float32)


# 입력된 벡터에 대하여 가장 의미상 가까운 단어 num 개를 반환
>>> recommender.vec2words(vec)
[('python', 1.0), ('sql', 0.900479257106781), ('tensorflow', 0.8909381628036499), ('javascript', 0.8879169821739197), ('ript', 0.8836684823036194), ('opengl', 0.8771063089370728), ('프로그래밍', 0.8725562691688538), ('nosql', 0.8630207777023315), ('자바스크립트', 0.861074686050415), ('framework', 0.859409749507904)]

# 두 토큰 or 토큰 리스트에 대한 의미적 유사도 반환
>>> recommender.doc_sim("java", "python")
0.8159679
>>> recommender.doc_sim("java","짜장면")
-0.37807456

# 두 임베딩 벡터간의 의미적 유사도 반환
>>> vec2 = recommender.doc2vec("java")
>>> recommender.vec_sim(vec1, vec2)
0.8159679

# 해당 토큰을 모델이 알고 있는지 검사
>>> recommender.is_in_dict("python")
True
>>> recommender.is_in_dict("서정민")
False
```

<br>

## Dependency
```shell
python 3.7.X
MongoDB 4.X
```
<br>

## How to use
```shell
pip install -r requirements.txt
python application.py
```
<br>

## About Me
🙋🏻‍♂️ Name: 837477

📧 E-mail: 8374770@gmail.com

🐱 Github: https://github.com/837477

<br>

## Contributing
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -m 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
