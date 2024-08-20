# 2장 트랜스포머 아키텍처  
이건 너무 많이 공부했던 부분이라 일단 익숙해져버림 관련 ppt 2개 직접 제작에 책 1권을 가지고 공부했던 경험이 있어서 읽는데 어려움 없었음  
마찬가지로 중요한 부분들만 직접 자료들 만들어서 진행할 예정  

텍스트 임베딩과정  
1. 텍스트를 적절한 단위로 잘라 숫자형 ID를 부여하는 토큰화(tokenization) 수행 EX) 나는 ->나 -느 -ㄴ
2. 토큰 ID를 토큰 임베딩 층을 통해 여러 숫자의 집합인 토큰 임베딩으로 변환
3. 위치 인코딩 층을 통해 토큰의 위치 정보를 담고 있는 위치 임베딩을 추가해 모델에 입력할 임베딩으로 만듬

토큰화란 텍스트를 적절한 단위로 나누고 숫자 아이디를 부여하는 것을 의미  

![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_token.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_token.png)

토큰화의 작은 단위와 큰 단위의 장단점이 뚜렷하기 때문에 데이터 등장 빈도에 따른토큰화 단위를 결정하는  
서브워드(subword) 토큰화 방식을 사용한다.  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_subtoken.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_subtoken.png)


처음 임베딩 층은 그저 입력 토큰 아이디를 16차원의 임의의 숫자 집합으로 바꿀 뿐이다. 이 지점에서 딥 러닝이 머신 러닝과 차별화 되는데  
딥 러닝에서는 모델이 특정 작업을 잘 수행하도록 임베딩을 만드는 방법도 함께 학습한다.  

RNN과 트랜스포머의 가장 큰 차이점은 입력을 순차적으로 처리하는지 여부이다.  
RNN은 입력을 순차적으로 처리하므로 알아서 입력 정보를 고려하게 된다.  
트랜스포머는 모든 입력을 동시에 처리하므로 입력 정보가 사라진다. 하지만 텍스트에서 순서는 매우 중요한 정보이므로 추가해야하므로 위치 임베딩을 추가한다.  

1. 절대적 위치 인코딩: 수식을 통해 위치 정보를 추가하는 방식이나 임베딩으로 위치 정보를 학습하는 방식은 결국 추론 단계에서 입력 토큰의 위치에 따라 고정된 임베딩을
   추가하기 때문에 이를 절대적 위치 인코딩으로 부른다.
2. 상대적 위치 인코딩: 절대적 위치 인코딩은 구현이 간단하지만 토큰과 토큰 사이의 상대적인 위치 정보를 활용하지 못하고 긴 텍스트에서 성능이 떨어진다.
   그래서 최근에는 상대적 위치 인코딩을 사용한다.

![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_graph.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_graph.png)

***
트랜스 포머의 중요한 개념인 어텐션을 공부한다.  
어텐션 요소인 쿼리, 키 벨류를 검색할 때 예를 들면

1. 쿼리: 우리가 입력하는 검색어
2. 키: 쿼리와 관련이 있는지 계산하기 위해 문서가 가진 특징
3. 밸류: 쿼리와 관린이 깊은 키를 가진 문서를 찾아 관련도순으로 정렬해서 문서를 제공할 때 문서

![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_search.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_search.png)

쿼리 키 밸류의 계산 플로우 차트를 보면 아래 사진과 같다.  
아래 사진은 내가 다른 책으로 공부할 때 직접 만든 플로우 차트이다.  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_flowchart.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_flowchart.png)  

위의 결과는 결국 student라는 단어? 토큰?의 어텐션 점수를 얻을 수 있다. 점수가 크면 그만큼 중요한 단어라는 것을 의미하겠네  

**그래서** 어떻게 계산한다는건데?? 밑에 사진을 통해 수식으로 알아본다.  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_dot.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_dot.png)


이다음에 **예제 2.8. 멀티 헤드 어텐션 구현** 코드 부분 무조건 이해하고 넘어가기  
어텐션의 multi head attention pseudo code에 관련해서 작성한 부분임  
왜 중요하냐? 어느 회사 코딩 테스트? 보러갔는데 이거 코드보고 어떤 부분이 뭐하는지 주석 작성하는 문제가 있었음    
## 이 뒤로 넘어가기 전에 코드 예제는 무조건 이해한 상태로 진행하기 트랜스포머의 제일 핵심적인 부분임 넘어가기 금지  

딥 러닝 모델에 데이터를 입력할 때, 입력 데이터의 분포가 서로 다르면 모델의 학습이 잘 안된다. 그래서 정규화가 필요하다.  
데이터를 정규화하여 모든 입력 변수가 비슷한 범위와 분포를 갖도록 조정한다. 이를 통해 모델은 각 입력 변수의 중요성을  
적절히 반영하여 좀 더 정확한 예측을 할 수 있게 된다. 딥러닝 분야에서는 층과 층 사이에 정규화를 추가해 학습을 안정적으로 만들었다.  
$x_\text{norm} = (x-x_\text{mean})/x_\text{std}$    

이미지 분야에서는 보통 배치 정규화를 사용한다. 왜냐면 항상 입력 데이터의 크기가 고정되어있기 때문에  
자연어 분야에서는 층 정규화를 사용한다. 왜냐면 데이터따라 길이가 다르기 때문이다.  
다르다고? padding 하는거아냐? 그래서 문제가 되는거같음 총 6개가 batch로 들어왔는데 5개가 pad 값이고 batch에 포함되면 정규화 하는 의미가 사라지잖아?  

정규화를 하는 순서에 따라 두가지로 나뉨  
1. 사후 정규화: 어텐션과 피드 포워드 층 이후 층 정규화를 적용
2. 사전 정규화: 층 정규화를 적용후 어텐션과 피드 포워드 층을 통과

논문을 통해 나온결과는 사전 정규화가 좀 더 안정적인 학습을 도와줌  

![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_nomal.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_nomal.png)  


# 인코더  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_encoder.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_encoder.png)  
원래 밑에 위치 인코딩이랑 토큰 임베딩 층이 있어야함  

# 디코더  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_decoder.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_decoder.png)  

인코더와 다른 점은 마스크드 멀티 헤드 어텐션을 사용한다.  
디코더는 생성을 담당하는 부분으로, 사람이 글을 쓸 때 앞 단어부터 순차적으로 작성하는 것처럼  
모델도 앞에서 생성한 토큰을 기반으로 다음 토큰을 생성한다.  
이런 특징을 인과적(causal) 또는 자기 회귀적(auto-regressive)라고 한다.  
중간에 인코더에서 들어오는 줄이 있다. 코드를 보면 확실히 알 수 있는데  
인코더에서 계산한 키와 벨류를 가져와서 사용한다. 이를 크로스 어텐션으로 부른다.    

트랜스포머 아키텍처를 활용한 모델은 크게 세 가지 그룹으로 나눈다. 
1. 인코더만 활용해 자연어 이해(Natural Language Understanding,NLU)
2. 디코더만 활용해 자연어 생성(Natural Language Generation,NLG)
3. 인코더와 디코더를 모두 활용해 더 넓은 범위의 작업

![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_ad.drawio.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/2_ad.drawio.png)  

# 1. 인코더를 활용한 BERT  
인코더만을 활용해 자연어 이해 작업에 집중한 대표적인 모델은 BERT(Bidirectional Encoder Representations from Transformers)이다.  
이름에서 알 수 있듯이 양방향 문맥을 모두 활용해 텍스트를 이해한다.  
입력 토큰의 일부를 마스크 토큰으로 대체하고 그 마스크 토큰을 맞추는 마스크 언어 모델링(MLM) 과제를 통해 사전 학습한다.  
이후 다운스트림 과제에 따라 미세 조정하여 사용한다.  
BERT는 텍스트 분류 뿐만 아니라 토큰 분류, 질문 답변, 자연어 추론 등 다양한 자연어 이해 작업에서 훌륭한 성능을 보인다.  
# 2. 디코더를 활용한 GPT  
GPT(Generative Pre-trained Transformer)는 생성 작업을 위해 만든 모델이다.  
생성 작업의 경우 입력 토큰이나 이전까지 생성한 토큰만을 문맥으로 활용하는 인과적 언어 모델링(Causal Language Modeling,CLM)을  
사용하므로 단방향 방식이다.  
GPT는 다음 토큰을 예측하는 방식으로 사전 학습한다.  
GPT는 개발 버전이 최신 일 수록 모델이 엄청나게 커진다.  
이 책에서 다루는 LLM은 대부분 디코더만을 사용한 생성 모델이다.  
# 3. 인코더와 디코더를 모두 사용하는 BART,T5  
BART(Bidirectional and Auto-Regressive Transformers) 와 T5(Text-to-Text Transfer Transformer)는 자연어 처리 분야에서  
인코더-디코더 트랜스포머 아키텍처를 대표하는 모델이다.  
BART는 사전 학습하기 위해 입력 테스트에 노이즈를 추가하고 노이즈가 제거된 결과를 생성하는 과제를 수행한다.  
T5는 모든 자연어 처리 작업은 결국 Text-to-Text 라는 아이디어를 바탕으로 한다.  
입력의 시작에 과제 종류를 지정해서 하나의 모델에서 지정한 작업 종류에 따라 다양한 동작을 하도록 학습시켰다.  
ex)"영어를 한글로 번역:", "문법적 적합성 판단:","문장 유사도 판단:","요약:"  



