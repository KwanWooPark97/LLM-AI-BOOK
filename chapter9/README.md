# LLM 어플리케이션 개발하기  

![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/9_1.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/9_1.png)   
그림에서 A,B,C는 LLM에 답변에 필요한 정보를 제공하는 기능을 수행한다.  
GPT는 환각 현상이 일어나서 필요한 정보를 프롬프트에 함께 전달하는 RAG를 사용하면 환각 현상을 줄일 수 있었다.  
그림에서 A는 검색하고 싶은 데이터를 데이터 소스에서 가져와 임베딩 모델을 통해 임베딩 벡터로 만들고 이를 DB에 저장하는 과정이다.  
C는 DB에서 요청과 관련된 데이터를 검색하고 검색한 결과를 프롬프트에 반영하는 과정이다.  
B는 검색한 문서를 사용자의 프롬프트에 반영하는 과정이다.  
앞에서 LLM 추론을 효율적으로 하는 다양한 기술을 공부했다. LLM 추론을 줄이기위해 이전에 같거나 비슷한 요청이 있었다면 그 결과를 활용하는 LLM 캐시를 활용한다.  
D가 그런 역할을 하며 비슷한 요청이 있었다면 추론하지않고 그 결과를 가져다 사용한다. 없었다면 E처럼 LLM 추론을 시행한다.  
LLM이 생성한 결과에는 부적절한 내용이 포함되지 않도록 해야한다. 이를 위해 F는 벡터 DB에서 검색한 결과를 확인하고, G는 LLM이 생성한 결과에 문제가 없는지 검증한다.  
서비스에 들어온 사용자의 요청과 LLM의 응답을 기록해야한다. 기록안하면 사용자의 문의에 대응하기 어렵고 서비스가 잘 작동하는지 확인할 수 없다.  
H는 사용자의 요청과 LLM 시스템의 생성 결과를 기록하는 모니터링 과정이다.  

## 검색 증강 생성(RAG)  
RAG란 LLM에게 단순히 질문이나 요청만 전달하고 생성하는 것이 아니라 답변에 필요한 충분한 정보와 맥락을 제공하고 답변하도록 하는 방법을 말한다.  
그림에서는 북쪽에 해당하는 과정이다.  
실은 이미 다른 repo에서 다룬 내용이다보니 대부분 알고있는거라 이렇게 간단히 정리했다.  
LLM 오케스트레이션 도구는 LLM 애플리케이션을 위한 다양한 구성요소를 연결하는 프레임워크로 대표적으로 라마인덱스,랭체인,캐노파 등이 있다.  
이 책에서는 라마인덱스를 사용한다.  

### 데이터 저장  
그림에서 A는 데이터 소스, 임베딩 모델, 벡터 DB로 구성된다.  
데이터 소스는 텍스트, 이미지와 같은 비정형 데이터가 저장된 데이터 저장소를 의미한다.  
데이터 소스의 텍스트를 임베딩 모델을 사용해 임베딩 벡터로 변환한다.  
변환된 임베딩 벡터는 벡터 사이의 거리를 기준으로 검색하는 특수한 DB인 벡터 DB에 저장한다.  
텍스트 임베딩 모델에는 대표적인 상업용 모델로 OpenAI의 text-embedding-ada-002가 있고 오픈소스로는 Sentence-Transformers 라이브러리를 이용한다.  

벡터 DB는 임베딩 벡터의 저장소이고 입력한 벡터와 유사한 벡터를 찾는 기능을 제공한다. 대표적인 DB는 크로마, 밀버스같은 오픈소스와 파인콘, 위비에이트 같은 상업 서비스가 있고  
최근에는 PostgreSQL 같은 관계형 DB에서도 벡터 검색 기능을 도입하고 강화한다.  
처음에는 문서를 임베딩 모델을 통해 임베딩 벡터로 변환하고 DB에 저장한다. 검색을 수행하는 경우 임베딩 모델을 통해 검색 쿼리를 벡터로 변환해 유사도를 계산한다.  

### 프롬프트에 검색 결과 통합  

LLM은 결과를 생성할 때 프롬프트만 입력으로 받는다. 따라서 그림과 같이  
사용자의 요청과 관련이 큰 문서를 벡터 DB에서 찾고(C) 검색 결괄르 프롬프트에 통합(B)해야 한다.  
검색 결과는 프롬프트 모듈에서 사용자의 요청과 하나로 통합된다.  
책 P.302를 보면 GPT의 환각현상을 보여준다. 진짜 모르고보면 완전 믿음직한 정보라고 착각할 정도이다. 따라서 RAG같은 기술 잘 사용해서 환각을 최대한 억제해야한다.  
여기부터 라마 인덱스를 이용한 RAG 구현을 실습으로 한다. 예제 9.1부터 시작.  

### LLM 캐시  

LLM을 통해 생성을 수행하는 작업은 시간과 비용이 많이 든다. 상업용 API를 사용할 경우 입력 프롬프트의 토큰 수와 생성하는 토큰 수에 따라 비용이 발생한다.  
또한 텍스트 생성할 때 걸리는 시간만큼 사용자는 응답을 기다리는데 이걸 최대한 줄여야한다.  
아니면 LLM 모델을 직접 만들어서 서빙하는 경우 요청이 많아지면 그 만큼 많은 GPU를 사용해야한다.  
따라서 모든 경우에 최대한 LLM을 사용한 추론을 줄이는게 최우선이다.  
그래서 기존에 LLM 추론을 통해 얻은 결과를 저장해두고 비슷한 요청이 들어오면 캐시에서 꺼내쓰며 최대한 LLM 생성 요청을 줄인다.  
### LLM 캐시 작동 원리  
LLM 캐시는 크게 두 가지 방식으로 나눈다.  
1. 일치 캐시: 요청이 완전히 일치하는 경우 저장된 응답을 반환, 파이썬의 딕셔너리 같이 키를 이용해서 있는지 확인 후 반환한다.
2. 유사 검색 캐시: 유사한 요청이 있었는지 확인하고 반환, 유사 검색이기 때문에 문자열을 임베딩 모델을 통해 변환한 임베딩 벡터를 비교한다.

바로 실습들어간다. 9.6  

## 데이터 검증  
안정적으로 LLM을 활용한 애플리케이션을 운영하기 위해서는 사용자의 요청 중에 적절하지 않은 요청에는 응답하지 않고 검색 결과나 LLM의 생성 결과에 적절하지 않은 내용이 포함됐는지 확인하는 절차가 필요하다.  

### 데이터 검증 방식  
데이터 검증이란, 벡터 검색 결과나 LLM 생성 결과에 포함되지 않아야 하는 데이터를 필터링하고 답변을 피해야하는 요청을 선별함으로써 LLM 애플리케이션이 생성한 텍스트로 인해 생길 수 있는 문제를 줄이는 방법을 말한다.  
벡터 DB 검색 데이터나 LLM 생성 데이터에 부적절한 데이터가 섞여 있을 수 있다. 이때 사용가능한 방법은 크게 4가지로 나눌 수 있다.  
1. 규칙 기반: 문자열 매칭이나 정규 표현식을 활용해 데이터를 확인하는 방식 EX) 핸드폰 번호, 날짜, 시간 같이 정해진 형식이 있는 데이터 처리
2. 분류 또는 회귀 모델: 명확한 문자열 패턴이 없는 경우 별도의 분류 또는 회귀 모델 만들기 EX) 부정적 긍정적 감정 분류
3. 임베딩 유사도 기반: EX) 정치와 관련된 답변을 피하고 싶은경우 정치와 관련된 내용을 임베딩 벡터로 만들고 요청의 임베딩이 정치 임베딩과 비슷하면 피함
4. LLM 활용: 텍스트 내에 부적절한 내용이 섞여 있는지 확인하는 방법

데이터 검증 실습 바로 들어감 예제 9.11  

# 정리   
LLM은 다양한 작업을 처리할 수 있는 뛰어난 모델이지만 환각 현상을 보이고 학습 비용이 높아 최신 정보를 반영하기 어렵다는 한계가 있다.  
이를위해 RAG기술을 통해 해결하려 했고 앞으로도 계속 깊게 배울것이다.  
LLM 캐시를 살펴보며 비슷한 요청이 있었는지 캐시를 통해 확인 후 LLM 추론을 줄이는 방향으로 효율적인 운영이 가능하게 해준다.  
LLM이 부적절한 답변을 하지 않도록 미리 필터링 하는 과정을 공부했다.  
서비스에 들어온 요청과 LLM이 생성한 응답을 기록하는 데이터 로깅에 대해 살펴봤다.
