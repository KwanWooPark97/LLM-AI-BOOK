* 임베딩: 데이터의 의미와 특징을 포착해 숫자로 표현한 것 임베딩된 데이터 사이의 거리를 계산하고 관련 있는 데이터를 구분 가능  
  구글에서 word2vec 모델을 통해 단어를 임베딩으로 변환하는 방법 소개함 단어를 임베딩으로 변환한 것을 일컬어  
  단어 임베딩(word embedding)이라고 함  
  * 검색 및 추천  
  * 클러스터링 및 분류  
  * 이상치 탐지
    
 
* 전이 학습: 하나의 문제를 해결하는 과정에서 얻은 지식과 정보를 다른 문제를 풀 때 사용하는 방식
  * 사전 학습:대량의 데이터로 모델을 학습시키는 과정
  * 미세 조정:특정한 문제를 해결하기위한 데이터로 추가 학습하는 과정
  * 다운스트림:사전 학습 모델을 미세 조정해 풀고자하는 과제

* 시퀀스:작은 단위(단어)의 데이터가 연결되고, 그 길이가 다양한 데이터의 형태
  EX)텍스트, 오디오, 시계열과 같은 데이터

기존에는 RNN을 사용하여 처리했지만 순차적인 처리 방식 때문에 시간이 오래걸린다.  
그래서 Transformer 구조를 사용하여 매우 빠른 처리속도로 개선하지만 비 효율적인 메모리를 가지고 간다.

![https://github.com/KwanWooPark97/LLM-AI-BOOK/tree/main/img/attention.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/attention.png)

한번에 입력의 모든 쿼리를 처리해야 하므로 입력이 매우 길면 메모리를 엄청 잡아먹는다.  
그래서 RNN과 Transformer는 아래와 같은 그래프로 표현 가능하다.  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/1_graph.drawio.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/1_graph.drawio.png)  

* 정렬:LLM이 생성하는 답변을 사용자의 요청 의도에 맞추는 것
  사용자가 LLM의 답변에서 얻고자 하는 가치를 반영해 LLM이 학습해서 LLM이 사용자에게 도움이 되고
  가치를 전달하도록 만듬
  
* 지시 데이터 셋:사용자가 요청 또는 지시한 사항과 그에 대한 적절한 응답을 정리한 데이터

LLM을 활용한 애플리케이션을 개발할 때 알고 있어야 할 핵심 개념들  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/1_basic.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/1_basic.png)  


LLM의 다재다능함을 활용하면 아래 그림과 같이 사용자가 LLM을 통해 검색한정보가 요약되고   
쉽게 해설된 결과를 통해 지식을 비교적 쉽게 습득할 수 있고, 습득한 지식을 더 빠른 시간안에 새로운 결과물로 조합할 수 있다.  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/1_relation.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/1_relation.png)  

LLM은 우리가 지식을 습득하고 활용하던 모든 측면에 영향을 주기 때문에 이전 AI 모델보다 사회에 미치는 영향이 크다.  
대부분의 지식 노동자가 수행하는 작업이 자동화 하기 어려웠는데 간단한 작업에서는 사람을 대체할 수 있다.    





