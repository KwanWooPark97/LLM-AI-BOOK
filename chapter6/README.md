# sLLM 만들기  
여기서는 자연어 요청으로부터 적합한 SQL(Structured Query Language)을 생성하는 Text2SQL sLLM을 만든다.  

## Text2SQL 데이터 셋  
대표적인 데이터셋으로는 WikiSQL과 Spider가 있다.  
SQL 생성을 위해서는 크게 두가지 데이터가 필요하다.  
1. 어떤 데이터가 있는지 알 수 있는 데이터베이스 정보(테이블과 컬럼)
2. 어떤 데이터를 추출하고 싶은지 나타낸 요청사항(REQUEST OR QUESTION)

wikisql은 다음과 같은 데이터 형식을 가지고 있다.  
****
This example was too long and was cropped:  
{
    "phase": 1,  
    "question": "How would you answer a second test question?",  
    "sql": {  
        "agg": 0,  
        "conds": {  
            "column_index": [2],  
            "condition": ["Some Entity"],  
            "operator_index": [0]  
        },  
        "human_readable": "SELECT Header1 FROM table WHERE Another Header = Some Entity",  
        "sel": 0  
    },  
    "table": "{\"caption\": \"L\", \"header\": [\"Header1\", \"Header 2\", \"Another Header\"], \"id\": \"1-10015132-9\", \"name\": \"table_10015132_11\", \"page_i..."  
}  
****  
대충 보면 SELECT Header1 FROM table WHERE Another Header = Some Entity 이 구문이 명령어 같다.  
table에 있는 Header1 열에서 Another Header가 Some Entity인 데이터를 가져와라 라고 해석 하는듯 하다.   
####  
해당 책에서 사용할 데이터는 https://huggingface.co/datasets/shangrilar/ko_text2sql 여기 데이터 셋을 사용한다.  
데이터 셋은 4개의 컬럼으로 구성돼 있다.  
db_id는 테이블이 포함된 데이터베이스의 아이디로, 동일한 db_id를 갖는 테이블은 같은 도메인을 공유한다.   
####  
## 성능 평가 파이프라인 준비  
Text2SQL 작업을 평가하기 위해 책에서는 GPT-4를 사용한다.  
뛰어난 성능의 LLM을 평가자로 활용하면 빠르게 평가를 수행하면서도 신뢰 가능한 평가 결과를 기대할 수 있다.   
기존 평가 방식은 생성한 SQL이 문자열 그대로 동일한지 확인하는 EM(Exact Match)방식과 쿼리를 수행할 수 있는 데이터베이스를 만들고 프로그래밍 방식으로 SQL 쿼리를 수행해 정답이 일치하는지 확인하는 실행 정확도(Execution Accuracy,EX)가 있다.  
EM 방식은 의미상으로 동일한 SQL 쿼리가 다양하게 나올 수 있는데 다르게 평가하는 문제가 있고  
EX 방식은 쿼리를 실행할 수 있는 데이터베이스를 추가로 만들어야 하므로 여기서는 사용하지 않는다.  
최근에는 LLM을 활용해 LLM의 생성 결과를 평가하는 방식이 활발하다. 사람이 하기에는 한계가 있으므로 GPT-4를 사용한다.  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/6_1.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/6_1.png)   

## 평가 데이터셋 구축  
실습 데이터셋은 8개 데이터베이스(도메인)에 대해 생성했다.  
모델의 성능 확인을 위해 7개의 데이터베이스 데이터는 학습에 사용하고 1개는 평가에 사용한다.  
db_id=1인 테이블은 게임을 가정했는데 다른 도메인과 달리 테이블 이름이 특화된 이름이 사용되므로 이를 평가 데이터셋으로 활용한다.  

## SQL 생성 프롬프트  
LLM이 SQL을 생성하도록 하기 위해서는 프롬프트가 필요하다. 학습에 사용한 프롬프트 형식을 추론할 때도 동일하게 사용해야 품질이 좋기 때문에 모든 상황에서 동일하게 사용한다.  
프롬프트 만드는 건 코드가 필요하므로 코드를 보자.  

## GPT 평가 프롬프트와 코드 준비  
GPT를 사용해 평가를 수행한다면 API요청해야한다. (돈을 써야한다)  
나는 예전에 공부할 때 미리 충전해둔게 있어서 그걸로 해보려고한다.  
그러므로 코드를 보자.  

## 실습: 미세 조정 수행하기  
해당 실습에서는 한국어 사전 학습 모델 중 가장 높은 성능을 보이는 beomi/Yi-Ko-6B 모델을 사용한다. 
코드에 주석을 달아서 내가 이해한 것들을 정리했다.  
하지만 미세 조정 하는데 8~9시간 정도 걸린다고 해서 포기하고 책으로 어떤 결과가 있었는지 보면서 여기에 다시 정리한다.  
5장에서 배운 LORA도 사용하므로 LORA의 하이퍼 파라미터인 r과 alpha를 바꿔가며 성능 비교를 한다.  
둘다 적당한 트레이드 오프가 필요해서 적당한 값은 실험적으로 찾아야한다.  
원본 데이터 셋은 결측치가 여러개 있고 품질 관리를 안했다. 그래서 GPT를 통해 필터링해서 만개의 데이터를 걸러냈고 성능을 비교해보니  
원본 데이터로 학습한 성능과 거의 비슷했다.  
데이터 셋은 기본적으로 많을 수록 성능이 올라가는데 만개의 데이터가 사라져도 성능이 비슷하다는 것은 데이터의 품질 또한 성능에 영향을 준다는 것을 증명한다.  
기초 모델을 변경해서 사용해본다. 무료버전에서는 실행이 아예 불가능해서 책으로 결과만 봤고 성능이 높아졌다.  
