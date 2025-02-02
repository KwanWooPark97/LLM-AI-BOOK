# LLM 운영하기  
LLM을 서비스에 활용하려는 시도가 늘면서 LLMOps도 주목받고 있다. 이를 위해서 MLOps를 알아보고 이어서 알아보도록 한다.  

## MLOps  
MLOps(Machine Learning Operations)는 데브옵스(DevOps)의 개념을 머신러닝과 데이터 과학 분야로 확장한 방법론이다.  
MLOps의 목표는 데이터 수집, 전처리 모델 학습, 평가, 배포, 모니터링 등 머신러닝 프로젝트의 전 과정을 자동화하고 효율화하는 것이다.  
![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/13_1.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/13_1.png)    
MLOps의 핵심은 위 그림과 같이 데이터 수집부터 전처리, 학습, 평가, 배포에 이르는 전체 과정을 자동화하고 관리하는 머신러닝 파이프라인이라고 할 수 있다.   
특히, MLOps에서는 모델의 **재현성**을 보장하는 것이 매우 중요하다.  
재현성이란 이전에 수행된 ML 워크 플로를 그대로 반복했을 때 동일한 모델을 얻을 수 있는지 여부를 의미한다.  
모델 개발 과정의 모든 단계가 문서화되고 버전 관리되어야 이전 모델을 동일하게 재현할 수 있다.  
MLOps에서는 모델 학습을 자동으로 트리거하여 새로운 데이터로 지속적으로 모델을 업데이트한다.  
만약 성능이 떨어지면 자동으로 재학습을 실시하고 배포하도록 하여 항상 최적의 상태를 유지하도록 만든다.  
이는 데브옵스의 지속적 통합/배포(CI/CD) 개념을 머신러닝에 적용한 것이다.  

### 데이터 관리  
모델 학습을 위한 데이터 준비 과정에는 여러가지 중요한 의사결정이 포함되고 그 의사 결정에 따라 다양한 형태의 데이터셋이 생성된다.  
예를들어, 포함 시킬 데이터의 범위를 선택하고, 어떤 전처리 방식을 포함시키고, 특성 공학을 통해 어떤 특성을 추가할지 등 다양하게 많다.  
모델 학습 결과를 재현하기 위해서는 데이터셋의 버전을 관리하고 어떤 학습 데이터셋으로 모델을 학습시켰는지 기록해야 한다. 이를 위해 DVC(Data Version Control)와 같은 도구를 사용한다.  

### 실험 관리  
머신러닝 모델을 학습시킬 때는 어떤 모델을 사용할지 정해야한다.  
예를 들어, 분류 문제를 푼다면 로지스틱 회귀, svm, 트리 모델 같은 머신러닝 모델부터 딥러닝 모델까지 다양하게 존재   
만약 혼자 실험한다면 규칙을 정해서 정리하면 가능 하지만 협업한다면 어려움 따라서 이런 문제를 해결하기 위한 실험 관리 도구를 사용함  
대표적으로 MLflow, W&B가 있다.  

### 모델 저장소  
MLOps에서 모델 저장소(model registry)는 머신 러닝 모델을 체계적으로 관리하고 버전 제어하는 데 필수적인 요소다.  
당연히 실험을 진행하며 어려 머신러닝 파이프라인에서 다양한 모델이 생기는데 모델 저장소를 활용하면 다양한 모델을 하나로 통합해서 관리할 수 있다.  
모델 저장소를 사용하면 모델의 전체 수명 주기를 추적하고 통합해서 관리할 수 있다. 만약 모델에 문제가 생기면 이전 버전으로 모델을 배포하여 문제를 해결한다.  
모델 저장소와 연도앟여 모델을 서빙하고 배포하는 과정을 자동화할 수 있어 모델의 배포와 관리가 간소화된다.  
대표적인 모델 저장소 도구로는 MLflow 모델 저장소, AWS 세이지메이커 모델 저장소가 있다.  

### 모델 모니터링  
머신러닝 모델이 실제 환경에 배포된 이후에도 지속적으로 의도한 대로 작동하는지 모니터링하는 것은 매우 중요하다.  
평소에는 정상적으로 요청에 응답하고 있어도 한번 엉뚱한 값을 반환한다면 문제가 생긴다.  
특히 학습한 이후 오랜 시간이 흘렀으면 입력 데이터의 변동이 생길 수 있고 이는 성능에 영향을 끼친다.  
따라서 모델 배포 이후에 입력 데이터에 변화가 생기지 않는지 유심히 살펴봐야하고 운영 과정에서 배포된 모델이 바뀌지 않았는지 모델의 버전도 확인해야 한다.  
대표적인 모니터링 도구로는 프로메테우스, 그라파냐가 있고 AWS 세이지메이커도 가능하다.  

## LLMOps는 무엇이 다른가?  
LLMOps도 머신러닝 모델의 개발과 운영을 통합하려는 목적을 갖고 있기 때문에 MLOps와 비슷하지만 중요한 몇 가지가 다르다.  
LLM은 기존 머신러닝 모델에 비해 훨씬 크고 API기반으로 상업용 모델을 사용한다. 그리고 LLM은 생성형 모델이므로 결과물을 정량적으로 평가하기 어렵다.  

### 상업용 모델과 오픈소스 모델 선택하기   
MLOps에서 사용하는 기존 ML모델들은 하나의 문제를 해결하는 모델이였다.  
하지만 LLM 모델들은 대부분 하나이상의 문제를 해결하기 위해 만들어지는 경우가 많다.  
즉, LLMOps에서는 MLOps보다 훨씬 크고 다양한 일을 할 수 있는 모델을 다룬다.  
LLM은 API를 사용할 수 있는 상업용 모델과 직접 모델을 학습시키고 실행해 활용할 수 있는 오픈소스 모델로 양분됐다.  

#### 상업용 모델  
상업용 모델은 일반적으로 성능이 높고 API를 통해 쉽게 사용 할 수 있다.  
하지만 사용 모델의 버전이나 업데이트에 따라 모델의 성능이 바뀌거나 잘 작동하던 프롬프트가 작동하지 않는 등 문제가 발생할 수 있어 지속적인 모니터링이 필요하다.  
또한 사용하는 만큼 토큰 당 비용이 발생하므로 비용 관리가 쉽지않다.  
대부분의 상업용 모델은 미세 조정 기능을 지원하지 않거나 일부 모델만 지원하기 때문에 사용자는 프롬프트나 일부 하이퍼파라미터로만 모델을 커스터마이징할 수 있다.  

#### 오픈소스 모델   
오픈소스 모델은 직접 모델 파라미터를 변경할 수 있기 때문에 자유롭게 미세 조정이 가능하고 필요한 기능 요구사항메 맞춰 추론 속도를 개선할 수 있다.  
하지만 추록 인프라를 관리해야 하고 미세 조정을 위해 인력과 실험을 필요하기 때문에 기술적 난이도가 높이 필요함  
오픈 소스 모델은 상업용 모델에 비해 크기가 작고 성능이 떨어지지만 미세 조정할 경우 비용 효율적이면서도 성능이 높을 수 있다.  

----  

모델의 크기는 추론할 때 가장 큰 문제가 된다.  
모델이 크면 추론에 더 많은 GPU를 사용해야하고 GPU를 많이 사용하면 비용이 발생한다. 따라서 모델의 용량을 줄여야한다.  
앞에서 양자화와 지식 증류를 공부했으니 이런 방법을 사용한다 설명하고 있음(7장)  

모델의 크기는 변경하지 않고 추론 방식을 변경해 처리량을 높이고 지연시간을 낮추는 방식이 있다.  
연속 배치 방법을 설명함  

어텐션 연산을 효율적으로 수행하도록 바꾸는 플래시 어텐션을 설명함  
페이지 어텐션을 통해 GPU 메모리에서 KV 캐시를 효율적으로 사용하는 방법 설명함(8장)  

### 모델 최적화 방법의 변화  

LLM을 사용 사례에 맞게 최적화 하는 방법은 크게 사전 학습, 미세 조정, 프롬프트 엔지니어링, 검색 증강 생성 네 가지로 나눌 수 있다.   
LLMOps 에서 다루는 LLM은 모델의 크기가 크기 때문에 일반적으로 사전 학습시키는 경우는 거의 없다.   
오픈 소스 모델은 미세 조정이 자유롭지만 상업용 모델은 지원하는 모델만 제한적으로 가능하다.   
프롬프트를 만들어 내가 원하는 출력 형식을 내도록 모델을 만들 수 있다.   
프롬프트에 답변 생성에 필요한 정보를 추가하는 RAG를 사용할 수도 있다.   

MLOps 경우는 머신 러닝이나 작은 딥 러닝 모델을 사용하므로 학습을 직접 시켰지만 LLM의 경우 모델이 너무 커서 대부분 사전 학습된 모델을 가져온다.  
이때 미세 조정은 4장에서 본 지도 미세 조정이나 DPO같은 방법으로 추가 학습한다.  
모델이 크기 때문에 모든 파라미터를 학습 할 수 없으므로 일부 파라미터만 학습하는 LORA나 QLORA같은 방법을 통해 적은 GPU로도 LLM학습이 가능해졌다.(5장)  

LLM은 입력으로 넣는 프롬프트에 따라 다양한 작업을 수행할 수 있다. 또한 프롬프트를 어떻게 구조화하느냐에 따라 동일한 요청에 대해서도 다른 품질의 결과를 생성한다.  
MLOps에서는 모델 개발 과정에서 학습할 때 설정한 하이퍼파라미터를 기록해 뒀다 LLMOps의 경우 프롬프트도 하나의 하이퍼 파라미터로 간주하여 실험의 대상이 된다. 

9~12장을 통해 살펴본 RAG도 프롬프트에 최신 정보나 답변에 필요한 정보를 추가해 모델을 최적화 하는 방법이다.  
RAG는 프롬프트를 보강하기 위해 임베딩 모델과 벡터 데이터베이스 같은 추가 요소가 필요하다.  

### LLM 평가의 어려움  
LLM의 경우 머신러닝 모델과 달리 다양한 작업을 수행할 수 있다.  
ML 모델의 경우 정량적인 지표를 통해 평가할 수 있지만 LLM의 경우 다양한 작업이 가능하기 때문에 특정 작업의 성능 평가 방식으로 모두 평가할 수 없고  
프롬프트에 따라 성능이 달라지기도 해서 명확한 기준을 잡기 어렵다.  
이런 어려움으로인해 LLM 모델의 평가는 아직 완전히 풀리지 않은 문제이다.  

## LLM 평가하기  
모델의 평가는 개발 과정에서는 모델이 개선됐는지 확인하고 배포를 해도 될지 결정하는 요소이고 배포 이후에도 모델이 재학습이 필요한지 확인하는 중요한 요소이다.  
하지만 LLM 모델은 평가하기 어렵다.  

### 정량적 지표  
텍스트 생성 작업을 평가할 때 사용할 수 있는 대표적인 세 가지 정량 지표가 있다.  
1. 번역 작업을 평가할 때 사용하는 BLEU는 번역 결과와 사람의 결과의 유사도를 측정하여 평가한다.  
   n-gram 기반으로 모델이 생성한 문장과 참조 문장의 정밀도를 계산한다.  
2. 요약이나 번역 등 자연어 생성 모델의 성능 평가에 사용하는 ROUGE는 모델이 생성한 요약문과 사람이 작성한 참조 요약문 사이의 n그램 중복도를 재현율 관점에서 측정한다.
3. 펄플렉시티는 모델이 새로운 단어를 생성할 때의 불확실성을 수치화한 것으로, 값이 낮을수록 모델의 예측 성능이 우수하다는 의미다.

세 가지 정량 지표 모두 빠르게 성능을 평가할 수 있다는 장점이 있지만 질적인 측면의 평가에는 한계가 있고 실제 사람의 주관적 판단과 불일치하는 경우가 많다.  
따라서 이런 정량 지표들은 언어 모델 평가 시 참고는 할 수 있으나, 절대적인 잣대로 삼기에는 무리가 있다.  

### 벤치마크 데이터셋을 활용한 평가  

앞에서 본 정량 지표는 평가에 사용하는 데이터셋에 따라 결과가 달라지기 때문에 서로 다른 모델의 성능을 비교할 때는 사용하기 어렵다.  
다양한 모델의 성능을 비교하기 위해 공통으로 사용하는 데이터셋을 벤치마크 데이터셋이라고 부른다.  
대표적으로 허깅페이스 리더 보드에서 사용하는 ARC,HellaSwag,MMLU,TruthfulQA 등이 있다.  
책을 보면 데이터 셋의 예시가 나와있음  
ARC는 사지선다형과학 문제다.  
HellaSwag는 다음으로 연결된 문장을 4개의 보기 중에 고르는 문제다.  
MMLU는 다양한 과제에 대한 모델의 성능을 평가하기 위한 데이터셋이다. 57개 분야에 대한 사지선다형 문제다.  
TruthfulQA는 신뢰할 수 있는 모델인지 확인하기 위한 질문으로 이루어져있다.  

한국어 LLM 리더보드 평가 데이터셋도 존재한다. 업스테이지가 발표한 것도 있고 W&B에서 발표한 호랑이 도 존재한다.  

### 사람이 직접 평가하는 방식  

앞에서 본대로 정량적 지표의 경우 빠르게 모델의 성능을 평가할 수 있다.  
하지만 결국 사람이 평가하는게 최고라고 생각해서 사람이 보완해야한다.  
개발 단계에서는 개발에 참여하는 구성원이 입력과 출력을 확인하면서 평가하고 별도의 평가자가 있는 경우도 있다.  
대신 시간이 오래 걸리고 비용이 많이 든다는 단점이 있다.  

#### LLM을 통한 평가  
벤치마크 데이터셋을 통해서 언어 모델이 '얼마나 똑똑한지'를 확인할 수 있으나 '사람의 요청에 얼마나 잘 대응하는지'에 대해서는 판단하지 못한다.   
사람의 선호를 잘 반영하고 멀티 턴 대화에서 사람의 요청과 잘 정렬된 대답을 하는지 확인하기 위해서 LLM을 평가자로 활용하는 방법을 발표했다.  
이 연구에서는 80개의 선별한 멀티 턴 질문 데이터인 MT-Bench와 챗봇 아레나 데이터를 활용한다.  
(멀티 턴 질문 데이터는 대화형 시스템(챗봇)에서 다양한 주제를 다룰 수 있도록 구성한 예시입니다. 각 질문은 이전 질문과 답변을 기반으로 발전하거나, 추가적인 정보를 요구합니다.) 

----
질문 1: "딥러닝에서 overfitting을 방지하는 방법이 있나요?"  
답변: "Overfitting을 방지하려면 정규화(L2 Regularization), 드롭아웃(Dropout), 데이터 증강(Data Augmentation) 등을 사용할 수 있습니다."  

질문 2: "그중 드롭아웃은 어떻게 작동하나요?"  
답변: "드롭아웃은 학습 중 특정 뉴런을 무작위로 비활성화하여 네트워크가 특정 패턴에 과적합되지 않도록 합니다."  

질문 3: "데이터 증강은 어떤 방식으로 구현할 수 있나요?"  
답변: "데이터 증강은 이미지 데이터를 회전, 자르기, 밝기 조절, 뒤집기 등으로 변형하여 학습 데이터를 증가시키는 방식입니다."  

----  

챗봇 아레나 데이터는 서로 다른 두 챗봇 모델이 동일한 질문에 어떻게 답변하는지를 비교하는 시나리오를 나타냅니다.  
각 챗봇의 응답을 통해 모델의 특징, 강점, 약점을 평가할 수 있습니다.   

----  
질문: "피곤함을 줄이고 에너지를 높이는 방법이 있을까요?"  

챗봇 A:  
"피곤함을 줄이려면 충분한 수면과 규칙적인 운동이 중요합니다. 또한, 물을 충분히 마시고 단백질이 풍부한 음식을 섭취하면 에너지를 높이는 데 도움이 됩니다."  

챗봇 B:  
"수면의 질을 높이고 카페인을 줄이는 것이 피로 회복에 좋습니다. 또한, 하루에 30분 정도 산책하거나 가벼운 유산소 운동을 하면 에너지가 회복될 수 있습니다."  

----

이렇게 사람이 직접 평가하는 방법에 LLM을 사용하면 비용과 시간을 확실하게 줄일 수 있다.  
또한 LLM은 이렇게 평가한 이유를 설명하도록 만들어서 LLM 평가자를 개선하거나 평가 기준을 정립하는 등 다양한 목적에 활용할 수 있다.  

#### RAG 평가  
RAG 방식의 성능도 평가하기 어렵다. RAG는 문서를 찾기 위한 검색 단계와 검색한 내용을 바탕으로 생성하는 단계로 나눌 수 있다.  
따라서 각 단계에 대한 성능을 평가하고 최초 요청과 최종 생성 결과를 기바능로 평가하기 위해 아래 그림처럼 세 가지 측면에서 RAG 성능을 평가할 수 있다.  

![https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/13_2.png](https://github.com/KwanWooPark97/LLM-AI-BOOK/blob/main/img/13_2.png)    

신뢰성(faithfulness): 생성된 응답이 검색된 맥락 데이터에 얼마나 사실적으로 부합하는지 평가  
답변 관련성(answer relevancy): 생성된 답변이 요청과 얼마나 관련성이 있는지 평가  
맥락 관련성(context relevancy): 검색 결과인 맥락 데이터가 요청과 얼마나 관련 있는지 평가  

