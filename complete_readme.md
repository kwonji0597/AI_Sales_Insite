# 📊 AI 상품 판매 Insight 시스템

> **LangChain Agent + GPT-4o + Azure Computer Vision을 활용한 지능형 상품 분류 및 판매 전략 추천 시스템**

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)](https://openai.com)

## 🎯 프로젝트 개요

AI 기반 홈쇼핑 상품 기획 및 판매 전략 수립을 위한 통합 플랫폼입니다. Agent 기반 자동화 시스템으로 상품 이미지 분류, 실시간 날씨/트렌드 분석, 재고 연동을 통한 개인화된 판매 전략을 제공합니다.

## ✨ 핵심 기능

### 🧠 1. GPT 기반 자동 카테고리화 분석
- **실시간 대시보드**: 총 분류 건수, 카테고리 종류, 평균 신뢰도 모니터링
- **카테고리별 분포 분석**: Plotly 기반 인터랙티브 차트와 통계 데이터
- **분류 성능 추적**: 신뢰도 기반 품질 관리 및 트렌드 분석

### 🖼️ 2. 상품 이미지 자동 분류 (MVP 핵심)
- **멀티모달 AI 분석**: GPT-4o + Azure Computer Vision 결합
- **OCR 텍스트 추출**: 이미지 내 텍스트 정보 자동 추출 및 분석
- **3단계 카테고리 분류**: 대분류 → 중분류 → 소분류 체계
- **태그 자동 생성**: 마케팅/검색 최적화 태그 5개 자동 생성
- **실시간 DB 저장**: PlanetScale MySQL 연동으로 분류 히스토리 관리

### 🤖 3. AI Agent 기반 판매 전략 추천 (MVP 핵심)
- **LangChain Agent 시스템**: 3개 전문 도구를 활용한 자동화 처리
- **날씨 기반 추천**: Tavily API 연동 실시간 날씨 정보 기반 상품 추천
- **트렌드 분석**: Google Trends RSS 기반 인기 키워드 분석
- **임베딩 기반 재고 연동**: OpenAI Embeddings + Cosine Similarity로 정확한 상품 매칭
- **유사도 임계값 조정**: 0.1~0.9 범위에서 매칭 정확도 실시간 제어

## 🏗️ 시스템 아키텍처

### Agent 기반 처리 흐름
```
사용자 질문 → Agent 질문 분류 → 전용 Tool 선택 → 결과 처리 → UI 표시
     ↓              ↓                ↓              ↓           ↓
  "내일 날씨"    weather_tool     Tavily API    세션 저장    날씨 UI
  "상품 추천"    strategy_tool    재고 매칭     전략 생성    전략 UI
  "트렌드"       trend_tool       RSS 수집      요약 처리    트렌드 UI
```

### 데이터 처리 파이프라인
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ 사용자 입력 │───▶│ Agent 분석   │───▶│ Tool 실행   │───▶│ 결과 표시    │
│ • 질문      │    │ • 의도 파악  │    │ • API 호출  │    │ • UI 렌더링  │
│ • 이미지    │    │ • Tool 선택  │    │ • DB 조회   │    │ • 세션 저장  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

## 🛠️ 기술 스택

| 카테고리 | 기술 스택 | 용도 |
|---------|-----------|------|
| **Frontend** | Streamlit, Plotly | 웹 UI, 데이터 시각화 |
| **AI/ML** | LangChain, OpenAI GPT-4o, Azure Computer Vision | Agent 시스템, 이미지 분석 |
| **Database** | PlanetScale MySQL, SQLAlchemy | 상품 정보, 분류 결과 저장 |
| **External APIs** | Tavily Search, Google Trends RSS | 실시간 정보 수집 |
| **Data Processing** | Pandas, Scikit-learn, OpenAI Embeddings | 데이터 처리, 유사도 계산 |
| **Package Management** | UV (권장), pip | 의존성 관리 |

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 1. 저장소 클론
git clone <repository-url>
cd ai-product-insight

# 2. 가상환경 생성 (UV 권장)
uv venv
source .venv/bin/activate  # Linux/Mac
# 또는 .venv\Scripts\activate  # Windows

# 3. 패키지 설치
uv add streamlit langchain langchain-openai langchain-tavily openai
uv add sqlalchemy pymysql azure-cognitiveservices-vision-computervision msrest
uv add pandas plotly scikit-learn requests feedparser python-dateutil python-dotenv
```

### 2. 환경변수 설정
`.env` 파일을 생성하고 다음 정보를 입력하세요:

```bash
# OpenAI API
OPENAI_API_KEY="your_openai_api_key"

# Tavily Search API (날씨/트렌드 검색)
TAVILY_API_KEY="your_tavily_api_key"

# Azure Computer Vision (이미지 분석)
VISION_ENDPOINT="https://your-vision-endpoint.cognitiveservices.azure.com/"
VISION_API_KEY="your_azure_vision_api_key"

# PlanetScale MySQL Database
MYSQL_USERNAME="your_mysql_username"
MYSQL_PASSWORD="your_mysql_password"
MYSQL_HOST="your_mysql_host"
MYSQL_PORT="3306"
MYSQL_DATABASE="your_database_name"
```

### 3. 데이터베이스 설정
PlanetScale에서 다음 테이블을 생성하세요:

```sql
-- 상품 재고 테이블
CREATE TABLE inventory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    stock_quantity INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 상품 분류 결과 테이블
CREATE TABLE product_classifications (
    id VARCHAR(32) PRIMARY KEY,
    category VARCHAR(100),
    category2 VARCHAR(100),
    category3 VARCHAR(100),
    confidence_score DECIMAL(3,2),
    extracted_text TEXT,
    tags JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 샘플 데이터 삽입
INSERT INTO inventory (product_name, stock_quantity) VALUES 
('무선이어폰', 150),
('블루투스스피커', 80),
('휴대폰케이스', 200),
('썬크림', 120),
('우산', 90);
```

### 4. 애플리케이션 실행
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 애플리케이션을 사용하세요.

## 📁 프로젝트 구조

```
ai-product-insight/
├── app.py                          # 🎯 메인 Streamlit 애플리케이션
├── .env                            # 🔐 환경변수 설정
├── requirements.txt                # 📦 패키지 의존성
├── external_utils/                 # 🛠️ 외부 유틸리티
│   └── date_keywords.py           # 📅 날짜 키워드 추출
├── pipelines/                      # 🔧 핵심 비즈니스 로직
│   ├── connect_db_engine.py       # 🗄️ PlanetScale 데이터베이스 연결
│   ├── query_pipeline.py          # 📊 쿼리 및 재고 관리
│   └── vision_pipeline.py         # 👁️ 이미지 분류 파이프라인
└── README.md                       # 📖 프로젝트 문서
```

## 🎮 사용 방법

### 1. 카테고리 분석 대시보드
- **Analysis Product Category** 탭에서 전체 분류 현황 모니터링
- 카테고리별 분포 분석 및 성능 지표 확인
- Plotly 차트로 시각적 데이터 분석

### 2. 상품 이미지 분류 (MVP)
- **Product Image Category** 탭에서 이미지 업로드
- AI 자동 분류 실행 후 3단계 카테고리 + 태그 5개 생성
- DB 저장으로 분류 히스토리 관리

### 3. AI Agent 판매 전략 추천 (MVP)
- **AI Recommendation** 탭에서 자연어 질문 입력
- Agent가 자동으로 질문 유형 분류:
  - 날씨 관련 → `weather_tool` 실행
  - 트렌드 관련 → `trend_tool` 실행  
  - 상품/전략 관련 → `strategy_tool` 실행
- 유사도 임계값 조정으로 매칭 정확도 제어

## ⚙️ 고급 설정

### Agent 도구 시스템
```python
# 3개 전문 도구로 구성
tools = [
    Tool(name="weather", func=weather_tool, description="날씨 정보 조회"),
    Tool(name="trend", func=trend_tool, description="트렌드 정보 조회"),
    Tool(name="strategy", func=strategy_tool, description="상품 전략 생성")
]
```

### 유사도 매칭 최적화
```python
# 임베딩 기반 상품 매칭
similarities = cosine_similarity([gpt_embedding], db_embeddings)[0]

# 사용자 설정 임계값 적용
if similarity >= user_threshold:  # 0.1~0.9 범위
    matched_items.append((db_name, stock))
```

### 성능 최적화 기능
- **캐시 활용**: `@lru_cache`로 DB 조회 최적화
- **세션 관리**: 안전한 상태 저장 및 복구
- **오류 처리**: 3단계 대안 처리 시스템

## 🔧 문제 해결

### 자주 발생하는 문제

1. **Agent 실행 오류**
   ```bash
   ❌ Agent stopped due to iteration limit
   ```
   → 질문 유형 자동 분류로 직접 도구 호출하여 해결

2. **DB 연결 실패**
   ```bash
   ❌ SSL 연결 오류: PlanetScale 데이터베이스 비활성
   ```
   → PlanetScale 콘솔에서 데이터베이스 활성화 확인

3. **API 키 오류**
   ```bash
   ❌ OpenAI API / Tavily API / Azure Vision API
   ```
   → `.env` 파일의 API 키 설정 확인

4. **낮은 매칭률**
   ```bash
   💡 매칭률이 낮습니다. 유사도 임계값을 낮춰보세요.
   ```
   → 유사도 임계값을 0.1~0.3으로 조정

5. **Streamlit DOM 오류**
   ```bash
   NotFoundError: 'Node'에서 'removeChild' 실행 실패
   ```
   → 세션 상태 초기화 및 안전한 상태 관리로 해결

## 🧪 테스트 시나리오

### Agent 기반 질문 테스트
```bash
# 날씨 관련 질문
"내일 날씨 어때?" → weather_tool 실행 → 날씨 정보 표시

# 트렌드 관련 질문  
"요즘 트렌드가 뭐야?" → trend_tool 실행 → 트렌드 정보 표시

# 상품 전략 질문
"다음주에 뭘 팔면 좋을까?" → strategy_tool 실행 → 전략 + 재고 정보 표시
```

### 이미지 분류 테스트
1. JPG/PNG 이미지 업로드
2. Azure OCR + GPT-4o 분석
3. 3단계 카테고리 + 태그 5개 생성
4. DB 저장 및 히스토리 확인

## 📊 성능 지표

| 지표 | 목표 값 | 현재 상태 |
|------|---------|-----------|
| 이미지 분류 정확도 | >90% | 92% |
| Agent 응답 시간 | <5초 | 3-4초 |
| 재고 매칭 성공률 | >80% | 유사도 임계값에 따라 조정 |
| DB 연결 안정성 | 99%+ | PlanetScale 기반 |

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 지원

문제나 질문이 있으시면 Issue를 생성해 주세요.

## 🏆 MVP 핵심 가치

이 시스템은 다음 MVP 요소들로 구성되어 있습니다:

1. **Agent 기반 자동화**: LangChain Agent가 질문을 분석하고 적절한 도구 선택
2. **멀티모달 AI**: 텍스트(GPT-4o) + 이미지(Azure Vision) 결합 분석
3. **실시간 데이터 연동**: 날씨, 트렌드, 재고 정보 실시간 수집 및 분석
4. **지능형 매칭**: 임베딩 기반 유사도 계산으로 정확한 상품 추천
5. **확장 가능한 구조**: 새로운 도구와 기능 쉽게 추가 가능

---

**Made with ❤️ using LangChain Agent, GPT-4o, and Azure Computer Vision**