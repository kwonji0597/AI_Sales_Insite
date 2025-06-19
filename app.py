# 실행: streamlit run app.py

import streamlit as st
import os
import re
import pandas as pd  # 🆕 추가
import plotly.express as px  # 🆕 추가 pip install pandas plotly or uv add pandas plotly
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity #필수
from langchain_tavily import TavilySearch
from collections import defaultdict 
from sqlalchemy import text
from external_utils.date_keywords import get_date_keywords
from pipelines.query_pipeline import get_all_inventory_from_db
from pipelines.connect_db_engine import create_db_engine, connection_db
from pipelines.vision_pipeline import (
    classify_product_image, 
    save_classification_to_db, 
    get_recent_classifications
)

# 1. 세션 상태 안전하게 초기화 (app.py 상단에 추가)
if 'weather_result' not in st.session_state:
    st.session_state.weather_result = None
if 'trend_result' not in st.session_state:
    st.session_state.trend_result = None
if 'strategy_result' not in st.session_state:
    st.session_state.strategy_result = None
if 'inventory_info' not in st.session_state:
    st.session_state.inventory_info = {}

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
VISION_API_KEY = os.getenv("VISION_API_KEY")

engine = create_db_engine()

# 모델 초기화
llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.3)
tavily = TavilySearch(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# 유틸리티 함수들
def get_weather_and_trends(user_question: str):
    if not tavily:
        return "날씨 정보 없음", "트렌드 정보 없음"
    
    weather_query, trend_query = get_date_keywords(user_question)
    weather_info = tavily.run(weather_query) # TAVILY 사용
    trend_info = tavily.run(trend_query)
    return weather_info, trend_info


def inventory_match_via_embedding(cleaned_gpt_product_names, similarity_threshold=0.35):

    # 전체 DB 재고 딕녀너리
    db_inventory_dict = get_all_inventory_from_db()
    print(f"DB 재고 정보: : {db_inventory_dict} ")
    result = defaultdict(list)

    if not db_inventory_dict:
        for gpt_name in cleaned_gpt_product_names:
            result[gpt_name] = [(gpt_name, 0)]
        return result
    
    db_names = list(db_inventory_dict.keys())
    print(f"DB 상품명 목록: : {db_names} ")

    # DB 상품명들의 임베딩 생성
    db_embeddings = embedding_model.embed_documents(db_names)
    
    # 각 GPT 추천 상품에 대해 매칭 수행 (gpt_name(gpt에서 받은 추천상품))
    for gpt_name in cleaned_gpt_product_names:

        # gpt_name 임베딩
        gpt_embedding = embedding_model.embed_query(gpt_name)

        # 유사도 계산
        similarities = cosine_similarity([gpt_embedding], db_embeddings)[0]
        matched_items = []

        for i, sim in enumerate(similarities):

            # 🔧 수정: 하드코딩된 0.30을 매개변수로 변경
            # if sim >= 0.3:  # 🆕 사용자 설정 유사도 사용
            if sim >= similarity_threshold:  # 기존: if sim >= 0.30:
                db_name = db_names[i]
                stock = db_inventory_dict[db_name]
                matched_items.append((db_name, stock))
                print(f"   ✅ 매칭: {gpt_name} → {db_name} (유사도: {sim:.3f})")  # 🆕 매칭 로그


        if not matched_items:
            result[gpt_name] = [(gpt_name, 0)]
            print(f"   ❌ 매칭 실패: {gpt_name} (최고 유사도: {max(similarities):.3f})")  # 🆕 실패 로그
        else:
            result[gpt_name] = matched_items

    return result


# 날씨 및 트렌드에 맞는 상품명 추출
def product_name_extract(user_question: str, weather_info: str, trend_info: str, similarity_threshold=0.3) -> dict:
    product_extraction_prompt = f"""
        사용자 질문: {user_question}    
        날씨 정보: {weather_info}
        트렌드 정보: {trend_info}

        위 정보를 참고하여 상품명을 5개 제시해주세요.
        상품명만 한 줄씩 나열해 주세요.
        
        예시:
        - 방수 우산 : 방수로 처리된 우산
        - 썬크림 
        - 텀블러
        - 에코백
        - 다이어리
    """
    
    response = llm.invoke(product_extraction_prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    
    raw_product_names = [line.strip("-• ").strip() for line in response_text.split('\n') if line.strip()]
    cleaned_gpt_product_names = [re.sub(r"^[0-9]+[\.)]?\s*", "", name).replace(" ", "") for name in raw_product_names]

    return inventory_match_via_embedding(cleaned_gpt_product_names)
   
def generate_strategic(user_question: str, weather_info: str, trend_info: str, inventory_infos: dict):
    inventory_lines = []
    for gpt_name, db_matches in inventory_infos.items():
        if db_matches:
            for db_name, qty in db_matches:
                inventory_lines.append(f"{db_name} (재고: {qty}개)")
        else:
            inventory_lines.append(f"{gpt_name} (재고: 0개)")       
                                
    inventory_text = "\n".join(inventory_lines)

    return f"""
            너는 홈쇼핑 방송 상품 기획 전문가야.

            사용자 질문: '{user_question}'
            날씨 정보: {weather_info}
            트렌드 정보: {trend_info}

            [추천 상품별 재고 정보]
            {inventory_text}

            위 형식으로 방송 전략을 구체적으로 작성해줘.
            """

# Streamlit UI 설정
st.set_page_config(page_title="📊 AI 상품 판매 Insite", layout="wide")
st.markdown("<h3 style='text-align: center;'>📊 AI 상품 판매 Insite</h3>", unsafe_allow_html=True)

# 탭 생성
tabs = st.tabs(["Analysis Product Category", "Product Image Category", "AI Recommendation"])

# 카테고리 상품 분석
with tabs[0]:
    st.header("🧠 GPT 기반 자동 카테고리화 분석")
    
    # ========================================
    # 1️⃣ 기본 통계 대시보드 (MVP 핵심 기능)
    # ========================================
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # DB에서 기본 통계 조회
        if engine:
            with engine.connect() as conn:
                # 전체 분류 건수
                total_query = text("SELECT COUNT(*) FROM product_classifications")
                total_count = conn.execute(total_query).fetchone()[0]
                
                # 카테고리 종류 수
                category_query = text("SELECT COUNT(DISTINCT category) FROM product_classifications")
                category_count = conn.execute(category_query).fetchone()[0]
                
                # 평균 신뢰도
                confidence_query = text("SELECT AVG(confidence_score) FROM product_classifications WHERE confidence_score IS NOT NULL")
                avg_confidence = conn.execute(confidence_query).fetchone()[0] or 0
                
                # 오늘 분류 건수
                today_query = text("SELECT COUNT(*) FROM product_classifications WHERE DATE(created_at) = CURDATE()")
                today_count = conn.execute(today_query).fetchone()[0]
                
        else:
            # DB 연결 실패시 기본값
            total_count, category_count, avg_confidence, today_count = 0, 0, 0, 0
            
    except Exception as e:
        st.error(f"❌ 통계 조회 실패: {e}")
        total_count, category_count, avg_confidence, today_count = 0, 0, 0, 0
    
    # 메트릭 표시
    with col1:
        st.metric("📊 총 분류 건수", f"{total_count:,}")
    with col2:
        st.metric("🏷️ 카테고리 종류", f"{category_count}")
    with col3:
        st.metric("🎯 평균 신뢰도", f"{avg_confidence:.2f}" if avg_confidence else "0.00")
    with col4:
        st.metric("📅 오늘 분류", f"{today_count}")
    
    st.divider()
    
    # ========================================
    # 2️⃣ 카테고리별 분포 분석 (실용적 인사이트)
    # ========================================
    # col_left, col_right = st.columns([1, 1])
    # with col_left:
    st.subheader("📈 카테고리별 분류 현황")
    
    if st.button("🔄 카테고리 분포 분석"):
        try:
            with engine.connect() as conn:
                # 카테고리별 통계 쿼리
                category_stats_query = text("""
                    SELECT 
                        category,category2, category3, tags, 
                        COUNT(*) as count,
                        AVG(confidence_score) as avg_confidence,
                        MAX(created_at) as last_classified
                    FROM product_classifications 
                    GROUP BY category,category2, category3, tags
                    ORDER BY count DESC
                """)
                
                result = conn.execute(category_stats_query).fetchall()
                
                if result:
                    # 결과를 DataFrame으로 변환
                    df = pd.DataFrame(result, columns=['카테고리', '카테고리2', '카테고리3', 'TAG', '분류수', '평균신뢰도', '마지막분류'])
                    
                    # 데이터 표시
                    st.dataframe(
                        df.style.format({
                            '분류수': '{:,}',
                            '평균신뢰도': '{:.2f}',
                            '마지막분류': lambda x: str(x)[:19] if x else ''
                        }),
                        use_container_width=True,
                        height=300
                    )
                    
                    # 간단한 차트도 표시
                    fig = px.bar(
                        df.head(8), 
                        x='카테고리', 
                        y='분류수',
                        title="카테고리별 분류 건수 (상위 8개)",
                        color='평균신뢰도',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("📭 분류된 데이터가 없습니다.")
                    
        except Exception as e:
            st.error(f"❌ 카테고리 분석 실패: {e}")
    

# Product Image Category Tab (개선된 DB 저장 기능)
with tabs[1]:
    st.subheader("🖼️ 상품 이미지 분류")
    
    # DB 상태 확인
    db_status = connection_db()
    if db_status["success"]:
        st.success(f"✅ {db_status['message']}")
    else:
        st.error(f"❌ {db_status['error']}")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader(
        "이미지 파일을 업로드하세요",
        type=["jpg", "jpeg", "png"],
        help="JPG, PNG 형식만 지원"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # 이미지 표시
            st.image(uploaded_file, caption=f"업로드된 이미지: {uploaded_file.name}", width=300)
            st.write(f"파일 크기: {uploaded_file.size:,} bytes")
        
        with col2:
            # 분류 실행
            if st.button("🔍 이미지 분류 실행", type="primary"):
                with st.spinner("분류 중..."):
                    try:
                        # 이미지 분류 실행
                        result = classify_product_image(uploaded_file)

                        if "error" in result:
                            st.error(f"분류 실패: {result['error']}")
                        else:
                            # 결과를 세션에 저장
                            st.session_state.last_result = result
                            
                            # 결과 표시
                            st.success("✅ 분류 완료!")
                            
                            # 메트릭 표시
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("카테고리", result['category'])
                            with col_b:
                                st.metric("신뢰도", f"{result['confidence_score']:.2f}")
                            
                            # 태그 표시
                            st.write("**태그:**", " | ".join(result['tags']))
                            
                            # 추출된 텍스트
                            if result['extracted_text']:
                                st.write("**추출된 정보:**", result['extracted_text'])
                    
                    except Exception as e:
                        st.error(f"분류 중 오류: {e}")
            
            # DB 저장 버튼 (분류 결과가 있을 때만 표시)
            if 'last_result' in st.session_state and st.session_state.last_result.get('db_ready'):
                st.markdown("---")
                
                if st.button("💾 DB에 저장", type="secondary"):
                    result = st.session_state.last_result
                    
                    with st.spinner("DB 저장 중..."):
                        # DB에 저장
                        save_result = save_classification_to_db(
                            result['image_id'],
                            result['category'],
                            result['category2'],
                            result['category3'],
                            result['confidence_score'],
                            result['extracted_text'],
                            result['tags']
                        )
                        
                        if save_result["success"]:
                            st.success(f"✅ {save_result['message']}")
                            # 저장 후 결과 클리어
                            if 'last_result' in st.session_state:
                                del st.session_state.last_result
                            st.rerun()
                        else:
                            st.error(f"❌ {save_result['error']}")

    # 최근 분류 히스토리 (간단 버전)
    if st.checkbox("📈 최근 분류 결과 보기"):
        recent_data = get_recent_classifications(limit=5)
        
        if recent_data:
            st.write("**최근 5개 분류:**")
            for i, item in enumerate(recent_data, 1):
                st.write(f"{i}. **{item['category']}** (신뢰도: {item['confidence']:.2f}) - {item['created_at']}")
        else:
            st.info("분류 기록이 없습니다.")

###########################################################
# Agent 오류 수정 - tabs[2] 전체 재구현

###########################################################
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# 간단하고 직접적인 Tool 함수들

def weather_tool(query):
    """날씨 정보를 조회하는 도구"""
    try:
        weather_info, trend_info = get_weather_and_trends(query)
        
        # 날씨 정보 요약 처리
        if weather_info and len(str(weather_info)) > 200:
            try:
                weather_summary_prompt = f"""
                다음 날씨 정보를 간단히 요약해주세요:
                {weather_info}
                
                형식:
                날짜: 
                기온: 
                날씨: 
                """
                weather_summary = llm.invoke(weather_summary_prompt)
                weather_info = weather_summary.content
            except:
                weather_info = str(weather_info)[:300] + "..."
        
        # 🔧 수정: 안전한 세션 상태 업데이트
        st.session_state.weather_result = str(weather_info) if weather_info else "날씨 정보 없음"
        st.session_state.trend_result = str(trend_info) if trend_info else "트렌드 정보 없음"
        
        return "WEATHER_SUCCESS"
    
    except Exception as e:
        print(f"weather_tool 오류: {e}")
        # 🔧 추가: 오류 시에도 세션 상태 안전하게 설정
        st.session_state.weather_result = f"날씨 조회 실패: {str(e)}"
        st.session_state.trend_result = "트렌드 정보 없음"
        return f"WEATHER_ERROR: {str(e)}"    
    

def trend_tool(query):
    """날씨 정보를 조회하는 도구"""
    try:
        weather_info, trend_info = get_weather_and_trends(query)
        
        # 날씨 정보 요약 처리
        if trend_info and len(str(trend_info)) > 200:
            try:
                trend_summary_prompt = f"""
                다음 트렌드 정보를 간단히 요약해주세요:
                {trend_info} 
                """
                trend_summary = llm.invoke(trend_summary_prompt)
                trend_info = trend_summary.content
            except:
                trend_info = str(trend_info)[:300] + "..."
        
        # 🔧 수정: 안전한 세션 상태 업데이트
        st.session_state.weather_result = str(weather_info) if weather_info else "날씨 정보 없음"
        st.session_state.trend_result = str(trend_info) if trend_info else "트렌드 정보 없음"
        
        return "TREND_SUCCESS"
    
    except Exception as e:
        print(f"trend_tool 오류: {e}")
        # 🔧 추가: 오류 시에도 세션 상태 안전하게 설정
        st.session_state.weather_result = f"날씨 조회 실패: {str(e)}"
        st.session_state.trend_result = "트렌드 정보 없음"
        return f"TREND_ERROR: {str(e)}"    



def strategy_tool(query):
    """상품 전략을 생성하는 도구"""
    try:
        weather_info, trend_info = get_weather_and_trends(query)
        inventory_info = product_name_extract(query, weather_info, trend_info, 0.35)
        final_prompt = generate_strategic(query, weather_info, trend_info, inventory_info)
        response = llm.invoke(final_prompt)
        
        # 🔧 수정: 안전한 세션 상태 업데이트
        st.session_state.strategy_result = str(response.content) if response.content else "전략 생성 실패"
        st.session_state.inventory_info = inventory_info if inventory_info else {}
        st.session_state.weather_info = str(weather_info) if weather_info else "날씨 정보 없음"
        st.session_state.trend_info = str(trend_info) if trend_info else "트렌드 정보 없음"
        
        return "STRATEGY_SUCCESS"
        
    except Exception as e:
        print(f"strategy_tool 오류: {e}")
        # 🔧 추가: 오류 시에도 세션 상태 안전하게 설정
        st.session_state.strategy_result = f"전략 생성 실패: {str(e)}"
        st.session_state.inventory_info = {}
        return f"STRATEGY_ERROR: {str(e)}"

# 간단한 Tool 등록
tools = [
    Tool(
        name="weather",
        func=weather_tool,
        description="Use this for weather questions. Input: weather query"
    ),
    Tool(
        name="trend",
        func=trend_tool,
        description="Use this for trend questions. Input: trend query"
    ),    
    Tool(
        name="strategy", 
        func=strategy_tool,
        description="Use this for product strategy questions. Input: strategy query"
    )
]

# 간단한 Agent 초기화
try:
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,  # 로그 출력 최소화
        max_iterations=1,  # 최대 1번만 실행
        early_stopping_method="force",  # 강제 종료
        handle_parsing_errors=True
    )
except Exception as e:
    st.error(f"Agent 초기화 실패: {e}")
    agent = None

# 질문 유형 자동 판별 함수
def classify_question(question):
    """질문 유형을 자동으로 판별"""
    question_lower = question.lower()
    
    weather_keywords = ["날씨", "weather", "기온", "비", "눈", "맑", "흐림", "온도"]
    trend_keywords = ["신상", "트렌드", "최신"]
    strategy_keywords = ["상품", "판매", "전략", "추천", "기획", "마케팅", "홈쇼핑"]
    
    if any(keyword in question_lower for keyword in weather_keywords):
        return "weather"
    elif any(keyword in question_lower for keyword in trend_keywords):   
        return "trend"
    else:
        return "strategy"  # 기본값

# tabs[2] 구현
with tabs[2]:
    st.subheader("🤖 AI 상품판매 전략 추천 (Agent 기반)")
    
    # 유사도 조정 UI
    col_input, col_similarity = st.columns([3, 1])
    
    with col_input:
        user_question = st.text_input(
            "질문을 입력하세요", 
            placeholder="예: 내일 날씨 알려줘 / 다음주에 어떤 상품을 판매하면 좋을까?"
        )
    
    with col_similarity:
        similarity_threshold = st.slider(
            "🎯 유사도 임계값", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.35,
            step=0.05
        )

    # Agent 실행 버튼
    if st.button("🚀 AI Agent 실행", type="primary") and user_question:
        with st.spinner("AI Agent가 작업 중입니다..."):
            
            # 🔧 개선: 질문 유형 미리 판별하여 직접 도구 호출
            question_type = classify_question(user_question)
            
            try:
                if question_type == "weather":
                    # 날씨 도구 직접 호출
                    result = weather_tool(user_question)
                elif question_type == "trend":
                    # 날씨 도구 직접 호출
                    result = trend_tool(user_question)             
                    
                else:  # strategy
                    # 전략 도구 직접 호출
                    result = strategy_tool(user_question)
                
                print(f"Direct Tool 결과: {result}")
                
                # 🔧 결과에 따른 UI 표시

                if result == "WEATHER_SUCCESS":
                    st.subheader("🌦️ 상세 날씨 정보")                    
                    # 🔧 수정: 안전한 세션 상태 확인
                    weather_result = st.session_state.get('weather_result', '날씨 정보 없음')
                    if weather_result and weather_result != '날씨 정보 없음':
                        st.markdown("**날씨 정보:**")
                        st.text(weather_result)  # markdown 대신 text 사용 (더 안전)
                        
                elif result == "TREND_SUCCESS":
                    st.subheader("🌦️ 트렌드 정보")
                    trend_result = st.session_state.get('trend_result', '트렌드 정보 없음')
                    if trend_result and trend_result != '트렌드 정보 없음':
                        st.markdown("**관련 트렌드:**")
                        # 트렌드 정보 길이 제한
                        if len(trend_result) > 300:
                            trend_result = trend_result[:300] + "..."
                        st.text(trend_result)

                elif result == "STRATEGY_SUCCESS":
                    col1, col2 = st.columns([1, 2])                    
                    with col1:
                        st.subheader("📊 추천 상품 및 재고")
                        st.success("✅ 전략 생성 완료")
                        st.info(f"🎯 설정된 유사도 임계값: {similarity_threshold:.2f}")
                        
                        # 🔧 수정: 안전한 세션 상태 확인
                        inventory_info = st.session_state.get('inventory_info', {})
                        if inventory_info:
                            total_matches = 0
                            total_products = len(inventory_info)
                            
                            for gpt_name, matches in inventory_info.items():
                                st.write(f"**{gpt_name}**")
                                
                                if matches and len(matches) > 0:
                                    try:
                                        if matches[0][1] > 0:
                                            for name, qty in matches:
                                                if qty > 0:
                                                    st.write(f"   ✅ {name} (재고: {qty}개)")
                                                    total_matches += 1
                                                else:
                                                    st.write(f"   ❌ {name} (품절)")
                                        else:
                                            st.write(f"   ❌ 매칭된 상품 없음")
                                    except (IndexError, TypeError):
                                        st.write(f"   ❌ 매칭된 상품 없음")
                                else:
                                    st.write(f"   ❌ 매칭된 상품 없음")
                            
                            # 매칭 성공률
                            success_rate = (total_matches / total_products * 100) if total_products > 0 else 0
                            st.markdown("---")
                            st.metric("📈 매칭 성공률", f"{success_rate:.1f}%")
                    
                    with col2:
                        st.subheader("🎯 AI 추천 전략")
                        
                        # 🔧 수정: 안전한 세션 상태 확인
                        strategy_result = st.session_state.get('strategy_result', '전략 정보 없음')
                        if strategy_result and strategy_result != '전략 정보 없음':
                            st.text(strategy_result)  # write 대신 text 사용

                elif "ERROR" in result:
                    # 오류 처리
                    st.error(f"❌ 처리 중 오류: {result}")
                    
                    # 대안 처리
                    st.info("🔄 대안 방법으로 처리 중...")
                    if question_type == "weather":
                        weather_info, trend_info = get_weather_and_trends(user_question)
                        st.write("**날씨 정보:**")
                        st.write(weather_info)

                else:
                    # 일반 응답
                    st.subheader("💬 AI 응답")
                    st.write(result)

            except Exception as e:
                st.error(f"❌ 실행 중 오류: {str(e)}")
                
                # 최종 대안: Agent 없이 직접 처리
                st.info("🔄 Agent 없이 직접 처리 중...")
                try:
                    if question_type == "weather":
                        weather_info, trend_info = get_weather_and_trends(user_question)
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.subheader("🌤️ 날씨 분석")
                            st.info("직접 처리 모드")
                        with col2:
                            st.subheader("🌦️ 날씨 정보")
                            st.write(weather_info)
                            
                    elif question_type == "strategy":
                        weather_info, trend_info = get_weather_and_trends(user_question)
                        inventory_info = product_name_extract(user_question, weather_info, trend_info, similarity_threshold)
                        final_prompt = generate_strategic(user_question, weather_info, trend_info, inventory_info)
                        response = llm.invoke(final_prompt)
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.subheader("📊 상품 분석")
                            st.info("직접 처리 모드")
                        with col2:
                            st.subheader("🎯 전략")
                            st.write(response.content)
                            
                except Exception as final_error:
                    st.error(f"최종 처리도 실패: {str(final_error)}")

                
# 사이드바 - 시스템 상태
with st.sidebar:
    st.header("🔧 시스템 상태")
    
    # API 키 상태
    if OPENAI_API_KEY:
        st.success("✅ OpenAI API")
    else:
        st.error("❌ OpenAI API")
    
    # TAVILY_API
    if TAVILY_API_KEY:
        st.success("✅ Tavily API")
    else:
        st.error("❌ Tavily API")

    # Azure_ComputerVision
    if VISION_API_KEY:
        st.success("✅ Azure Computer Vision API")
    else:
        st.error("❌ Azure Computer Vision API")
    
    # DB 연결 테스트
    if st.button("DB 연결 테스트"):
        print("DB 연결 테스트")
        status = connection_db()
        print(f"status : {status}")
        if status["success"]:
            st.success(status["message"])
        else:
            st.error(status["error"])

    # 🆕 유사도 가이드 추가
    st.markdown("---")
    st.header("🎯 유사도 가이드")
    
    with st.expander("📖 유사도 임계값 설명"):
        st.write("""
        **유사도 임계값이란?**
        - AI가 상품을 매칭할 때 사용하는 기준
        - 0.1 ~ 0.9 사이의 값으로 설정 가능
        
        **추천 설정:**
        - **0.1~0.3**: 느슨한 매칭 (더 많은 상품 표시)
        - **0.3~0.6**: 균형잡힌 매칭 (권장)
        - **0.6~0.9**: 엄격한 매칭 (정확한 상품만)
        
        **사용 팁:**
        - 매칭되는 상품이 너무 적으면 → 값을 낮추세요
        - 관련 없는 상품이 많으면 → 값을 높이세요
        """)
