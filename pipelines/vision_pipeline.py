import time
import hashlib
import json
import pymysql
import os
from datetime import datetime
from io import BytesIO
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pipelines.connect_db_engine import create_db_engine
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

load_dotenv()

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_API_KEY = os.getenv("VISION_API_KEY")

vision_client = ComputerVisionClient(
    endpoint=VISION_ENDPOINT,
    credentials=CognitiveServicesCredentials(VISION_API_KEY)
)

class ProductImageClassifier:

    # 클래스 초기화 - OpenAI와 DB 연결 설정
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")

        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=self.openai_api_key,
            temperature=0.3
        )

        self.db_engine = create_db_engine()

    # 이미지에서 텍스트 추출 (OCR)
    def c_extract_text_from_image(self, image_bytes: bytes) -> str:
        try:

            # Azure OCR API 호출
            ocr_result = vision_client.read_in_stream(BytesIO(image_bytes), raw=True)
            operation_location = ocr_result.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

            # OCR 작업 완료까지 대기
            while True:
                result = vision_client.get_read_result(operation_id)
                if result.status not in ['notStarted', 'running']:
                    break
                time.sleep(0.5)

            # 텍스트 추출
            text = ""
            if result.status == "succeeded":
                for page in result.analyze_result.read_results:
                    for line in page.lines:
                        text += line.text + " "
            return text.strip()
        except Exception as e:
            print(f"❌ OCR 실패: {e}")
            return ""


    # Azure Computer Vision API를 통해 이미지의 태그 및 카테고리 추출
    def c_analyze_visual_features(self, image_bytes: bytes) -> dict:
        """Azure Computer Vision API를 통해 이미지의 태그 및 카테고리 추출"""
        from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes

        try:
            # Azure Vision API로 이미지 분석
            analysis = vision_client.analyze_image_in_stream(
                BytesIO(image_bytes),
                visual_features=[VisualFeatureTypes.categories, VisualFeatureTypes.tags, VisualFeatureTypes.description]
            )

            # 결과 추출
            tags = [tag.name for tag in analysis.tags[:10]] if analysis.tags else []
            categories = [cat.name for cat in analysis.categories] if analysis.categories else []
            descs = analysis.description.captions[0].text if analysis.description and analysis.description.captions else ""

            return {
                "vision_tags": tags,
                "vision_categories": categories,
                "vision_description": descs
            }
        except Exception as e:
            print(f"Vision API 분석 실패: {e}")
            return {"vision_tags": [], "vision_categories": [], "vision_description": ""}

    # GPT를 사용하여 카테고리와 태그 생성
    def c_generate_category_and_tags(self, filename: str, extracted_text: str, vision_info: dict) -> tuple:   
        try:
            prompt = f"""
                    전자상거래용 상품 이미지 분석 결과입니다. 아래 정보들을 종합하여 상품의 카테고리와 태그를 추론하세요:

                    파일명: "{filename}"
                    OCR 텍스트: "{extracted_text}"
                    Vision 태그: {', '.join(vision_info['vision_tags'])}
                    Vision 카테고리: {', '.join(vision_info['vision_categories'])}
                    이미지 설명: "{vision_info['vision_description']}"

                    요구사항:
                    1. 카테고리: 전자제품, 생활용품, 의류, 식품, 화장품, 스포츠용품 중 하나
                    2. 카테고리2: '카테고리'의 하위 카테고리
                    3. 카테고리3: '카테고리2'의 하위 카테고리
                    3. 태그: 최대 5개, 마케팅/검색용 키워드

                    응답 형식(JSON만 응답):
                    {{
                    "category": "카테고리명",
                    "category2": "카테고리명",
                    "category3": "카테고리명",
                    "tags": ["태그1", "태그2", "태그3", "태그4", "태그5"]
                    }}
                    """
            # GPT 호출 및 응답 처리
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # JSON 파싱
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
                result = json.loads(json_text)
                category1 = result.get("category", "의류")
                category2 = result.get("category2", "여성패션")
                category3 = result.get("category3", "스커트")
                tags = result.get("tags", ["상품", "홈쇼핑"])

                # 태그 개수 보정
                if len(tags) < 5:
                    tags.extend(["기본태그"] * (5 - len(tags)))
                # return category, tags[:5]
                return category1, category2, category3, tags[:5]
        except Exception as e:
            print(f"GPT 분류 실패: {e}")
        #return "생활용품", ["상품", "홈쇼핑", "추천", "인기", "기본"]
        return "의류","여성패션","스커트", ["상품", "홈쇼핑", "추천", "인기", "기본"]


    # 메인 이미지 분류 메소드
    def c_classify_product_image(self, uploaded_file, filename: str = "") -> dict:

        print ("start-ProductImageClassifier.classify_product_image")
        start_time = time.time()
        try:

            # 1. 이미지 데이터 읽기
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            if not image_bytes:
                raise ValueError("이미지 데이터가 없습니다")

            # 2. 이미지 ID 생성 (MD5 해시). 이미지 바이트 데이터를 고유한 ID로 변환
            image_id = hashlib.md5(image_bytes).hexdigest()
            
            # 3. 이미지를 텍스트로 추출
            extracted_text = self.c_extract_text_from_image(image_bytes)
            print(f"extracted_text : {extracted_text}")
        
            # 4. Azure Computer Vision API를 통해 이미지의 태그 및 카테고리 추출
            vision_info = self.c_analyze_visual_features(image_bytes)
            print(f"vision_info : {vision_info}")

            # 기존 : 
            # 5. GPT를 사용하여 카테고리와 태그 생성
            # category, tags = self.c_generate_category_and_tags(filename, extracted_text, vision_info)
            # print(f"category, tags : {category, tags}")
            # 수정_ 20250619: 
            category, category2, category3, tags = self.c_generate_category_and_tags(filename, extracted_text, vision_info)
            print(f"category, category2, category3, tags : {category, category2, category3, tags}")

            # 6. 결과 생성
            confidence = 0.92  # 향후 개선 가능
            processing_time = time.time() - start_time

            return {
                "image_id": image_id,
                "category": category,
                "category2": category2,
                "category3": category3,
                "tags": tags,
                "extracted_text": extracted_text,
                "confidence_score": round(confidence, 2),
                "processing_time": round(processing_time, 2),
                "db_ready": bool(self.db_engine), # DB 연결 상태 확인
                "filename": filename,
                "vision_tags": vision_info['vision_tags'],
                "vision_description": vision_info['vision_description'],
                "vision_categories": vision_info['vision_categories'],
                "display_summary": f"""
                    **📌 분류 요약:**
                    - **카테고리:** {category}
                    - **카테고리2:** {category2}
                    - **카테고리3:** {category3}
                    - **신뢰도:** {round(confidence, 2)}
                    - **GPT 태그:** {', '.join(tags)}
                    - **OCR 텍스트:** {extracted_text if extracted_text else '없음'}
                    - **🔎 Vision 태그:** {', '.join(vision_info['vision_tags'])}
                    - **🖼 이미지 설명:** {vision_info['vision_description']}
                    - **📂 Vision 카테고리:** {', '.join(vision_info['vision_categories'])}
                """
            }
        except Exception as e:
            return {
                "image_id": "error",
                "category": "분류실패",
                "category2": "분류실패2",
                "category3": "분류실패3",
                "tags": ["오류"],
                "extracted_text": "",
                "confidence_score": 0.0,
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e),
                "db_ready": False
            }

    # GPT 클러스터링
    def c_cluster_categories_from_tags(self, tag_samples: list) -> dict:
        """GPT를 사용하여 태그 기반 자동 카테고리 그룹화 수행"""
        prompt = f"""
                다음은 상품 태그 리스트입니다. 의미적으로 유사한 항목들을 묶어서 적절한 카테고리명을 부여해 주세요.

                태그 샘플:
                {json.dumps(tag_samples, ensure_ascii=False, indent=2)}

                응답 형식(JSON만 응답):
                {{
                "카테고리1": [["태그1", "태그2", ...]],
                "카테고리2": [[...]],
                ...
                }}
                """
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                result_json = response_text[json_start:json_end]
                result = json.loads(result_json)
                return result
        except Exception as e:
                    print(f"카테고리 클러스터링 실패: {e}")
                    return {}
   
   
        
     # DB에 분류 결과 저장
    def c_save_to_db(self, image_id: str, category: str, category2: str, category3: str, confidence: float, extracted_text: str, tags: list) -> dict:
        print(f"💾 DB 저장 시작: {image_id}, {category}, {category2}, {category3}, {confidence}, {len(tags)}개 태그")       

        if not self.db_engine:
            return {"success": False, "error": "DB 연결 없음"}
        try:
            # 1. 태그를 JSON 문자열로 변환
            tags_json = json.dumps(tags, ensure_ascii=False)
            print(f"📝 태그 JSON 변환: {tags_json}")

            # 2. SQL 쿼리 준비 (product_classifications 테이블 스키마에 맞춤)
            insert_sql = text("""
                REPLACE INTO product_classifications 
                (id, category, category2, category3, confidence_score, extracted_text, tags)
                VALUES (:id, :category, :category2, :category3, :confidence, :text, :tags)
            """)

            # 3. DB에 데이터 저장
            with self.db_engine.begin() as conn:
                conn.execute(insert_sql, {
                    "id": image_id,                                         # varchar(32) - 이미지 MD5 해시
                    "category": category,                                   # varchar(100) - 카테고리명
                    "category2": category2,                                   # varchar(100) - 카테고리명
                    "category3": category3,                                   # varchar(100) - 카테고리명
                    "confidence": float(confidence),                        # decimal(3,2) - 신뢰도 점수
                    "text": extracted_text[:500] if extracted_text else "", # text - OCR 텍스트 (500자 제한)
                    "tags": tags_json                                       # json - 태그 배열을 JSON으로 저장
                })
            print("✅ DB 저장 성공!")
            return {"success": True, "message": "DB 저장 완료"}
        except Exception as e:
            print(f"❌ DB 저장 실패: {e}")
            return {"success": False, "error": str(e)}
        
############# END : class ProductImageClassifier: #########################

# ============================================
# 📋 DB 저장 흐름 설명:
# ============================================
# 1. 사용자가 "🔍 이미지 분류 실행" 클릭
# 2. classify_product_image() 호출 → 분류 결과 생성
# 3. 결과가 st.session_state.last_result에 저장
# 4. 사용자가 "💾 DB에 저장" 클릭  
# 5. save_classification_to_db() 호출
# 6. ProductImageClassifier.save_to_db() 메소드 실행
# 7. product_classifications 테이블에 INSERT/REPLACE
# 8. 성공/실패 메시지 반환



from pipelines.vision_pipeline import ProductImageClassifier

# 전역 - 이미지 분류 실행
def classify_product_image(uploaded_file, filename: str = "") -> dict:
    classifier = ProductImageClassifier()
    print("start - classifier.c_classify_product_image ")
    return classifier.c_classify_product_image(uploaded_file, filename)

# 전역 - DB에 저장
def save_classification_to_db(image_id: str, category: str, category2: str, category3: str, confidence: float, text: str, tags: list) -> dict:
    classifier = ProductImageClassifier()
    print(f"🔄 c_save_to_db 호출: {image_id[:8]}...")
    return classifier.c_save_to_db(image_id, category, category2, category3, confidence, text, tags)

# 전역 - 최근 분류 조회(포함미정)
def get_recent_classifications(limit: int = 5) -> list:
    classifier = ProductImageClassifier()
    if not classifier.db_engine:
        return []
    try:
        query = text("""
            SELECT id, category, category2, category3, confidence_score, created_at 
            FROM product_classifications 
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        with classifier.db_engine.connect() as conn:
            results = conn.execute(query, {"limit": limit}).fetchall()
        return [
            {
                "id": row[0][:8] + "...",
                "category": row[1],
                "category2": row[2],
                "category3": row[3],
                "confidence": float(row[4]) if row[4] else 0.0,
                "created_at": str(row[5])[:21]
            }
            for row in results
        ]
    except Exception as e:
        print(f"분류 히스토리 조회 실패: {e}")
        return []
    
