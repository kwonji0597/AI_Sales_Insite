# connect_db_engine.py - PlanetScale 연결

import os
import pymysql
from sqlalchemy import create_engine, text
import sqlalchemy
from sqlalchemy import text
from dotenv import load_dotenv
load_dotenv()

MYSQL_USERNAME = os.getenv("MYSQL_USERNAME")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

print(f"ENV - USERNAME: {MYSQL_USERNAME}, HOST: {MYSQL_HOST}, PORT: {MYSQL_PORT}, DB: {MYSQL_DATABASE}")

if None in [MYSQL_USERNAME, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE]:
    raise ValueError("❗ DB 접속 정보가 누락되었습니다. .env 파일을 확인하세요.")

def create_db_engine(): 
    try:
        # 테스트에서 성공한 연결 문자열 사용
        connection_string = (
            f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
            "?charset=utf8mb4&ssl=true"
        )
        
        # SSL 설정을 connect_args로 전달 (방법 2에서 사용한 설정)
        connect_args = {
            "ssl": {
                "ssl_disabled": False
            }
        }
        
        engine = sqlalchemy.create_engine(
            connection_string,
            connect_args=connect_args,
            pool_pre_ping=True,  # 연결 상태 확인
            pool_recycle=300,    # 5분마다 연결 재생성
            echo=False           # SQL 로그 출력 비활성화
        )
        
        # 연결 테스트
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ DB 연결 성공!")
            
            # 테이블 확인
            tables_result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in tables_result.fetchall()]
            print(f"📋 사용 가능한 테이블: {tables}")
            
        return engine
        
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return None
   

# 전역 엔진 생성
engine = create_db_engine()


# DB연결 테스트
def connection_db():
    """
    Streamlit app.py에서 사용할 DB 연결 테스트 함수
    Returns: {"success": bool, "message": str, "error": str}
    """
    
    # 환경변수 확인
    required_vars = ["MYSQL_HOST", "MYSQL_PORT", "MYSQL_USERNAME", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        return {
            "success": False,
            "error": f"환경변수 누락: {', '.join(missing_vars)}"
        }
    
    # 연결 정보
    host = os.getenv("MYSQL_HOST")
    port = int(os.getenv("MYSQL_PORT"))
    user = os.getenv("MYSQL_USERNAME")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    
    try:
        # SQLAlchemy 엔진 생성 (PlanetScale용)
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(
            url,
            connect_args={"ssl": {"ca": None}},  # SSL 강제 활성화
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # 연결 테스트
        with engine.connect() as conn:
            # 기본 연결 확인
            result = conn.execute(text("SELECT 1 as test")).fetchone()
            
            # 테이블 목록 조회
            tables_result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in tables_result.fetchall()]
            
            # inventory 테이블 특별 확인
            inventory_info = ""
            if 'inventory' in tables:
                count_result = conn.execute(text("SELECT COUNT(*) FROM inventory"))
                count = count_result.fetchone()[0]
                inventory_info = f", inventory 테이블: {count}개 상품"
            
            # product_classifications 테이블 확인
            classification_info = ""
            if 'product_classifications' in tables:
                count_result = conn.execute(text("SELECT COUNT(*) FROM product_classifications"))
                count = count_result.fetchone()[0]
                classification_info = f", 분류 기록: {count}개"
        
        return {            
            "success": True,
            "message": f"✅ DB 연결 성공! 테이블 {len(tables)}개{inventory_info}{classification_info}",
            "engine": engine,
            "tables": tables
        }
        
        
    except pymysql.err.OperationalError as e:
        if "SSL/TLS" in str(e):
            return {
                "success": False,
                "error": f"SSL 연결 오류: PlanetScale 데이터베이스가 비활성 상태이거나 SSL 설정 문제입니다. {str(e)}"
            }
        else:
            return {
                "success": False,
                "error": f"DB 연결 오류: {str(e)}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }