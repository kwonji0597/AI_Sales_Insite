# MySQL 기반 Sales Insights / Inventory (지금은 임시 비활성화 OK)

import pandas as pd
from sqlalchemy import text
import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy
from sqlalchemy import text
import pymysql
import os
from pipelines.connect_db_engine import create_db_engine
from dotenv import load_dotenv
load_dotenv()

# 전역 엔진 생성
engine = create_db_engine()

# 질문에 따른 날짜 분석(사용안함)
def run_natural_language_query(user_question):
    """
    사용자 질문을 기반으로 SQL 쿼리를 실행하고 결과를 반환합니다.
    """

    # 사용자 질문에 따른 SQL 쿼리 매핑
    if "지난 30일" in user_question or "최근 30일" in user_question:
        sql_query = """
        SELECT date, SUM(sales_amount) AS total_sales
        FROM sales_data
        WHERE date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY date
        ORDER BY date ASC
        """
    elif "지난 7일" in user_question or "최근 7일" in user_question:
        sql_query = """
        SELECT date, SUM(sales_amount) AS total_sales
        FROM sales_data
        WHERE date >= CURDATE() - INTERVAL 7 DAY
        GROUP BY date
        ORDER BY date ASC
        """
    elif "상품별 매출" in user_question:
        sql_query = """
        SELECT product_name, SUM(sales_amount) AS total_sales
        FROM sales_data
        GROUP BY product_name
        ORDER BY total_sales DESC
        """
    else:
        # 기본 쿼리 (예: 지난 30일 매출)
        sql_query = """
        SELECT date, SUM(sales_amount) AS total_sales
        FROM sales_data
        WHERE date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY date
        ORDER BY date ASC
        """

    try:
        # # SQL 실행 및 데이터프레임 생성
        # df = pd.read_sql_query(sql_query, engine)

        # SQL 실행 및 데이터프레임 생성
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql_query), conn)

        # 데이터 처리
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')
            df = df.dropna(subset=['date', 'total_sales'])
            df = df.sort_values(by='date')
        else:
            raise ValueError("Query returned no results.")

        # 데이터 시각화
        fig, ax = plt.subplots()
        df.plot(x='date', y='total_sales', ax=ax, title="Sales Data", xlabel="Date", ylabel="Total Sales")

        return df, fig

    except Exception as e:
        # 예외 처리: 실제 테이블이 없으므로 더미 데이터 반환
        print(f"Sales 테이블이 없음, 더미 데이터 사용: {e}")
    


def run_inventory_sql_query(product_name):
    """단일 상품 재고 조회"""
    if engine is None:
        print("⚠️ DB 연결 불가 - 더미 데이터 반환")
        return 100  # 더미 재고량    
    try:
        sql_query = "SELECT stock_quantity FROM inventory WHERE product_name = %s"
        
        with engine.connect() as conn:
            result = conn.execute(text(sql_query), {"product_name": product_name})
            row = result.fetchone()

        print(f"SQL Query: {sql_query}, Params: {product_name}")
        print(f"Query Result: {row}")

        # 결과가 있으면 재고량 반환
        if row:
            return int(row[0])
        else:
            return "No inventory info"
            
    except Exception as e:
        print(f"❌ 재고 조회 오류: {e}")
        return "No inventory info"


def run_natural_language_query(user_question):
    """
    사용자 질문을 기반으로 SQL 쿼리를 실행하고 결과를 반환합니다.
    """
    
    # 사용자 질문에 따른 SQL 쿼리 매핑
    if "지난 30일" in user_question or "최근 30일" in user_question:
        sql_query = """
        SELECT date, SUM(sales_amount) AS total_sales
        FROM sales_data
        WHERE date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY date
        ORDER BY date ASC
        """
    elif "지난 7일" in user_question or "최근 7일" in user_question:
        sql_query = """
        SELECT date, SUM(sales_amount) AS total_sales
        FROM sales_data
        WHERE date >= CURDATE() - INTERVAL 7 DAY
        GROUP BY date
        ORDER BY date ASC
        """
    elif "상품별 매출" in user_question:
        sql_query = """
        SELECT product_name, SUM(sales_amount) AS total_sales
        FROM sales_data
        GROUP BY product_name
        ORDER BY total_sales DESC
        """
    else:
        # 기본 쿼리 (예: 지난 30일 매출)
        sql_query = """
        SELECT date, SUM(sales_amount) AS total_sales
        FROM sales_data
        WHERE date >= CURDATE() - INTERVAL 30 DAY
        GROUP BY date
        ORDER BY date ASC
        """

    try:
        # SQL 실행 및 데이터프레임 생성
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql_query), conn)

        # 데이터 처리
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce')
            df = df.dropna(subset=['date', 'total_sales'])
            df = df.sort_values(by='date')
        else:
            raise ValueError("Query returned no results.")

        # 데이터 시각화
        fig, ax = plt.subplots()
        df.plot(x='date', y='total_sales', ax=ax, title="Sales Data", xlabel="Date", ylabel="Total Sales")

        return df, fig

    except Exception as e:
        # 예외 처리: 실제 테이블이 없으므로 더미 데이터 반환
        print(f"Sales 테이블이 없음{e}")


def run_inventory_sql_query(product_name):
    """단일 상품 재고 조회"""
       
    try:
        sql_query = "SELECT stock_quantity FROM inventory WHERE product_name = %s"
        
        with engine.connect() as conn:
            result = conn.execute(text(sql_query), {"product_name": product_name})
            row = result.fetchone()

        print(f"SQL Query: {sql_query}, Params: {product_name}")
        print(f"Query Result: {row}")

        # 결과가 있으면 재고량 반환
        if row:
            return int(row[0])
        else:
            return "No inventory info"
            
    except Exception as e:
        print(f"❌ 재고 조회 오류: {e}")
        return "No inventory info"

from pipelines.vision_pipeline import ProductImageClassifier
from functools import lru_cache



@lru_cache(maxsize=1)  # 1회만 캐시
def get_all_inventory_from_db():
    """전체 상품+재고 딕셔너리 반환 (실제 DB 데이터 사용)"""   
    try:
        sql_query = "SELECT product_name, stock_quantity FROM inventory"        
        # 실제 DB에서 데이터 조회
        with engine.connect() as conn:
            print("🔗 DB 연결 성공, 쿼리 실행 중...")
            df = pd.read_sql_query(text(sql_query), conn)

        print(f"📊 쿼리 결과 DataFrame: \n{df}")
       
        # 실제 DB 데이터를 딕셔너리로 변환 (공백 제거)
        inventory_dict = {
            str(row['product_name']).replace(" ", ""): int(row['stock_quantity'])
            for _, row in df.iterrows()
        }

        print(f"📦 실제 DB에서 가져온 재고 목록: {inventory_dict}")
        return inventory_dict

    except Exception as e:
        print(f"❌ 전체 재고 조회 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 상세: {str(e)}")

