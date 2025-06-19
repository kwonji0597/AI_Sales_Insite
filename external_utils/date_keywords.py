# external_utils/date_utils.py

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re


# external_utils/date_utils.py

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re

def get_date_keywords(user_question: str):
    """
    사용자 질문에서 날짜 관련 키워드를 추출하여
    날씨 및 트렌드 검색에 사용할 날짜 기반 쿼리를 생성합니다.
    """
    today = datetime.now()
    daydate = today.strftime('%Y-%m-%d')

    # "9월", "10월" 형태만 추출하되 '9월 1일' 등은 제외
    month_only_match = re.findall(r'\b(1[0-2]|[1-9])월(?!\s*\d+일)', user_question)

    if '오늘' in user_question:
        daydate = today.strftime('%Y-%m-%d')
    elif '내일' in user_question:
        daydate = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    elif '모레' in user_question:
        daydate = (today + timedelta(days=2)).strftime('%Y-%m-%d')
    elif '이번주' in user_question:
        daydate = today.strftime('%Y-%m-%d')
    elif '이번달' in user_question:
        daydate = today.replace(day=1).strftime('%Y-%m-%d')
    elif '다음주' in user_question:
        daydate = (today + timedelta(weeks=1)).strftime('%Y-%m-%d')
    elif '다음달' in user_question:
        next_month = today.replace(day=1) + relativedelta(months=1)
        daydate = next_month.strftime('%Y-%m-%d')
    elif month_only_match:
        target_month = int(month_only_match[0])
        target_year = today.year if target_month >= today.month else today.year + 1
        daydate = f"{target_year}-{target_month:02d}-01"
    else:
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', user_question)
        if date_match:
            daydate = date_match.group(0)

    print("keyword date:", daydate)
    weather_query = f"({daydate}) 한국 날씨 예보"
    trend_query = f"({daydate}) 한국 트렌드 검색어"

    return weather_query, trend_query
