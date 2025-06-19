# Google Trends or Naver Trends 연동
# pip install pytrends - X (너무 잘 안됨됨)
# pip install import feedparser

import feedparser

# 트렌드정보 수집(사용안함 - Tavily API를 통해 정보 수집)
def get_google_trending_searches():
    rss_url = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=KR"
    feed = feedparser.parse(rss_url)

    trending_list = []
    for entry in feed.entries[:5]:  # 상위 5개만 사용
        trending_list.append(entry.title)

    return "\n".join(trending_list)

