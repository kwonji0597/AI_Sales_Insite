# OpenWeather API 연동

import requests
# import config

# 날씨정보 수집(사용안함 - Tavily API를 통해 날씨 정보 수집)
def get_weather_forecast(city="Seoul", target_date=None):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={config.OPENWEATHER_API_KEY}&units=metric&lang=kr"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200:
        return f"날씨 정보를 가져오는 데 실패했습니다: {data.get('message', 'Unknown error')}"

    forecast_texts = []
    for item in data['list']:
        dt_txt = item['dt_txt']
        date_part = dt_txt.split(' ')[0]  # "2025-06-14"
        
        # target_date가 None이면 전체 출력, 아니면 해당 날짜만 출력
        if target_date is None or date_part == target_date:
            temp = item['main']['temp']
            weather_desc = item['weather'][0]['description']
            forecast_texts.append(f"{dt_txt}: {temp}°C, {weather_desc}")

    if not forecast_texts:
        return f"{target_date}에 대한 날씨 정보가 없습니다."

    return "\n".join(forecast_texts)