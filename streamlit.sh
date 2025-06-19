
# Core 패키지
pip install streamlit
pip install python-dotenv
pip install pandas
pip install plotly

# LangChain 관련
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install langchain-tavily


# vision_pipeline.py 필요 패키지
# Azure Computer Vision API
pip install azure-cognitiveservices-vision-computervision>=0.9.0
pip install msrest>=0.7.1
pip install azure-core>=1.28.0
pip install azure-common>=1.1.28

# 데이터베이스 관련
pip install sqlalchemy
pip install pymysql
pip install cryptography

# 머신러닝/임베딩
pip install scikit-learn
pip install numpy

# 데이터 시각화 (query_pipeline.py용)
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0  # 선택사항 (더 예쁜 차트용)

# connect_db_engine.py 필요 패키지
pip install sqlalchemy>=2.0.0
pip install pymysql>=1.1.0  
pip install cryptography>=41.0.0  # PlanetScale SSL 연결용


python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0