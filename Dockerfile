FROM python:3.8

WORKDIR /PortfolioRebalancer

COPY requirements.txt /PortfolioRebalancer

RUN pip install -r requirements.txt
RUN ls

COPY . /PortfolioRebalancer

EXPOSE 8501

# CMD ["python", "PortfolioRebalancingTool.py"]

ENTRYPOINT ["streamlit", "run", "PortfolioRebalancingTool.py", "--server.port=8501", "--server.address=0.0.0.0"]
