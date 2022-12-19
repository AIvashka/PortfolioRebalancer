FROM python:3.8

WORKDIR /PortfolioRebalancer

COPY requirements.txt /PortfolioRebalancer

RUN pip install -r requirements.txt
RUN ls

COPY . /PortfolioRebalancer

CMD ["python", "PortfolioRebalancingTool.py"]