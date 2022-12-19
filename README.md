# PortfolioRebalancer


## Task


Service that proposes portfolio rebalancing trades
1. Inputs:
    1. portfolio tickers, quantities, purchase prices (csv-file “ticker” (string), “quantity” (int), purchase_price (decimal(2)),
    2. portfolio tickers, target weights and constraints (csv-file “ticker” (string), “target_weight” (float))
    3. transaction fee (number, float, % of trade volume)
    4. maximum acceptable transaction fee (number, float, % of trade volume)
    5. min transaction fee (number, float, $)
    6. minimum weight (number, float, % of trade)
    7. maximum weight (number, float, % of trade)
    8. execution probability (number, float, as a percentage)
2. We need to calculate actual mark-to-market using actual prices (uploaded from free resources)
3. We need to calculate the rebalancing trades following the logic below:
    1. we want to be as close to target weights as possible
    2. quantities should be round
    3. we have to pay a fixed min transaction fee for a trade if % fee is lower than min transaction fee and we don’t want it to exceed a maximum acceptable transaction fee as a % of trade volume (so if, for example, the max acceptable is 1%, and we have to pay $500/$25000, it would be too much)
    4. the sum of resulting weights should be as close to 100% as possible, but not higher than 100% (we can left some allocation in cash)
    5. we would like to determine reasonable limit prices so the orders should be fulfilled in the next 2 trading days with execution probability set as an input. But we want to make sure the resulting trade price would be as good as possible.
    6. all resulting weights have to be in the interval between the minimum weight and maximum weight (e.g. 5% and 20%)
4. Output:
    1. portfolio ticker, quantity, limit price, trade direction (csv-file “ticker” (string), “quantity” (int), limit price (decimal(2)), trade direction (string, ‘buy’ or ‘sell’))


## Further Enhancements

    Logic side:
      1. If constraint is met we need to implement a logic to handle it. For now if we met a fee or a target weight constraint we just skip the trade.
   
    Technical side:
      1. Unit tests
      2. CI/CD 
      3. Add authentication to streamlit app
      4. Set up a database to store user data
      

RUN:

    # docker rm -f streamlit_app
    docker build -f Dockerfile -t streamlit_app_image .
    docker run -d --name streamlit_app -p 8501:8501 streamlit_app_image

Currently deployed on EC2 AWS Instance

http://35.79.131.185:8501/
