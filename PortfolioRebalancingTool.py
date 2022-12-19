from __future__ import annotations

import csv
import dataclasses
import datetime
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from scipy import special
from enum import Enum
from typing import List, Dict

import pandas as pd
import yfinance as yf
import streamlit as st
from streamlit import cli as stcli


@dataclass
class Fee:
    fee: float
    min_fee: float
    max_fee: float


@dataclass
class Weight:
    min_weight: float
    max_weight: float


@dataclass
class Trade:
    ticker: str
    quantity: int
    limit_price: float
    side: str

    def __str__(self):
        return f'{self.side} {self.quantity} {self.ticker} at {self.limit_price}'

    def asdict(self):
        return dataclasses.asdict(self)


class Side(Enum):
    buy: str = 'buy'
    sell: str = 'sell'

    def __str__(self):
        return self.value


@dataclass
class Stock:
    ticker: str
    quantity: int
    purchase_price: float
    target_weight: float
    current_market_price: float
    current_weight: float = field(init=False)
    current_portfolio_value: float = field(init=False)
    execution_probability: float = field(init=False)

    def set_current_market_price(self, current_market_price):
        self.current_market_price = current_market_price

    def set_execution_probability(self, execution_probability):
        self.execution_probability = execution_probability

    def set_current_weight(self, current_portfolio_value):
        self.current_weight = (self.quantity * self.current_market_price / current_portfolio_value) * 100
        self.current_portfolio_value = current_portfolio_value

    def get_current_market_value(self):
        return self.quantity * self.current_market_price

    def get_initial_market_value(self):
        return self.quantity * self.purchase_price

    def get_pnl(self):
        return self.get_current_market_value() - self.get_initial_market_value()

    def get_target_quantity(self) -> int:
        if self.current_weight > self.target_weight:
            # sell
            return math.floor((self.current_weight - self.target_weight) * 0.01 *
                              self.get_current_market_value() / self.current_market_price)
        else:
            # buy
            return math.floor((self.target_weight - self.current_weight) * 0.01 *
                              self.get_current_market_value() / self.current_market_price)

    def get_limit_price(self, execution_probability: float = 0.5) -> float:
        vol = get_annualized_vol(self.ticker)
        if self.get_trade_direction() == "buy":

            return round(self.current_market_price * (1 + vol[self.ticker]) ** (
                - special.erfinv(1 - execution_probability) * math.sqrt(math.pi) * math.sqrt(2 / 365)), 2)
        else:
            return round(self.current_market_price * (1 - vol[self.ticker]) ** (
                - special.erfinv(1 - execution_probability) * math.sqrt(math.pi) * math.sqrt(2 / 365)), 2)

    def get_trade_direction(self) -> str:
        if self.current_weight > self.target_weight:
            return "sell"
        else:
            return "buy"

    def get_rebalancing_trade(self):
        quantity = self.get_target_quantity()
        if quantity != 0:
            return Trade(
                ticker=self.ticker,
                quantity=quantity,
                limit_price=self.get_limit_price(self.execution_probability),
                side=self.get_trade_direction(),
            )


@dataclass
class Portfolio:
    stocks: List[Stock]
    execution_probability: float
    initial_portfolio_value: float = field(init=False)
    actual_portfolio_value: float = field(init=False)

    def __post_init__(self):
        self.initial_portfolio_value = sum([stock.quantity * stock.purchase_price for stock in self.stocks])
        self.actual_portfolio_value = sum([stock.quantity * stock.current_market_price for stock in self.stocks])
        [stock.set_current_weight(self.actual_portfolio_value) for stock in self.stocks]
        [stock.set_execution_probability(self.execution_probability) for stock in self.stocks]

    def get_actual_portfolio_value(self) -> float:
        return self.actual_portfolio_value

    def get_initial_portfolio_value(self) -> float:
        return self.initial_portfolio_value


@dataclass
class PortfolioRebalancer:
    portfolio: Portfolio
    fee: Fee
    weight: Weight
    rebalancing_trades: List[Trade] = None

    def get_rebalancing_trades(self) -> List[Trade]:
        if not self.rebalancing_trades:
            rebalancing_trades = []
            for stock in self.portfolio.stocks:
                trade = stock.get_rebalancing_trade()
                if trade:
                    if self.check_fee_constraint(trade) and self.check_weight_constraint(stock):
                        rebalancing_trades.append(trade)
                    else:
                        continue
            self.rebalancing_trades = rebalancing_trades
            return rebalancing_trades
        else:
            return self.rebalancing_trades

    def check_fee_constraint(self, trade: Trade) -> bool:
        if self.get_fee_pct(trade) < self.fee.max_fee:
            return True
        else:
            txt = f'Fee constraint is not met for trade {trade}'
            logging.error(txt)
            st.error(txt)
            return False

    def get_fee_pct(self, trade: Trade) -> float:
        return max(self.fee.min_fee, trade.quantity * trade.limit_price * self.fee.fee) / (trade.quantity *
                                                                                           trade.limit_price)

    def check_weight_constraint(self, stock: Stock) -> bool:
        if stock.target_weight == 0:
            return True
        elif (self.weight.min_weight * 100) <= stock.target_weight <= (self.weight.max_weight * 100):
            return True
        else:
            txt = f'Weight constraint is not met for trade {stock}'
            logging.error(txt)
            st.error(txt)
            return False

    def get_rebalancing_trades_df(self) -> pd.DataFrame:
        trades = self.get_rebalancing_trades()
        return pd.json_normalize(trade.asdict() for trade in trades)

    def generate_rebalancing_trades_csv(self):
        with open('rebalancing_trades.csv', 'w', newline='') as csvfile:
            fieldnames = ['ticker', 'quantity', 'limit_price', 'side']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for trade in self.get_rebalancing_trades():
                writer.writerow(trade.asdict())


def get_annualized_vol(ticker: str | List[str]) -> float | Dict[str, float]:
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)
    if isinstance(ticker, str):
        vol = yf.download(ticker, start=start, end=end, period='1d', progress=False)[
                  'Close'].pct_change().std() * math.sqrt(252)
        return {ticker: vol}
    elif isinstance(ticker, list):
        return (yf.download(tickers=ticker, period='1d', progress=False)['Close'].pct_change().std() * math.sqrt(
            252)).to_dict()


def get_ticker_close_price(ticker: str | List[str]) -> float | Dict[str, float]:
    if isinstance(ticker, str):
        price = yf.download(ticker, period='1d', progress=False)['Close'][0]
        return {ticker: price}
    elif isinstance(ticker, list):
        return yf.download(tickers=ticker, period='1d', progress=False)['Close'].iloc[0].to_dict()


def get_portfolio_df(portfolio_file) -> pd.DataFrame:
    df_portfolio = pd.read_csv(portfolio_file)
    df_portfolio = df_portfolio.astype({'quantity': 'int64', 'purchase_price': 'float64'})
    df_portfolio['current_market_price'] = get_ticker_close_price(df_portfolio['ticker'].to_list()).values()
    df_portfolio['initial_market_value'] = df_portfolio['quantity'] * df_portfolio['purchase_price']
    df_portfolio['mark_to_market'] = df_portfolio['quantity'] * df_portfolio['current_market_price']
    df_portfolio['pnl'] = df_portfolio['quantity'] * (
        df_portfolio['current_market_price'] - df_portfolio['purchase_price'])
    return df_portfolio


def get_stocks(portfolio: pd.DataFrame, target_weights_file) -> List[Stock]:
    df_target_weights = pd.read_csv(target_weights_file)
    stocks = []
    for _, row in portfolio.iterrows():
        stocks.append(Stock(ticker=row['ticker'],
                            quantity=row['quantity'],
                            purchase_price=row['purchase_price'],
                            target_weight=
                            df_target_weights[df_target_weights['ticker'] == row['ticker']]['target_weight'].values[0],
                            current_market_price=row['current_market_price']))

    if sum([stock.target_weight for stock in stocks]) > 100:
        raise ValueError('Sum of target weights should be less than 1')
    if len(stocks) < len(df_target_weights):
        new_tickers = set(df_target_weights['ticker'].to_list()) - set(portfolio['ticker'].to_list())
        for ticker in new_tickers:
            stocks.append(Stock(ticker=ticker,
                                quantity=0,
                                purchase_price=0,
                                target_weight=
                                df_target_weights[df_target_weights['ticker'] == ticker]['target_weight'].values[0],
                                current_market_price=get_ticker_close_price(ticker)[ticker]))

    return stocks


def main():
    st.title("Portfolio Rebalancing Tool")

    df_portfolio, fee, weight, stocks = None, None, None, None

    inputs, files = st.columns(2)

    with files:
        portfolio_csv = st.file_uploader("Upload portfolio file", type="csv")
        target_weights_file = st.file_uploader("Upload target weights file", type="csv")

    with inputs:

        with st.form("Inputs"):
            fee = st.number_input("Fee %", value=0.01, min_value=0.0, max_value=1.0, step=0.01)
            min_fee = st.number_input("Min Fee $", value=5.0, min_value=0.0, max_value=100.0, step=1.0)
            max_fee = st.number_input("Max Fee %", value=5.0, min_value=0.0, max_value=100.0, step=1.0)
            min_weight = st.number_input("Min Weight", value=0.03, min_value=0.0, max_value=1.0, step=0.01)
            max_weight = st.number_input("Max Weight", value=0.2, min_value=0.0, max_value=1.0, step=0.01)
            execution_probability = st.number_input("Execution Probability", value=0.95, min_value=0.0, max_value=1.0, step=0.01)
            button = st.form_submit_button("Submit to generate rebalancing trades")
            if button:
                fee = Fee(fee / 100, min_fee, max_fee / 100)
                weight = Weight(min_weight, max_weight)

    if portfolio_csv:
        df_portfolio = get_portfolio_df(portfolio_csv)
        st.header("Current portfolio state")
        st.write(df_portfolio)
        st.write("PNL: ", df_portfolio['pnl'].sum().round(2))

    if target_weights_file and df_portfolio is not None:
        stocks = get_stocks(df_portfolio, target_weights_file)

    if stocks and fee and weight:
        portfolio = Portfolio(stocks=stocks, execution_probability=execution_probability)
        rebalancer = PortfolioRebalancer(portfolio, fee=fee, weight=weight)

        st.header("Rebalancing trades")
        st.write(rebalancer.get_rebalancing_trades_df())

        rebalancer.generate_rebalancing_trades_csv()
        with open('rebalancing_trades.csv', 'r') as f:
            if st.download_button(label="Download rebalancing trades csv",
                                  data=f,
                                  file_name="rebalancing_trades.csv", mime="csv"):
                os.remove("reb_trades.csv")
                st.balloons()


if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
