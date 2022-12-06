import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr

# global variables
stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
timeRange = 300  # in days
initialPortfolio = 10000
repeat = 100
timeFrame = timeRange // 3  # in days


# get yahoo stock data
def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']

    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=timeRange)

meanReturns, covMatrix = getData(stocks, startDate, endDate)

# random weights for the portfolio
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

#
# monte carlo simulation
#

# place to store data from simulations in memory
meanMatrix = np.full(shape=(timeFrame, len(weights)), fill_value=meanReturns)
meanMatrix = meanMatrix.T
simulationMatrix = np.full(shape=(timeFrame, repeat), fill_value=0.0)

for simulationNum in range(0, repeat):
    # daily returns formula
    Z = np.random.normal(size=(timeFrame, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanMatrix + np.inner(L, Z)

    # store data
    simulationMatrix[:, simulationNum] = np.cumprod(
        np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

# plotting results
plt.plot(simulationMatrix)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation Of Stock Portfolio')
plt.show()