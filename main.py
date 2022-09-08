import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller


df=pd.read_csv("DAX1v4.csv")

df.head()

df.tail()

df.drop(["Open","High","Low","Tickvol","vol"], axis=1, inplace=True)
df.head()

# Convert Month into Datetime
df['date']=pd.to_datetime(df['date'])

df.head()

df.set_index('date',inplace=True)

df.head()

df.plot()

plt.show()

test_result=adfuller(df['Close'])

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(df['Close'])


#differencing

df['Close First Difference'] = df['Close'] - df['Close'].shift(1)
df['Close'].shift(1)

df['Hour First Difference']=df['Close']-df['Close'].shift(60)
df.head(62)
## Again test dickey fuller test
adfuller_test(df['Hour First Difference'].dropna())
df['Hour First Difference'].plot()
plt.show()

#Auto Regressive Model

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Close'])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Hour First Difference'].iloc[61:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Hour First Difference'].iloc[61:],lags=40,ax=ax2)
plt.show()



# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(df['Close'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()
df['forecast']=model_fit.predict(start=80000,end=82105,dynamic=True)
df[['Close','forecast']].plot(figsize=(12,8))
plt.show()

model=sm.tsa.statespace.SARIMAX(df['Close'],order=(1, 1, 1),seasonal_order=(1,1,1,60))
results=model.fit()

df['forecast']=results.predict(start=80000,end=82105,dynamic=True)
df[['Close','forecast']].plot(figsize=(12,8))


from pandas.tseries.offsets import DateOffset
#future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_dates=[df.index[-1]+ DateOffset(minutes=x)for x in range(0,60)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()

future_df=pd.concat([df,future_datest_df])


future_df['forecast'] = results.predict(start = 82105, end = 82165, dynamic= True)  
future_df[['Close', 'forecast']].plot(figsize=(12, 8)) 



