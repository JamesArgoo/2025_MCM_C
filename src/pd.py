import pandas as pd
stocks=pd.Series([1,3,5,6],index=['hel','lo','wor','dd']);
print(stocks);
print("描述统计信息\n",stocks.describe());
print(stocks.iloc[0]);
print(stocks.loc['hel']);#获取值

filenamestr='C:\\Users\\James\\github\\2025_MCM_C\\data\\summerOly_medal_counts.csv';
medal_counts=pd.read_csv(filenamestr);
print(medal_counts.head());
print(medal_counts.shape);

print(medal_counts.dtypes);
