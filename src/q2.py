import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# =================================================================
# 1. 核心数学模型 (Core Models)
# =================================================================

def calculate_entropy_weights(df, cols):
    """客观权重计算：基于信息熵"""
    x = df[cols].copy()
    x = (x - x.min()) / (x.max() - x.min()) + 1e-9
    p = x / x.sum(axis=0)
    n = len(df)
    e = - (1 / np.log(n)) * (p * np.log(p)).sum(axis=0)
    return (1 - e) / (1 - e).sum()

def gm11_predict_2028(series):
    """灰色预测模型 GM(1,1)：带置信区间"""
    x0 = series.dropna().values
    if len(x0) < 4: return np.nan, np.nan, np.nan
    
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2.0
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x0[1:].reshape(-1, 1)
    
    try:
        a, b = np.linalg.lstsq(B, Y, rcond=None)[0].flatten()
        def f(k): return (x0[0] - b/a) * np.exp(-a*k) + b/a
        
        fitted_x1 = [f(i) for i in range(len(x0))]
        fitted_x0 = np.diff(fitted_x1, prepend=f(0))
        fitted_x0[0] = x0[0]
        
        prediction = f(len(x0)) - f(len(x0)-1)
        # 核心：残差决定了置信区间的宽度
        std_err = np.std(x0 - fitted_x0)
        return prediction, max(0, prediction - 1.96 * std_err), prediction + 1.96 * std_err
    except:
        return np.nan, np.nan, np.nan

# =================================================================
# 2. 异常值清洗逻辑 (Outlier Removal)
# =================================================================

def handle_outliers(df, column='Composite_Score', z_threshold=2.0):
    """
    识别各国的异常值（如1984年的美国），并用该国均值替换。
    减小方差，从而收窄置信区间。
    """
    df_cleaned = df.copy()
    for team in df_cleaned['Team'].unique():
        mask = df_cleaned['Team'] == team
        values = df_cleaned.loc[mask, column]
        if len(values) < 3: continue
        
        # 计算 Z-Score
        z_scores = np.abs((values - values.mean()) / (values.std() + 1e-9))
        
        # 发现异常点 (例如 1984 年美国得分可能远超均值 2 个标准差)
        outliers = z_scores > z_threshold
        if outliers.any():
            # 使用非异常值的均值进行填充（平滑处理）
            normal_mean = values[~outliers].mean()
            df_cleaned.loc[mask & (z_scores > z_threshold), column] = normal_mean
            
    return df_cleaned

# =================================================================
# 3. 完整预测流程 (Full Pipeline)
# =================================================================

# 加载数据
medal_counts = pd.read_csv('./data/summerOly_medal_counts.csv').rename(columns={'NOC': 'Team'})
athletes = pd.read_csv('./data/summerOly_athletes.csv')

# 基础整合
athlete_summary = athletes.groupby(['Year', 'Team']).agg(Ath_Count=('Name', 'nunique')).reset_index()
data = pd.merge(medal_counts, athlete_summary, on=['Year', 'Team'], how='left').fillna(0)

# 第一步：客观赋权
target_cols = ['Gold', 'Silver', 'Bronze']
weights = calculate_entropy_weights(data, target_cols)
data['Composite_Score'] = (data[target_cols] * weights).sum(axis=1)

# 第二步：剔除离群点 (优化关键：收窄美国等国的误差棒)
data = handle_outliers(data, column='Composite_Score', z_threshold=2.0)

# 第三步：指数级活跃度筛选 (确保苏联等已消失国家不进入预测)
data['Years_Ago'] = 2028 - data['Year']
data['Exp_Weight'] = np.exp(-0.2 * data['Years_Ago']) # Lambda=0.2 激进衰减
team_activity = data.groupby('Team').apply(lambda x: (x['Composite_Score'] * x['Exp_Weight']).sum())
recent_teams = data[data['Year'] >= 2012]['Team'].unique()
active_list = team_activity[(team_activity > 0.05) & (team_activity.index.isin(recent_teams))].index.tolist()

# 第四步：执行预测
ts_pivot = data.pivot_table(index='Year', columns='Team', values='Composite_Score')
forecast_results = []
HOST_2028 = 'USA'

for team in tqdm(active_list, desc="Forecasting with Outlier Handling"):
    series = ts_pivot[team]
    pred, low, high = gm11_predict_2028(series)
    
    # 东道主加成
    if team == HOST_2028 and not np.isnan(pred):
        pred, low, high = pred * 1.15, low * 1.15, high * 1.15
        
    if not np.isnan(pred):
        forecast_results.append({'Team': team, '2028_Score': pred, 'Lower_CI': low, 'Upper_CI': high})

forecast_df = pd.DataFrame(forecast_results).sort_values(by='2028_Score', ascending=False)

# =================================================================
# 4. 可视化 (Visualization)
# =================================================================

plt.figure(figsize=(14, 7))
top_15 = forecast_df.head(15)

# 使用更窄的误差棒展示
plt.errorbar(top_15['Team'], top_15['2028_Score'], 
             yerr=[top_15['2028_Score'] - top_15['Lower_CI'], top_15['Upper_CI'] - top_15['2028_Score']],
             fmt='o', color='#2C3E50', ecolor='#95A5A6', elinewidth=2, capsize=4, label='Forecast with Reduced Outlier Impact')

plt.title('2028 Olympic Performance Prediction (Outlier-Corrected)', fontsize=15)
plt.ylabel('Weighted Performance Score')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Final Results (Optimized) ---")
print(forecast_df.head(10))
