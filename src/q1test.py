import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# 绘图风格设置
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# =================================================================
# 1. 新陈代谢版 GM(1,1) 类 (Metabolic GM(1,1))
# =================================================================

class GM11_Metabolic:
    def __init__(self):
        self.a = None
        self.b = None
        self.c_shift = 0
        self.x0_orig = None    # 存储用于训练的“窗口数据”
        self.fitted = None
        self.residuals = None
        self.metrics = {}

    def _step_ratio_check(self, x):
        """级比检验"""
        n = len(x)
        if n < 3: return True
        lambdas = x[:-1] / x[1:]
        lower, upper = np.exp(-2/(n+1)), np.exp(2/(n+1))
        return np.all((lambdas > lower) & (lambdas < upper))

    def fit(self, x0, window=6):
        """
        训练模型
        :param window: 回顾窗口大小，默认为6 (即只看最近6届奥运会)
        """
        # --- 核心改进：只截取最近 N 届数据 ---
        if len(x0) > window:
            self.x0_orig = np.array(x0[-window:], dtype=float)
        else:
            self.x0_orig = np.array(x0, dtype=float)
            
        # 1. 级比检验与自动平移
        if not self._step_ratio_check(self.x0_orig):
            # 如果数据波动大，加上一个较大的常数进行平滑
            self.c_shift = np.max(self.x0_orig) * 1.5
        else:
            self.c_shift = 0
            
        x_process = self.x0_orig + self.c_shift
        
        # 2. GM(1,1) 标准流程
        x1 = np.cumsum(x_process)
        z1 = (x1[:-1] + x1[1:]) / 2.0
        
        B = np.vstack([-z1, np.ones(len(z1))]).T
        Y = x_process[1:].reshape(-1, 1)
        
        try:
            self.a, self.b = np.linalg.lstsq(B, Y, rcond=None)[0].flatten()
        except:
            self.a, self.b = 0, 0
            
        # 3. 生成拟合值
        def f(k):
            return (x_process[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
        
        x1_hat = [f(k) for k in range(len(x_process))]
        x0_hat = np.diff(x1_hat, prepend=f(0))
        x0_hat[0] = x_process[0]
        
        self.fitted = x0_hat - self.c_shift
        self.residuals = self.x0_orig - self.fitted
        
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        res = self.residuals
        actual = self.x0_orig
        
        # MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs(res / actual)) * 100
        
        # C Ratio
        s1 = np.std(actual)
        s2 = np.std(res)
        c_val = s2 / (s1 + 1e-9)
        
        # P Value
        mean_res = np.mean(res)
        p_val = np.sum(np.abs(res - mean_res) < 0.6745 * s1) / len(actual)
        
        self.metrics = {'MAPE': mape, 'C': c_val, 'P': p_val}

    def predict(self, steps=1):
        if self.a is None: return np.nan, np.nan, np.nan
        
        n = len(self.x0_orig)
        x0_start = self.x0_orig[0] + self.c_shift
        
        def f(k):
            return (x0_start - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
        
        preds = []
        for k in range(n, n + steps):
            val = f(k) - f(k-1)
            preds.append(max(0, val - self.c_shift))
        
        pred_val = preds[-1]
        
        # 基于最近窗口残差的置信区间
        std_resid = np.std(self.residuals)
        ci_low = max(0, pred_val - 1.96 * std_resid)
        ci_high = pred_val + 1.96 * std_resid
        
        return pred_val, ci_low, ci_high

# =================================================================
# 2. 辅助函数
# =================================================================

def calculate_entropy_weights(df, cols):
    X = df[cols].copy()
    X = (X - X.min()) / (X.max() - X.min()) + 1e-9
    P = X / X.sum(axis=0)
    E = - (1 / np.log(len(df))) * (P * np.log(P)).sum(axis=0)
    d = 1 - E
    return d / d.sum()

def handle_outliers(series, threshold=1.5):
    """
    Z-Score 平滑。
    对于 '新陈代谢' 模型，阈值设为 1.5 比较合适，
    因为样本量变少了，需要更积极地压制像日本2020这样的极端值。
    """
    if len(series) < 3: return series
    mean, std = series.mean(), series.std()
    if std == 0: return series
    
    z_scores = np.abs((series - mean) / std)
    cleaned = series.copy()
    cleaned[z_scores > threshold] = mean # 用均值替换
    return cleaned

# =================================================================
# 3. 主程序
# =================================================================

# --- (A) 读取数据 ---
try:
    medal_counts = pd.read_csv('./data/summerOly_medal_counts.csv').rename(columns={'NOC': 'Team'})
    athletes = pd.read_csv('./data/summerOly_athletes.csv')
except:
    print("Error: Files not found.")
    exit()

athlete_summary = athletes.groupby(['Year', 'Team']).agg(Ath_Count=('Name', 'nunique')).reset_index()
data = pd.merge(medal_counts, athlete_summary, on=['Year', 'Team'], how='left').fillna(0)

# --- (B) 熵权计算 ---
target_cols = ['Gold', 'Silver', 'Bronze']
weights = calculate_entropy_weights(data, target_cols)
data['Score'] = (data[target_cols] * weights).sum(axis=1)

# --- (C) 活跃国筛选 ---
# 既然用新陈代谢模型，我们主要看 2008 以后
data_recent = data[data['Year'] >= 2008]
active_teams = data_recent.groupby('Team')['Score'].sum().sort_values(ascending=False).head(30).index.tolist()

# --- (D) 预测 ---
ts_pivot = data.pivot_table(index='Year', columns='Team', values='Score')
results = []
HOST_2028 = 'United States' if 'United States' in active_teams else 'USA'

print(f"开始对前 {len(active_teams)} 个国家进行新陈代谢 GM(1,1) 预测 (Window=5)...")

for team in tqdm(active_teams):
    series = ts_pivot[team].dropna()
    if len(series) < 4: continue
    
    # 1. 离群点平滑 (压制东道主尖峰)
    series_clean = handle_outliers(series, threshold=1.5)
    
    model = GM11_Metabolic()
    model.fit(series_clean.values, window=5) 
    
    # 3. 预测
    pred, low, high = model.predict(steps=1)
    
    # 4. 2028 东道主加成
    if team == HOST_2028:
        pred *= 1.15
        low *= 1.15
        high *= 1.15
        
    results.append({
        'Team': team,
        'Pred_2028': pred,
        'CI_Low': low,
        'CI_High': high,
        'MAPE': model.metrics['MAPE'],
        'C_Ratio': model.metrics['C'],
        'P_Value': model.metrics['P']
    })

# --- (E) 结果输出 ---
df_res = pd.DataFrame(results).sort_values(by='Pred_2028', ascending=False).round(3)

print("\n" + "="*60)
print("优化后的模型精度表 (New Metabolic GM(1,1))")
print("="*60)
# 重点检查 Japan, Great Britain, South Korea 的 MAPE 是否下降
print(df_res[['Team', 'MAPE', 'C_Ratio', 'P_Value']].head(15))

# --- (F) 可视化 ---
plt.figure(figsize=(14, 8))
top_plot = df_res.head(15)
x = range(len(top_plot))

# 绘制误差棒
plt.errorbar(x, top_plot['Pred_2028'], 
             yerr=[top_plot['Pred_2028'] - top_plot['CI_Low'], top_plot['CI_High'] - top_plot['Pred_2028']],
             fmt='o', color='crimson', ecolor='gray', elinewidth=2, capsize=4, label='Forecast (Recent Trend)')

# 绘制柱状图
plt.bar(x, top_plot['Pred_2028'], alpha=0.3, color='steelblue')

# 标注
for i, val in enumerate(top_plot['Pred_2028']):
    plt.text(i, val + (top_plot.iloc[i]['CI_High']-val)*1.1, f"{val:.1f}", ha='center', fontsize=9)

plt.xticks(x, top_plot['Team'], rotation=45, ha='right')
plt.ylabel('Weighted Composite Score')
plt.title('2028 Prediction using Metabolic GM(1,1)\n(Based on performance since ~2004)')
plt.legend()
plt.tight_layout()
plt.show()
