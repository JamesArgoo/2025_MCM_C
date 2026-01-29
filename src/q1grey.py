import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 优先使用Arial，备用黑体
plt.rcParams['axes.unicode_minus'] = False


class GM11_Robust:
    def __init__(self):
        self.a = None
        self.b = None
        self.c_shift = 0       # 平移常数
        self.x0_orig = None    # 原始数据
        self.fitted = None     # 拟合数据
        self.residuals = None  # 残差
        self.metrics = {}      # 检验指标
        
    def _step_ratio_check(self, x):
        """级比检验"""
        n = len(x)
        if n < 3: return True # 数据太少无法检验，默认通过
        lambdas = x[:-1] / x[1:]
        lower, upper = np.exp(-2/(n+1)), np.exp(2/(n+1))
        # 只要所有级比都在区间内，返回 True
        return np.all((lambdas > lower) & (lambdas < upper))

    def fit(self, x0):
        """训练模型"""
        self.x0_orig = np.array(x0, dtype=float)
        
        # 1. 级比检验与自动平移 (Translation Transformation)
        if not self._step_ratio_check(self.x0_orig):
            # 如果未通过，加上最大值以平滑数据 (启发式策略)
            self.c_shift = np.max(self.x0_orig) * 1.5
        else:
            self.c_shift = 0
            
        x_process = self.x0_orig + self.c_shift
        
        # 2. GM(1,1) 建模过程
        x1 = np.cumsum(x_process)                    # 累加生成 AGO
        z1 = (x1[:-1] + x1[1:]) / 2.0                # 紧邻均值生成
        
        B = np.vstack([-z1, np.ones(len(z1))]).T
        Y = x_process[1:].reshape(-1, 1)
        
        try:
            self.a, self.b = np.linalg.lstsq(B, Y, rcond=None)[0].flatten()
        except:
            self.a, self.b = 0, 0 # 奇异矩阵兜底
            
        # 3. 生成拟合值 (还原过程)
        def f(k):
            return (x_process[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
        
        x1_hat = [f(k) for k in range(len(x_process))]
        x0_hat = np.diff(x1_hat, prepend=f(0))
        x0_hat[0] = x_process[0] # 强制对其首项
        
        # 减去平移常数，得到最终拟合值
        self.fitted = x0_hat - self.c_shift
        self.residuals = self.x0_orig - self.fitted
        
        # 4. 计算模型精度指标 (用于论文表格)
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """计算 MAPE, C值, P值"""
        res = self.residuals
        actual = self.x0_orig
        
        # MAPE
        mape = np.mean(np.abs(res / actual)) * 100
        
        # 后验差比 C (Posterior Variance Ratio)
        s1 = np.std(actual)
        s2 = np.std(res)
        c_val = s2 / (s1 + 1e-9)
        
        # 小误差概率 P (Small Error Probability)
        mean_res = np.mean(res)
        p_val = np.sum(np.abs(res - mean_res) < 0.6745 * s1) / len(actual)
        
        self.metrics = {'MAPE': mape, 'C': c_val, 'P': p_val}

    def predict(self, steps=1):
        """向后预测 steps 步"""
        if self.a is None: return np.nan, np.nan, np.nan
        
        n = len(self.x0_orig)
        x0_start = self.x0_orig[0] + self.c_shift
        
        def f(k):
            return (x0_start - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
        
        preds = []
        # 预测序列：从 n 到 n+steps
        for k in range(n, n + steps):
            val = f(k) - f(k-1)
            preds.append(max(0, val - self.c_shift)) # 保证非负
        
        pred_val = preds[-1] # 取最后一步（2028）
        
        # 计算 95% 置信区间 (基于残差标准差)
        std_resid = np.std(self.residuals)
        ci_low = max(0, pred_val - 1.96 * std_resid)
        ci_high = pred_val + 1.96 * std_resid
        
        return pred_val, ci_low, ci_high

# =================================================================
# 2. 数据处理与辅助函数
# =================================================================

def calculate_entropy_weights(df, cols):
    """熵权法计算客观权重"""
    X = df[cols].copy()
    X = (X - X.min()) / (X.max() - X.min()) + 1e-9
    P = X / X.sum(axis=0)
    E = - (1 / np.log(len(df))) * (P * np.log(P)).sum(axis=0)
    d = 1 - E
    return d / d.sum()

def handle_outliers(series, threshold=2.0):
    """Z-Score 离群点平滑 (处理美国1984等异常)"""
    if len(series) < 5: return series
    mean, std = series.mean(), series.std()
    if std == 0: return series
    
    z_scores = np.abs((series - mean) / std)
    cleaned = series.copy()
    # 用均值替换异常值
    cleaned[z_scores > threshold] = mean
    return cleaned

# =================================================================
# 3. 主程序执行流程
# =================================================================

# --- (A) 加载数据 ---
print("正在读取数据...")
try:
    medal_counts = pd.read_csv('./data/summerOly_medal_counts.csv')
    athletes = pd.read_csv('./data/summerOly_athletes.csv')
except FileNotFoundError:
    print("错误：找不到文件，请确认 './data/' 目录下存在 csv 文件。")
    exit()

# 数据重命名与合并
medal_counts.rename(columns={'NOC': 'Team'}, inplace=True)
athlete_summary = athletes.groupby(['Year', 'Team']).agg(Ath_Count=('Name', 'nunique')).reset_index()
data = pd.merge(medal_counts, athlete_summary, on=['Year', 'Team'], how='left').fillna(0)

# --- (B) 熵权法赋权 ---
print("计算熵权得分...")
target_cols = ['Gold', 'Silver', 'Bronze']
weights = calculate_entropy_weights(data, target_cols)
print(f"   > 权重结果: 金={weights['Gold']:.3f}, 银={weights['Silver']:.3f}, 铜={weights['Bronze']:.3f}")
data['Score'] = (data[target_cols] * weights).sum(axis=1)

# --- (C) 筛选活跃国家 (指数衰减 + 硬约束) ---
print("筛选 2028 活跃参赛国...")
TARGET_YEAR = 2028
data['Time_Gap'] = TARGET_YEAR - data['Year']
# 指数衰减权重：历史越久远，权重越低 (Lambda=0.15)
data['Exp_Weight'] = np.exp(-0.15 * data['Time_Gap'])

team_activity = data.groupby('Team').apply(lambda x: (x['Score'] * x['Exp_Weight']).sum())
# 硬约束：必须在 2012 年(含)以后参加过奥运会
recent_participants = data[data['Year'] >= 2012]['Team'].unique()

active_teams = team_activity[
    (team_activity > 0.5) & (team_activity.index.isin(recent_participants))
].sort_values(ascending=False).index.tolist()

print(f"   > 筛选出 {len(active_teams)} 个活跃国家/地区")

# --- (D) 预测与验证循环 ---
print("开始 GM(1,1) 建模预测...")
ts_pivot = data.pivot_table(index='Year', columns='Team', values='Score')
results = []
HOST_2028 = 'United States' # 东道主 (请确保名称与 CSV 中一致，如 'USA' 或 'United States')

# 检查 CSV 中的美国名字
if 'United States' not in active_teams and 'USA' in active_teams:
    HOST_2028 = 'USA'

for team in tqdm(active_teams):
    series = ts_pivot[team].dropna()
    if len(series) < 4: continue # 数据太少不预测
    
    # 1. 离群点处理
    series_clean = handle_outliers(series)
    
    # 2. 建模
    model = GM11_Robust()
    model.fit(series_clean.values)
    
    # 3. 预测 2028 (假设数据截止到2024，向前预测1步)
    # 这里的 steps 取决于你的数据截止年份。如果是2024，预测2028就是 step 1。
    pred, low, high = model.predict(steps=1)
    
    # 4. 东道主加成 (Host Effect)
    if team == HOST_2028:
        pred *= 1.15
        low *= 1.15
        high *= 1.15
        
    # 5. 记录结果
    metrics = model.metrics
    results.append({
        'Team': team,
        'Pred_2028': pred,
        'CI_Low': low,
        'CI_High': high,
        'MAPE': metrics['MAPE'],
        'C_Ratio': metrics['C'],
        'P_Value': metrics['P'],
        'Passed_Step_Ratio': model.c_shift == 0 # 如果 c_shift > 0 说明进行了平移修复
    })

# --- (E) 结果整理与输出 ---
df_res = pd.DataFrame(results).sort_values(by='Pred_2028', ascending=False)
df_res = df_res.round(3)

print("\n" + "="*50)
print("TOP 10 预测结果 (2028 洛杉矶奥运会)")
print("="*50)
print(df_res[['Team', 'Pred_2028', 'CI_Low', 'CI_High']].head(10))

print("\n" + "="*50)
print("模型精度检验表 (用于论文附录)")
print("="*50)
# C < 0.35 为好, P > 0.95 为好
print(df_res[['Team', 'MAPE', 'C_Ratio', 'P_Value']].head(10))

# --- (F) 可视化绘图 ---
top_plot = df_res.head(15) # 只画前15名
plt.figure(figsize=(14, 7))

# 绘制误差棒
x = range(len(top_plot))
plt.errorbar(x, top_plot['Pred_2028'], 
             yerr=[top_plot['Pred_2028'] - top_plot['CI_Low'], top_plot['CI_High'] - top_plot['Pred_2028']],
             fmt='o', color='#d35400', ecolor='gray', elinewidth=2, capsize=5, label='Forecast with 95% CI')

# 绘制柱状图背景
plt.bar(x, top_plot['Pred_2028'], alpha=0.2, color='#f39c12')

# 标注东道主
host_idx = top_plot[top_plot['Team'] == HOST_2028].index
if not host_idx.empty:
    h_x = x[top_plot.index.get_loc(host_idx[0])]
    plt.text(h_x, top_plot.loc[host_idx[0], 'CI_High'] + 1, 'HOST (1.15x)', 
             ha='center', color='red', fontweight='bold')

plt.xticks(x, top_plot['Team'], rotation=45, ha='right')
plt.ylabel('Weighted Composite Score')
plt.title('2028 Olympic Performance Prediction: Top 15 Nations\n(Model: Entropy-Weighted GM(1,1) with Robust Filtering)')
plt.legend()
plt.tight_layout()
plt.show()

print("\n程序执行完毕。")
