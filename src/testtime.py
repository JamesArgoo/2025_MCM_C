import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =================================================================
# 1. 基础数据准备 (同前)
# =================================================================

def load_and_prep_data():
    try:
        medal_counts = pd.read_csv('./data/summerOly_medal_counts.csv').rename(columns={'NOC': 'Team'})
        athletes = pd.read_csv('./data/summerOly_athletes.csv')
    except:
        print("请确保数据文件在 ./data/ 目录下")
        return None

    # 简单加权计算 Score
    # 这里为了速度，直接使用简单的加权，聚焦于算法测试
    data = medal_counts.copy()
    data['Score'] = data['Gold'] * 3 + data['Silver'] * 2 + data['Bronze'] * 1
    return data

# =================================================================
# 2. GM(1,1) 极简核心 (只用于回测)
# =================================================================

def gm11_predict_one_step(history_data):
    """
    输入历史数据列表，预测下一个点。
    返回: (prediction, residual_std)
    """
    x0 = np.array(history_data, dtype=float)
    n = len(x0)
    if n < 3: return np.nan, np.nan # 数据太少
    
    # 级比检验与平移
    lambdas = x0[:-1] / x0[1:]
    lower, upper = np.exp(-2/(n+1)), np.exp(2/(n+1))
    c_shift = 0
    if not np.all((lambdas > lower) & (lambdas < upper)):
        c_shift = np.max(x0) * 1.5
    
    x_process = x0 + c_shift
    x1 = np.cumsum(x_process)
    z1 = (x1[:-1] + x1[1:]) / 2.0
    
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x_process[1:].reshape(-1, 1)
    
    try:
        a, b = np.linalg.lstsq(B, Y, rcond=None)[0].flatten()
    except:
        return np.nan, np.nan
        
    # 预测下一个点 (index = n)
    def f(k):
        return (x_process[0] - b/a) * np.exp(-a * k) + b/a
    
    pred_process = f(n) - f(n-1)
    pred_final = max(0, pred_process - c_shift)
    
    # 计算拟合残差标准差 (用于衡量置信度)
    fitted_process = [f(k) - f(k-1) if k>0 else x_process[0] for k in range(n)]
    residuals = x_process - np.array(fitted_process)
    std = np.std(residuals)
    
    return pred_final, std

# =================================================================
# 3. 循环测试核心逻辑 (Loop Test)
# =================================================================

def run_window_optimization(data):
    # 1. 筛选数据丰富的 top 国家进行测试 (避免小国数据缺失干扰测试)
    top_teams = data.groupby('Team')['Score'].sum().nlargest(15).index.tolist()
    ts_pivot = data.pivot_table(index='Year', columns='Team', values='Score')
    
    # 定义测试参数
    # N的取值范围：最近4届 到 最近12届
    window_sizes = list(range(4, 13)) 
    # 回测年份：我们假装站在这些年份的前夕，去预测这些年份
    test_years = [2008, 2012, 2016, 2020, 2024]
    
    results = []
    
    print("开始进行滑动窗口超参数寻优 (Grid Search)...")
    
    for N in tqdm(window_sizes, desc="Testing Windows"):
        errors_mape = []
        uncertainties = [] # 记录残差标准差，代表置信区间的宽度
        
        for team in top_teams:
            series = ts_pivot[team].dropna()
            
            for target_year in test_years:
                if target_year not in series.index: continue
                
                # 获取真实值
                actual = series[target_year]
                
                # 获取该年份之前的 N 个数据作为训练集
                # 例如：预测 2016，N=4，则取 [2000, 2004, 2008, 2012]
                history = series[series.index < target_year]
                if len(history) < N: continue # 历史不够长，跳过
                
                train_window = history.iloc[-N:].values
                
                # 预测
                pred, std = gm11_predict_one_step(train_window)
                
                if not np.isnan(pred) and actual > 0:
                    # 记录误差 (MAPE)
                    mape = abs(pred - actual) / actual
                    # 过滤掉极端离谱的预测 (比如 MAPE > 200% 的异常点) 以免拉偏均值
                    if mape < 2.0: 
                        errors_mape.append(mape)
                        uncertainties.append(std)
        
        # 汇总该 N 下的所有测试结果
        avg_mape = np.mean(errors_mape) * 100
        avg_std = np.mean(uncertainties)
        
        results.append({
            'Window_Size (N)': N,
            'Avg_MAPE (%)': avg_mape,
            'Model_Uncertainty (Std)': avg_std
        })
        
    return pd.DataFrame(results)

# =================================================================
# 4. 运行与绘图
# =================================================================

df = load_and_prep_data()
if df is not None:
    res_df = run_window_optimization(df)
    
    print("\n测试结果：")
    print(res_df)
    
    # 寻找最佳 N
    best_row = res_df.loc[res_df['Avg_MAPE (%)'].idxmin()]
    best_N = int(best_row['Window_Size (N)'])
    print(f"\n>>> 最佳窗口大小是: {best_N} 届 (MAPE: {best_row['Avg_MAPE (%)']:.2f}%)")

    # --- 绘图 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Window Size (N - Recent Years Used)', fontsize=12)
    ax1.set_ylabel('Prediction Error (MAPE %)', color=color, fontsize=12)
    line1 = ax1.plot(res_df['Window_Size (N)'], res_df['Avg_MAPE (%)'], marker='o', color=color, linewidth=2, label='Accuracy (MAPE)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 双轴：画出不确定性 (标准差)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Model Uncertainty (Residual Std)', color=color, fontsize=12)
    line2 = ax2.plot(res_df['Window_Size (N)'], res_df['Model_Uncertainty (Std)'], marker='s', linestyle='--', color=color, label='Uncertainty (Width)')
    ax2.tick_params(axis='y', labelcolor=color)

    # 标注最佳点
    ax1.annotate(f'Optimal N={best_N}', 
                 xy=(best_N, best_row['Avg_MAPE (%)']), 
                 xytext=(best_N, best_row['Avg_MAPE (%)']+5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, fontweight='bold')

    plt.title('Hyperparameter Tuning: Optimal Window Size for Prediction', fontsize=14)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    plt.tight_layout()
    plt.show()
