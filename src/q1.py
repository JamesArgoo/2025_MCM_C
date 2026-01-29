import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor


def calculate_entropy_weights(df, cols):
    x = df[cols].copy()
    x = (x - x.min()) / (x.max() - x.min())
    x = x + 1e-9
    p = x / x.sum(axis=0)
    n = len(df)
    e = - (1 / np.log(n)) * (p * np.log(p)).sum(axis=0)
    d = 1 - e
    weights = d / d.sum()
    
    return weights

# 读取数据
try:
    medal_counts = pd.read_csv('./data/summerOly_medal_counts.csv')
    print("medal_counts 文件读取成功！")
    print(medal_counts.head())  # 打印前几行数据，检查列名和内容
except FileNotFoundError:
    print("错误：未找到 summerOly_medal_counts.csv 文件，请检查文件路径！")
    exit()
except UnicodeDecodeError:
    print("错误：medal_counts 文件编码格式不支持 UTF-8，请尝试其他编码格式！")
    exit()

try:
    athletes = pd.read_csv('./data/summerOly_athletes.csv')
    print("athletes 文件读取成功！")
    print(athletes.head())
except FileNotFoundError:
    print("错误：未找到 summerOly_athletes.csv 文件，请检查文件路径！")
    exit()
except UnicodeDecodeError:
    print("错误：athletes 文件编码格式不支持 UTF-8，请尝试其他编码格式！")
    exit()

try:
    hosts = pd.read_csv('./data/summerOly_hosts.csv')
    print("hosts 文件读取成功！")
    print(hosts.head())
except FileNotFoundError:
    print("错误：未找到 summerOly_hosts.csv 文件，请检查文件路径！")
    exit()
except UnicodeDecodeError:
    print("错误：hosts 文件编码格式不支持 UTF-8，请尝试其他编码格式！")
    exit()


# 尝试不同的编码格式读取 programs 文件
encodings = ['utf-8', 'GBK', 'ISO-8859-1', 'latin1']  # 常见的编码格式
programs = None
for encoding in encodings:
    try:
        programs = pd.read_csv('./data/summerOly_programs.csv', encoding=encoding)
        print(f"programs 文件读取成功！编码格式：{encoding}")
        print(programs.head())
        break
    except UnicodeDecodeError:
        print(f"尝试编码格式 {encoding} 失败，继续尝试其他编码格式...")
    except FileNotFoundError:
        print("错误：未找到 summerOly_programs.csv 文件，请检查文件路径！")
        exit()

if programs is None:
    print("错误：无法读取 programs 文件，请检查文件编码格式！")
    exit()

# 1. 数据整合
# 计算每个国家的运动员数量和获奖率
athlete_count = athletes.groupby(['Year', 'Team'])['Name'].nunique().reset_index()
athlete_count.columns = ['Year', 'Team', 'Athlete_Count']

medal_count = athletes[athletes['Medal'] != 'No medal'].groupby(['Year', 'Team'])['Name'].nunique().reset_index()
medal_count.columns = ['Year', 'Team', 'Medal_Count']

# 合并运动员数量和获奖率
athlete_medal = pd.merge(athlete_count, medal_count, on=['Year', 'Team'], how='left')
athlete_medal['Medal_Rate'] = athlete_medal['Medal_Count'] / athlete_medal['Athlete_Count']

# 合并奖牌数据和运动员数据
# 确保 medal_counts 的列名与 athlete_medal 的列名一致
medal_counts.rename(columns={'NOC': 'Team'}, inplace=True)  # 将 NOC 列重命名为 Team
data = pd.merge(medal_counts, athlete_medal, on=['Year', 'Team'], how='left')

# 合并主办国信息
# 检查 hosts 文件中的 Host 列和 medal_counts 文件中的 Team 列是否一致
print("hosts 文件中的 Host 列：")
print(hosts['Host'].unique())

print("medal_counts 文件中的 Team 列：")
print(medal_counts['Team'].unique())

# 将 hosts 文件中的 Host 列与 medal_counts 文件中的 Team 列对齐
hosts['Host'] = hosts['Host'].str.strip()  # 去除 Host 列中的空格
medal_counts['Team'] = medal_counts['Team'].str.strip()  # 去除 Team 列中的空格

# 合并主办国信息
data = pd.merge(data, hosts, on='Year', how='left')
data['Is_Host'] = data['Host'] == data['Team']  # 是否为主办国

# 检查合并后的主办国信息
print("合并后的主办国信息：")
print(data[data['Is_Host'] == 1][['Year', 'Team', 'Host', 'Is_Host']].head())

# 合并运动项目信息
# 将 programs 数据从宽表转换为长表 
programs_long = programs.melt(id_vars=['Sport', 'Discipline', 'Code', 'Sports Governing Body'], 
                              var_name='Year', value_name='Included')
programs_long = programs_long[programs_long['Included'] == 1]  # 只保留包含的项目

# 清理 Year 列中的非数字字符（如 '1906*'）
programs_long['Year'] = programs_long['Year'].str.replace(r'\D', '', regex=True)  # 移除非数字字符
programs_long['Year'] = programs_long['Year'].astype(int)  # 将年份转换为整数

# 计算每届奥运会的运动项目数量
program_counts = programs_long.groupby('Year')['Sport'].nunique().reset_index()
program_counts.columns = ['Year', 'Sport_Count']

# 合并运动项目数量
data = pd.merge(data, program_counts, on='Year', how='left')

# 2. 数据清洗
# 处理缺失值（使用平均值填充）
data['Athlete_Count'] = data['Athlete_Count'].fillna(data['Athlete_Count'].mean())
data['Medal_Count'] = data['Medal_Count'].fillna(data['Medal_Count'].mean())
data['Medal_Rate'] = data['Medal_Rate'].fillna(data['Medal_Rate'].mean())
data['Sport_Count'] = data['Sport_Count'].fillna(data['Sport_Count'].mean())

# 处理重复数据
data = data.drop_duplicates()

# --- 修复后的数据清洗部分 ---

def correct_outliers(series, window=3):
    # 强制转换为 float64，防止 int64 无法接收小数或 NaN
    s = series.astype(float).copy()
    
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR 
    upper = Q3 + 1.5 * IQR
    
    # 定义异常值掩码
    outliers = (s < lower) | (s > upper)
    
    # 计算平滑值：使用上文提到的 rolling.mean().shift()
    # 确保索引对齐
    replacements = s.rolling(window, min_periods=1).mean().shift(1)
    
    # 只有异常值位置被替换
    s.loc[outliers] = replacements.loc[outliers]
    
    # 处理可能产生的 NaN（比如开头第一个数就是异常值时）
    s = s.ffill().bfill() 
    return s

# 应用修复（注意：在标准化之前处理异常值）
data['Gold'] = correct_outliers(data['Gold'])
data['Athlete_Count'] = correct_outliers(data['Athlete_Count'])
data['Medal_Rate'] = correct_outliers(data['Medal_Rate'])
data['Sport_Count'] = correct_outliers(data['Sport_Count'])

# --- 3. 特征工程优化 ---
# 计算金牌占比时，注意处理分母为 0 的情况
data['Gold_Ratio'] = data['Gold'] / data['Total'].replace(0, np.nan)
data['Gold_Ratio'] = data['Gold_Ratio'].fillna(0)
data['Gold'] = correct_outliers(data['Gold'])
data['Athlete_Count'] = correct_outliers(data['Athlete_Count'])
data['Medal_Rate'] = correct_outliers(data['Medal_Rate'])
data['Sport_Count'] = correct_outliers(data['Sport_Count'])

# 3. 特征工程
data['Gold_Ratio'] = data['Gold'] / data['Total']  # 金牌占比
data['Is_Host'] = data['Is_Host'].astype(int)  # 是否为主办国

# 数值化分类变量
label_encoder = LabelEncoder()
data['Team_encoded'] = label_encoder.fit_transform(data['Team'])

scaler = StandardScaler()
data[['Gold', 'Athlete_Count', 'Medal_Rate', 'Sport_Count']] = scaler.fit_transform(
    data[['Gold', 'Athlete_Count', 'Medal_Rate', 'Sport_Count']]
)
target_cols = ['Gold', 'Silver', 'Bronze'] 
weights = calculate_entropy_weights(data, target_cols)

for col, w in zip(target_cols, weights):
    print(f"{col} 的权重: {w:.4f}")
print(weights)

