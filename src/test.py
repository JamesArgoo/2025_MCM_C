import pandas as pd
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

print(athletes[athletes['Year']==2024])
