#!/usr/bin/env python3
"""分析数据集格式，并创建 100 条小数据集用于 Exp-02 快速验证"""
import pandas as pd
import json

df = pd.read_parquet('data/train.parquet')
print(f'总行数: {len(df)}')
print(f'列名: {list(df.columns)}')

row = df.iloc[0]
print('\n=== 第一行完整结构 ===')
for col in df.columns:
    val = row[col]
    if col == 'extra_info':
        print(f'{col}: {json.dumps(val, ensure_ascii=False)[:300]}')
    else:
        print(f'{col}: {str(val)[:300]}')

print('\n=== prompt 内容示例 ===')
print(df['prompt'].iloc[0][:500])

print('\n=== data_source 分布 ===')
print(df['data_source'].value_counts().head(10))

# 抽样100条，保持分布
sample_df = df.sample(n=100, random_state=42)
print(f'\n抽样100条完成')
print(sample_df['data_source'].value_counts())

# 保存到 dataset/ 目录
output_path = 'data/web_search_agent_100.parquet'
sample_df.to_parquet(output_path, index=False)
print(f'已保存到 {output_path}')
