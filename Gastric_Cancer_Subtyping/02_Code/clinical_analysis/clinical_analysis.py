#!/usr/bin/env python3
# 临床关联分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.formula.api import ols

def load_subtype_data():
    """加载亚型分类数据"""
    return pd.read_csv("../03_Results/Tables/final_subtypes.csv", index_col=0)

def survival_analysis(subtype_df):
    """生存分析"""
    print("Performing survival analysis...")
    
    # 初始化Kaplan-Meier绘图
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    # 按亚型分组绘制生存曲线
    for subtype in sorted(subtype_df['SNF_Cluster'].unique()):
        mask = subtype_df['SNF_Cluster'] == subtype
        kmf.fit(
            durations=subtype_df.loc[mask, 'OS_MONTHS'],
            event_observed=subtype_df.loc[mask, 'OS_STATUS'],
            label=f'Subtype {subtype}'
        )
        kmf.plot_survival_function()
    
    # 添加图表信息
    plt.title('Kaplan-Meier Survival Curve by Subtype')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.ylim(0, 1)
    
    # 保存图表
    plt.savefig("../03_Results/Figures/survival_analysis.png")
    plt.close()
    
    # 对数秩检验
    results = []
    for i in range(len(subtype_df['SNF_Cluster'].unique())):
        for j in range(i+1, len(subtype_df['SNF_Cluster'].unique())):
            mask_i = subtype_df['SNF_Cluster'] == i
            mask_j = subtype_df['SNF_Cluster'] == j
            
            result = logrank_test(
                subtype_df.loc[mask_i, 'OS_MONTHS'],
                subtype_df.loc[mask_j, 'OS_MONTHS'],
                event_observed_A=subtype_df.loc[mask_i, 'OS_STATUS'],
                event_observed_B=subtype_df.loc[mask_j, 'OS_STATUS']
            )
            
            results.append({
                'Subtype_A': i,
                'Subtype_B': j,
                'p_value': result.p_value
            })
    
    # 保存统计结果
    pd.DataFrame(results).to_csv("../03_Results/Tables/survival_stats.csv", index=False)

def clinical_feature_association(subtype_df):
    """临床特征关联分析"""
    print("Analyzing clinical feature associations...")
    
    results = []
    categorical_vars = ['STAGE', 'HISTOLOGY', 'GENDER', 'TREATMENT_RESPONSE']
    continuous_vars = ['AGE']
    
    # 分类变量分析 (Fisher精确检验)
    for var in categorical_vars:
        contingency = pd.crosstab(subtype_df['SNF_Cluster'], subtype_df[var])
        _, pvalue, _, _ = fisher_exact(contingency)
        
        results.append({
            'Feature': var,
            'Test': 'Fisher_exact',
            'p_value': pvalue
        })
    
    # 连续变量分析 (ANOVA)
    for var in continuous_vars:
        model = ols(f'{var} ~ C(SNF_Cluster)', data=subtype_df).fit()
        pvalue = model.f_pvalue
        
        results.append({
            'Feature': var,
            'Test': 'ANOVA',
            'p_value': pvalue
        })
    
    # 保存结果
    pd.DataFrame(results).to_csv("../03_Results/Tables/clinical_associations.csv", index=False)

def generate_clinical_summary(subtype_df):
    """生成临床特征汇总表"""
    print("Generating clinical summary...")
    
    # 按亚型分组统计
    summary = subtype_df.groupby('SNF_Cluster').agg({
        'AGE': ['mean', 'std'],
        'GENDER': lambda x: (x == 'Male').mean(),
        'STAGE': lambda x: x.mode()[0],
        'HISTOLOGY': lambda x: x.mode()[0],
        'OS_MONTHS': 'median',
        'OS_STATUS': 'mean'
    })
    
    # 重命名列
    summary.columns = [
        'Mean Age', 'Age SD', 
        'Male Proportion', 
        'Most Common Stage', 
        'Most Common Histology',
        'Median Survival (months)',
        'Mortality Rate'
    ]
    
    # 保存汇总表
    summary.to_csv("../03_Results/Tables/clinical_summary.csv")

if __name__ == "__main__":
    # 加载数据
    subtype_df = load_subtype_data()
    
    # 生存分析
    survival_analysis(subtype_df)
    
    # 临床特征关联
    clinical_feature_association(subtype_df)
    
    # 生成临床汇总
    generate_clinical_summary(subtype_df)
    
    print("Clinical analysis completed!")