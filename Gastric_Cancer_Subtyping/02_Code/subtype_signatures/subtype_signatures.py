#!/usr/bin/env python3
# 亚型特征标记分析

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

def load_omics_data():
    """加载组学数据"""
    return {
        'expression': pd.read_csv("../01_Preprocessed/merged_expression.tsv", sep='\t', index_col=0),
        'methylation': pd.read_csv("../01_Preprocessed/methylation_betas_filtered.csv", index_col=0),
        'subtypes': pd.read_csv("../03_Results/Tables/final_subtypes.csv", index_col=0)
    }

def find_differential_features(omics_data, subtype_col='SNF_Cluster'):
    """寻找差异特征"""
    print("Identifying differential features...")
    
    results = []
    
    # 分析表达数据
    expr = omics_data['expression']
    for gene in expr.index:
        groups = [expr.loc[gene, omics_data['subtypes'][subtype_col] == st] 
                 for st in sorted(omics_data['subtypes'][subtype_col].unique())]
        
        # ANOVA检验
        _, pval = f_oneway(*groups)
        results.append({
            'feature': gene,
            'omics': 'expression',
            'p_value': pval
        })
    
    # 分析甲基化数据
    meth = omics_data['methylation']
    for probe in meth.index[:5000]:  # 限制探针数量
        groups = [meth.loc[probe, omics_data['subtypes'][subtype_col] == st] 
                 for st in sorted(omics_data['subtypes'][subtype_col].unique())]
        
        # ANOVA检验
        _, pval = f_oneway(*groups)
        results.append({
            'feature': probe,
            'omics': 'methylation',
            'p_value': pval
        })
    
    # 多重检验校正
    df = pd.DataFrame(results)
    df['fdr'] = multipletests(df['p_value'], method='fdr_bh')[1]
    
    # 筛选显著特征
    sig_features = df[df['fdr'] < 0.05].sort_values('fdr')
    
    return sig_features

def visualize_top_features(sig_features, omics_data, top_n=10):
    """可视化top特征"""
    print("Visualizing top features...")
    
    # 按组学类型分开
    expr_features = sig_features[sig_features['omics'] == 'expression'].head(top_n)
    meth_features = sig_features[sig_features['omics'] == 'methylation'].head(top_n)
    
    # 绘制表达特征
    plt.figure(figsize=(12, 6))
    for i, gene in enumerate(expr_features['feature']):
        plt.subplot(2, 5, i+1)
        sns.boxplot(
            x=omics_data['subtypes']['SNF_Cluster'],
            y=omics_data['expression'].loc[gene],
            palette='viridis'
        )
        plt.title(gene)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("../03_Results/Figures/top_expression_features.png")
    plt.close()
    
    # 绘制甲基化特征
    plt.figure(figsize=(12, 6))
    for i, probe in enumerate(meth_features['feature']):
        plt.subplot(2, 5, i+1)
        sns.boxplot(
            x=omics_data['subtypes']['SNF_Cluster'],
            y=omics_data['methylation'].loc[probe],
            palette='viridis'
        )
        plt.title(probe[:15] + "...")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("../03_Results/Figures/top_methylation_features.png")
    plt.close()

def save_signature_genes(sig_features):
    """保存特征标记基因"""
    print("Saving signature genes...")
    
    # 按组学类型分开保存
    sig_features[sig_features['omics'] == 'expression'].to_csv(
        "../03_Results/Tables/expression_signatures.csv", index=False)
    
    sig_features[sig_features['omics'] == 'methylation'].to_csv(
        "../03_Results/Tables/methylation_signatures.csv", index=False)

if __name__ == "__main__":
    # 加载数据
    omics_data = load_omics_data()
    
    # 寻找差异特征
    sig_features = find_differential_features(omics_data)
    
    # 可视化top特征
    visualize_top_features(sig_features, omics_data)
    
    # 保存结果
    save_signature_genes(sig_features)
    
    print("Subtype signature analysis completed!")