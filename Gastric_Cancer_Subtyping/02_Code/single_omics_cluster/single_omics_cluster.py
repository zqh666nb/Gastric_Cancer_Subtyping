#!/usr/bin/env python3
# 单组学聚类分析

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import consensus_clustering as cc
import umap

# 设置绘图风格
plt.style.use('seaborn')
sns.set_palette("husl")

def load_preprocessed_data():
    """加载预处理后的数据"""
    data_dir = "../01_Preprocessed/"
    return {
        'expression': pd.read_csv(f"{data_dir}/merged_expression.tsv", sep='\t', index_col=0),
        'methylation': pd.read_csv(f"{data_dir}/methylation_betas_filtered.csv", index_col=0),
        'cnv': pd.read_csv(f"{data_dir}/cnv_matrix.csv", index_col=0),
        'clinical': pd.read_csv(f"{data_dir}/clinical_annotated.csv")
    }

def expression_clustering(expression_data, n_clusters=3):
    """RNA-seq表达数据聚类"""
    print("Performing expression clustering...")
    
    # 数据标准化
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expression_data.T)
    
    # 一致性聚类
    consensus = cc.ConsensusCluster(n_clusters=n_clusters)
    consensus.fit(expr_scaled)
    
    # 获取聚类结果
    clusters = pd.Series(consensus.labels_, index=expression_data.columns, name='Expression_Cluster')
    
    # 可视化
    plot_cluster_heatmap(expr_scaled, clusters, "Expression")
    
    return clusters

def methylation_clustering(methylation_data, n_clusters=3):
    """甲基化数据聚类"""
    print("Performing methylation clustering...")
    
    # 选择变异最大的探针
    top_probes = methylation_data.var(axis=1).sort_values(ascending=False)[:5000].index
    meth_filtered = methylation_data.loc[top_probes]
    
    # UMAP降维
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(meth_filtered.T)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embedding)
    clusters = pd.Series(clusters, index=methylation_data.columns, name='Methylation_Cluster')
    
    # 可视化
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=clusters, palette='viridis', s=50)
    plt.title("Methylation UMAP Clustering")
    plt.savefig("../03_Results/Figures/methylation_umap.png")
    plt.close()
    
    return clusters

def integrate_clusters(cluster_results, clinical_data):
    """整合单组学聚类结果"""
    print("Integrating single-omics clusters...")
    
    # 合并所有聚类结果
    cluster_df = pd.concat(cluster_results, axis=1)
    cluster_df = cluster_df.merge(clinical_data.set_index('sample_id'), left_index=True, right_index=True)
    
    # 保存结果
    cluster_df.to_csv("../03_Results/Tables/single_omics_clusters.csv")
    
    return cluster_df

def plot_cluster_heatmap(data, clusters, title):
    """绘制聚类热图"""
    plt.figure(figsize=(12, 8))
    
    # 按聚类排序
    cluster_order = clusters.sort_values().index
    data_ordered = data[clusters.argsort()]
    
    # 绘制热图
    sns.heatmap(
        data_ordered.T,
        cmap='viridis',
        yticklabels=False,
        xticklabels=False,
        cbar_kws={'label': 'Z-score'}
    )
    
    plt.title(f"{title} Cluster Heatmap")
    plt.savefig(f"../03_Results/Figures/{title.lower()}_heatmap.png")
    plt.close()

if __name__ == "__main__":
    # 加载数据
    data = load_preprocessed_data()
    
    # 单组学聚类
    expr_clusters = expression_clustering(data['expression'])
    meth_clusters = methylation_clustering(data['methylation'])
    
    # 整合结果
    cluster_df = integrate_clusters([expr_clusters, meth_clusters], data['clinical'])
    
    print("Single-omics clustering completed!")