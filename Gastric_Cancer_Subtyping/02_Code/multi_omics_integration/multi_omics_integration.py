#!/usr/bin/env python3
# 多组学整合分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import mointegrator as moi  # 假设使用多组学整合库

def load_integration_data():
    """加载整合所需数据"""
    data_dir = "../01_Preprocessed/"
    return {
        'expression': pd.read_csv(f"{data_dir}/merged_expression.tsv", sep='\t', index_col=0),
        'methylation': pd.read_csv(f"{data_dir}/methylation_betas_filtered.csv", index_col=0),
        'cnv': pd.read_csv(f"{data_dir}/cnv_matrix.csv", index_col=0),
        'clinical': pd.read_csv(f"{data_dir}/clinical_annotated.csv"),
        'clusters': pd.read_csv("../03_Results/Tables/single_omics_clusters.csv", index_col=0)
    }

def perform_pca_integration(data_dict, n_components=50):
    """基于PCA的多组学整合"""
    print("Performing PCA-based integration...")
    
    # 对各组学数据分别进行PCA
    pca_results = {}
    for omics, df in data_dict.items():
        if omics == 'clinical':
            continue
            
        # 标准化
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.T)
        
        # PCA降维
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled)
        pca_results[omics] = pd.DataFrame(pca_result, index=df.columns)
    
    # 合并PCA结果
    integrated = pd.concat(pca_results.values(), axis=1)
    
    return integrated

def similarity_network_fusion(integrated_data, n_clusters=3):
    """相似性网络融合聚类"""
    print("Performing SNF integration...")
    
    # 使用MOI库进行SNF整合
    snf = moi.SNF(k=20, mu=0.5)
    snf_integrated = snf.fit_transform(integrated_data)
    
    # 谱聚类
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    clusters = spectral.fit_predict(snf_integrated)
    clusters = pd.Series(clusters, index=integrated_data.index, name='SNF_Cluster')
    
    return clusters

def visualize_integration(integrated_data, clusters):
    """可视化整合结果"""
    print("Visualizing integration results...")
    
    # t-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(integrated_data)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_result[:, 0], 
        y=tsne_result[:, 1], 
        hue=clusters,
        palette='viridis',
        s=50
    )
    plt.title("Multi-omics Integration (t-SNE)")
    plt.savefig("../03_Results/Figures/multiomics_tsne.png")
    plt.close()

def save_final_subtypes(clusters, clinical_data):
    """保存最终亚型分类"""
    print("Saving final subtypes...")
    
    # 合并临床数据
    subtype_df = clusters.to_frame().merge(
        clinical_data.set_index('sample_id'), 
        left_index=True, 
        right_index=True
    )
    
    # 保存结果
    subtype_df.to_csv("../03_Results/Tables/final_subtypes.csv")
    
    return subtype_df

if __name__ == "__main__":
    # 加载数据
    data = load_integration_data()
    
    # 多组学整合
    integrated = perform_pca_integration({
        'expression': data['expression'],
        'methylation': data['methylation'],
        'cnv': data['cnv']
    })
    
    # SNF聚类
    clusters = similarity_network_fusion(integrated)
    
    # 可视化
    visualize_integration(integrated, clusters)
    
    # 保存最终亚型
    final_subtypes = save_final_subtypes(clusters, data['clinical'])
    
    print("Multi-omics integration completed!")