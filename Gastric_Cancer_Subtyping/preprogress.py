# 胃癌多组学数据预处理 (XLSX 输入版本)
# 输入：00_RawData/ 下所有xlsx文件 输出：01_Preprocessed/

import os
import yaml
import pandas as pd
import numpy as np
from functools import reduce
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# 加载配置文件
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# 创建预处理目录
os.makedirs("01_Preprocessed", exist_ok=True)

def read_xlsx_sheets(file_path):
    """读取Excel文件中的所有sheet"""
    xl = pd.ExcelFile(file_path)
    return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

## 1. 临床数据预处理 ----
def preprocess_clinical():
    """处理临床数据Excel文件"""
    print("Processing clinical data...")
    
    # 读取临床数据Excel
    clin_data = pd.read_excel(
        "00_RawData/TCGA.STAD.sampleMap_STAD_clinicalMatrix.xlsx",
        na_values=["", "NA", "Not Available", "Unknown", "N/A"]
    )
    
    # 重命名列以匹配后续处理
    clin_clean = clin_data.rename(columns={
        "sampleID": "sample_id",
        "CDE_ID_3226963": "MSI_STATUS",  # MSI状态
        "_INTEGRATION": "INTEGRATION_ID"
    })
    
    # 选择关键字段
    clin_final = clin_clean[[
        "sample_id", "MSI_STATUS", "INTEGRATION_ID"
    ]]
    
    # 保存结构化数据
    clin_final.to_csv("01_Preprocessed/clinical_annotated.csv", index=False)
    return clin_final

## 2. 基因组数据预处理 ----
def preprocess_genomics():
    """处理基因组Excel数据"""
    print("Processing genomics data...")
    
    # 突变数据
    mut_data = pd.read_excel(
        "00_RawData/STAD_mc3_gene_level.xlsx"
    )
    
    # 转换为长格式
    mut_long = mut_data.melt(
        id_vars=["sample"],
        var_name="Tumor_Sample_Barcode",
        value_name="Mutation_Status"
    )
    
    # 过滤非零突变
    mut_filtered = mut_long[mut_long["Mutation_Status"] != 0]
    
    # 重命名列以匹配MAF格式
    maf_df = mut_filtered.rename(columns={
        "sample": "Hugo_Symbol",
        "Mutation_Status": "Variant_Classification"
    })
    
    # 添加必要的MAF列
    maf_df["Chromosome"] = "Unknown"
    maf_df["Start_Position"] = 0
    maf_df["Reference_Allele"] = "Unknown"
    maf_df["Tumor_Seq_Allele2"] = "Unknown"
    
    # 过滤低频突变 (≥3%样本中出现)
    variant_counts = maf_df['Hugo_Symbol'].value_counts()
    common_variants = variant_counts[variant_counts >= len(maf_df['Tumor_Sample_Barcode'].unique()) * 0.03].index
    maf_filtered = maf_df[maf_df['Hugo_Symbol'].isin(common_variants)]
    
    # 保存MAF文件
    maf_filtered.to_csv("01_Preprocessed/filtered_mutations.maf", sep="\t", index=False)
    
    # CNV数据
    cnv = pd.read_excel(
        "00_RawData/Gistic2_CopyNumber_Gistic2_all_thresholded.xlsx",
        index_col=0
    )
    cnv.to_csv("01_Preprocessed/cnv_matrix.csv")

## 3. 转录组数据预处理 ----
def preprocess_transcriptomics():
    """处理转录组Excel数据"""
    print("Processing transcriptomics data...")
    
    # 读取表达矩阵
    expr_data = pd.read_excel(
        "00_RawData/HiSeqV2.xlsx",
        index_col=0
    )
    
    # 过滤低表达基因 (至少在10%样本中TPM>1)
    expr_filtered = expr_data[
        (expr_data > 1).sum(axis=1) >= len(expr_data.columns) * 0.1
    ]
    
    # log2转换
    expr_log2 = np.log2(expr_filtered + 1)
    
    # 保存表达矩阵
    expr_log2.to_csv("01_Preprocessed/merged_expression.tsv", sep="\t", index=True)

## 4. 甲基化数据预处理 ----
def preprocess_methylation():
    """处理甲基化Excel数据"""
    print("Processing methylation data...")
    
    # 读取β值矩阵
    beta_values = pd.read_excel(
        "00_RawData/HumanMethylation450.xlsx",
        index_col=0
    )
    
    # 过滤探针
    # 1. 去除在>10%样本中缺失的探针
    beta_filtered = beta_values.dropna(thresh=len(beta_values.columns)*0.9, axis=0)
    
    # 2. 去除变异小的探针 (标准差<0.05)
    beta_filtered = beta_filtered[beta_filtered.std(axis=1) >= 0.05]
    
    # 保存过滤后的矩阵
    beta_filtered.to_csv("01_Preprocessed/methylation_betas_filtered.csv")

## 5. 数据整合 ----
def integrate_data():
    """整合多组学数据到统一样本集"""
    print("Integrating multi-omics data...")
    
    # 加载临床数据作为基准
    clinical = pd.read_csv("01_Preprocessed/clinical_annotated.csv")
    sample_ids = clinical["sample_id"].tolist()
    
    # 加载各组学数据
    omics_data = {
        "clinical": clinical.set_index("sample_id"),
        "expression": pd.read_csv("01_Preprocessed/merged_expression.tsv", sep="\t", index_col=0).T,
        "methylation": pd.read_csv("01_Preprocessed/methylation_betas_filtered.csv", index_col=0).T,
        "cnv": pd.read_csv("01_Preprocessed/cnv_matrix.csv", index_col=0).T,
        "mutations": pd.read_csv("01_Preprocessed/filtered_mutations.maf", sep="\t")
    }
    
    # 对齐样本
    common_samples = set(sample_ids)
    for name, data in omics_data.items():
        if isinstance(data, pd.DataFrame):
            common_samples = common_samples.intersection(data.index)
    
    # 保存对齐后的数据
    for name, data in omics_data.items():
        if isinstance(data, pd.DataFrame):
            data_filtered = data.loc[list(common_samples)] if name != "mutations" else \
                          data[data["Tumor_Sample_Barcode"].isin(common_samples)]
            output_path = f"01_Preprocessed/{name}_aligned.csv"
            data_filtered.to_csv(output_path)
    
    print(f"Final integrated dataset contains {len(common_samples)} samples")

## 6. 质量控制可视化 ----
def generate_qc_plots():
    """生成各数据质量控制的可视化图表"""
    print("Generating QC plots...")
    
    # 创建QC目录
    os.makedirs("01_Preprocessed/QC_Plots", exist_ok=True)
    
    # 临床数据QC
    clinical = pd.read_csv("01_Preprocessed/clinical_annotated.csv")
    
    # 生存时间分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data=clinical, x="MSI_STATUS", hue="INTEGRATION_ID", bins=30, kde=True)
    plt.title("MSI Status Distribution")
    plt.savefig("01_Preprocessed/QC_Plots/msi_status_distribution.png")
    plt.close()
    
    # 表达数据QC
    expr = pd.read_csv("01_Preprocessed/merged_expression.tsv", sep="\t", index_col=0)
    
    # 样本相关性热图
    plt.figure(figsize=(12, 10))
    sample_corr = expr.corr()
    sns.clustermap(sample_corr, cmap="viridis", figsize=(12, 10))
    plt.title("Sample Correlation Heatmap")
    plt.savefig("01_Preprocessed/QC_Plots/expression_sample_correlation.png")
    plt.close()

## 主函数 ----
if __name__ == "__main__":
    # 执行预处理流程
    preprocess_clinical()
    preprocess_genomics()
    preprocess_transcriptomics()
    preprocess_methylation()
    integrate_data()
    generate_qc_plots()
    
    print("Preprocessing pipeline completed!")