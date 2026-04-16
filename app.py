import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, permutations
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io
import re

# ==========================================
# 1. 终极 UI 兼容性控制 (强制高对比度浅色主题)
# ==========================================
st.set_page_config(page_title="qPCR 数据分析台", layout="wide", initial_sidebar_state="expanded")

# 强制注入 CSS：锁定颜色体系，防止深色模式劫持导致的不可见
st.markdown("""
    <style>
        /* 强制全局背景与文字颜色 */
        .stApp {
            background-color: #ffffff !important;
            color: #1e293b !important;
        }
        
        /* 侧边栏样式锁定 */
        [data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0;
        }
        
        /* 强制所有标签、说明文字、标题为深色 (防止白字出现) */
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] label, 
        .stMarkdown p, 
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
        label, .stText {
            color: #0f172a !important;
            font-weight: 500;
        }

        /* 输入框、下拉框控件外观强化 */
        .stSelectbox, .stTextInput, .stNumberInput {
            background-color: #ffffff !important;
            border-radius: 4px;
        }

        /* 顶部标签页 (Tabs) 模拟 RStudio 风格 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 2px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 10px 20px;
            background-color: #f8fafc !important;
            color: #64748b !important;
            border: 1px solid #e2e8f0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #2563eb !important;
            border-top: 3px solid #2563eb !important;
            border-bottom: 2px solid #ffffff !important;
            font-weight: bold;
        }

        /* 统一卡片容器 */
        .plot-container {
            border: 1px solid #e2e8f0;
            padding: 20px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心数学与统计算法
# ==========================================
def get_tukey_letters(tukey, means_s):
    res = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    res['group1'], res['group2'] = res['group1'].astype(str), res['group2'].astype(str)
    def is_diff(g1, g2):
        mask = ((res['group1'] == str(g1)) & (res['group2'] == str(g2))) | ((res['group1'] == str(g2)) & (res['group2'] == str(g1)))
        return res.loc[mask, 'reject'].values[0] if not res.loc[mask].empty else False

    sorted_groups = means_s.sort_values(ascending=False).index.astype(str).tolist()
    letters = {g: set() for g in sorted_groups}
    current_letter, cliques = 'a', []
    for g in sorted_groups:
        added = False
        for clique in cliques:
            if all(not is_diff(g, member) for member in clique['members']):
                clique['members'].add(g); letters[g].add(clique['letter']); added = True
        if not added:
            cliques.append({'letter': current_letter, 'members': {g}})
            letters[g].add(current_letter); current_letter = chr(ord(current_letter) + 1)
    return {g: "".join(sorted(list(ls))) for g, ls in letters.items()}

@st.cache_data 
def process_data_logic(df, target_gene, mode, ctrl, treat, k):
    # 此处包含 N选K 寻优逻辑
    gene_data = df[df['Gene'] == target_gene].copy()
    if gene_data.empty: return None
    all_ages = gene_data['Age'].unique() if mode != "paired" else [ctrl, treat]
    ctrl_reps = gene_data[gene_data['Age'] == ctrl]['Bio_Rep'].unique()
    if len(ctrl_reps) < k: return None

    all_scenarios = []
    for d0_combo in list(combinations(ctrl_reps, k)):
        d0_df = gene_data[(gene_data['Age'] == ctrl) & (gene_data['Bio_Rep'].isin(d0_combo))].drop_duplicates('Bio_Rep')
        d0_mean = d0_df['Delta_Ct_Mean'].values
        scenario_data, total_sd = [], 0
        
        for grp in all_ages:
            grp_df = gene_data[gene_data['Age'] == grp]
            if grp == ctrl:
                res = grp_df[grp_df['Bio_Rep'].isin(d0_combo)].copy()
                res['Bio_RelExp'] = res['Delta_Ct_Mean'].apply(lambda x: 2**-(x - d0_mean.mean()))
                scenario_data.append(res); total_sd += res['Bio_RelExp'].std()
            else:
                reps = grp_df['Bio_Rep'].unique()
                if len(reps) >= k:
                    best_sd, best_df = float('inf'), None
                    for combo in list(combinations(reps, k)):
                        test_vals = grp_df[grp_df['Bio_Rep'].isin(combo)].drop_duplicates('Bio_Rep')['Delta_Ct_Mean'].values
                        for p in permutations(test_vals):
                            rel = 2**-(np.array(p) - d0_mean)
                            if np.std(rel) < best_sd: 
                                best_sd = np.std(rel)
                                best_df = pd.DataFrame({'Gene': target_gene, 'Age': grp, 'Bio_RelExp': rel, 'Bio_Rep': combo})
                    scenario_data.append(best_df); total_sd += best_sd
        all_scenarios.append({'score': total_sd, 'data': pd.concat(scenario_data)})
    
    all_scenarios.sort(key=lambda x: x['score'])
    return all_scenarios[0]['data'] if all_scenarios else None

# ==========================================
# 3. 界面布局: 环境配置 (侧边栏)
# ==========================================
with st.sidebar:
    st.markdown("### 🗂️ 环境与数据")
    file = st.file_uploader("导入数据 (CSV/Excel)", type=["csv", "xlsx"])
    mode = st.radio("实验设计", ["trend", "paired", "grouped"], format_func=lambda x: {"trend":"单因素 (趋势)", "paired":"单因素 (配对)", "grouped":"多因素 (分组)"}[x])
    
    if file:
        raw = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        # 极简预处理：识别 Sample, Target, CT 列
        try:
            # 此处省略复杂的正则解析，假设数据已初步规范
            raw['Ct_Target'] = pd.to_numeric(raw.iloc[:, 2], errors='coerce') # 暂定第3列为CT
            raw['Gene'] = raw.iloc[:, 1].astype(str)
            raw['Age'] = raw.iloc[:, 0].astype(str).str.extract(r'(.*?)(\d+)$')[0]
            raw['Bio_Rep'] = raw.iloc[:, 0].astype(str).str.extract(r'(.*?)(\d+)$')[1]
            raw['Delta_Ct_Mean'] = raw['Ct_Target'] - 20 # 暂定内参为20
            unique_ages = sorted(raw['Age'].dropna().unique())
            
            if mode == "trend": ctrl = st.selectbox("归一化对照", unique_ages); treat = None
            elif mode == "paired": ctrl = st.selectbox("对照组", unique_ages); treat = st.selectbox("实验组", unique_ages, 1)
            else: ctrl = unique_ages[0]; treat = None # Grouped 逻辑略
            
            k = st.slider("重复数 (N选K)", 3, 5, 3)
            run = st.button("▶ 运行分析", use_container_width=True, type="primary")
        except: st.error("数据列识别失败，请检查文件格式。")

# ==========================================
# 4. 界面布局: 主视窗 (Source & Viewer)
# ==========================================
if file and 'run' in locals() and run:
    with st.spinner("正在处理数据..."):
        results = {}
        for g in raw['Gene'].unique():
            res = process_data_logic(raw, g, mode, ctrl, treat, k)
            if res is not None: results[g] = res
        st.session_state['results'] = results

if 'results' in st.session_state:
    res_dict = st.session_state['results']
    col_cfg, col_plt = st.columns([1, 2.5], gap="large")
    
    with col_cfg:
        st.markdown("### 📝 配置项")
        sel_g = st.selectbox("选择基因", list(res_dict.keys()))
        t1, t2, t3 = st.tabs(["图形样式", "坐标轴", "统计标注"])
        
        with t1:
            p_type = st.selectbox("图表类型", ["bar", "box", "violin"])
            palette = st.selectbox("期刊配色", ["npg", "aaas", "lancet", "nejm", "d3"])
            b_w = st.slider("柱体宽度", 0.1, 1.0, 0.6)
            show_pts = st.checkbox("显示散点", True)
        with t2:
            xlab = st.text_input("X轴标题", "Groups")
            ylab = st.text_input("Y轴标题", "Relative Expression")
            x_rot = st.slider("标签旋转", 0, 90, 0, 15)
        with t3:
            sig_size = st.slider("字号", 10, 30, 16)
            sig_offset = st.slider("垂直偏移", 0.01, 0.2, 0.05)

    with col_plt:
        st.markdown("### 📊 渲染结果")
        dat = res_dict[sel_g].copy()
        
        # 绘图逻辑
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(data=dat, x='Age', y='Bio_RelExp', width=b_w, palette=palette, ax=ax, edgecolor="black", capsize=0.1)
        if show_pts:
            sns.stripplot(data=dat, x='Age', y='Bio_RelExp', color="white", edgecolor="black", linewidth=1, size=5, jitter=0.1, ax=ax)
        
        # 显著性：防止重叠逻辑
        stats = dat.groupby('Age')['Bio_RelExp'].agg(['mean', 'std', 'max']).reset_index()
        y_max = stats['max'].max()
        if len(stats) > 2:
            tukey = pairwise_tukeyhsd(dat['Bio_RelExp'], dat['Age'])
            let_map = get_tukey_letters(tukey, dat.groupby('Age')['Bio_RelExp'].mean())
            for i, row in stats.iterrows():
                # 核心改进：字母位置 = 局部最大值 + 全局量级的偏移，确保不穿模
                ax.text(i, row['max'] + y_max * sig_offset, let_map.get(row['Age'], ""), ha='center', fontsize=sig_size, color="#000000")
        
        # 细节优化
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); sns.despine()
        plt.xticks(rotation=x_rot, ha="right" if x_rot > 0 else "center")
        
        st.pyplot(fig)
        
        # 导出区
        st.markdown("###### 导出高质量图表")
        c1, c2, c3 = st.columns(3)
        buf_png = io.BytesIO(); fig.savefig(buf_png, format="png", dpi=600, bbox_inches='tight')
        c1.download_button("📸 PNG (600 DPI)", buf_png.getvalue(), f"{sel_g}.png", "image/png", use_container_width=True)
        buf_pdf = io.BytesIO(); fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        c2.download_button("📄 PDF (矢量图)", buf_pdf.getvalue(), f"{sel_g}.pdf", "application/pdf", use_container_width=True)
        buf_svg = io.BytesIO(); fig.savefig(buf_svg, format="svg", bbox_inches='tight')
        c3.download_button("📐 SVG (可编辑)", buf_svg.getvalue(), f"{sel_g}.svg", "image/svg+xml", use_container_width=True)

    st.markdown("---")
    st.markdown("### 💻 终端数据检视")
    st.dataframe(dat, use_container_width=True)
else:
    st.info("👈 请在左侧上传数据文件，选择实验设计（趋势、配对或分组），然后点击「运行分析」。")