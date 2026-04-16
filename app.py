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
# 强制浅色模式与 UI 深度优化 CSS
# ==========================================
st.set_page_config(page_title="qPCR 数据分析工作台", layout="wide", initial_sidebar_state="expanded")

# 这里的 CSS 增加了强制颜色覆盖 (!important)，防止被系统深色模式劫持
st.markdown("""
    <style>
        /* 1. 强制全局背景为浅色 */
        .stApp {
            background-color: #ffffff !important;
            color: #1e293b !important;
        }
        
        /* 2. 侧边栏样式锁定 */
        [data-testid="stSidebar"] {
            background-color: #f8fafc !important;
            border-right: 1px solid #e2e8f0;
        }
        
        /* 3. 侧边栏文字颜色强制为深色 */
        [data-testid="stSidebar"] .stText, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {
            color: #334155 !important;
        }

        /* 4. 顶部标签页 (Tabs) 模拟 RStudio/Dashboard */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #cbd5e1;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 10px 20px;
            background-color: #f1f5f9 !important;
            color: #64748b !important;
            border: 1px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #2563eb !important;
            border-top: 3px solid #2563eb !important;
            border-left: 1px solid #cbd5e1 !important;
            border-right: 1px solid #cbd5e1 !important;
            border-bottom: 1px solid #ffffff !important;
            font-weight: bold;
        }

        /* 5. 标题与分割线 */
        h3 {
            color: #0f172a !important;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }
        
        /* 6. 按钮美化 */
        .stButton>button {
            border-radius: 6px;
            border: none;
            background-color: #2563eb;
            color: white;
            padding: 0.5rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 0. 显著性算法
# ==========================================
def get_tukey_letters(tukey, means_s):
    res = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    res['group1'], res['group2'] = res['group1'].astype(str), res['group2'].astype(str)
    def is_diff(g1, g2):
        mask = ((res['group1'] == str(g1)) & (res['group2'] == str(g2))) | ((res['group1'] == str(g2)) & (res['group2'] == str(g1)))
        if mask.sum() == 0: return False
        return res.loc[mask, 'reject'].values[0]

    sorted_groups = means_s.sort_values(ascending=False).index.astype(str).tolist()
    letters = {g: set() for g in sorted_groups}
    current_letter, cliques = 'a', []
    for g in sorted_groups:
        added = False
        for clique in cliques:
            if all(not is_diff(g, member) for member in clique['members']):
                clique['members'].add(g)
                letters[g].add(clique['letter'])
                added = True
        if not added:
            cliques.append({'letter': current_letter, 'members': {g}})
            letters[g].add(current_letter)
            current_letter = chr(ord(current_letter) + 1)
    return {g: "".join(sorted(list(ls))) for g, ls in letters.items()}

# ==========================================
# 1. 计算引擎
# ==========================================
@st.cache_data 
def process_gene_logic(df, target_gene, mode, trend_ctrl, paired_ctrl, paired_treat, top_n, k_reps):
    gene_data = df[df['Gene'] == target_gene].copy()
    if gene_data.empty: return None
    ctrl_name, all_groups = (trend_ctrl, gene_data['Age'].unique()) if mode in ["trend", "grouped"] else (paired_ctrl, [paired_ctrl, paired_treat])
    if mode == "paired": gene_data = gene_data[gene_data['Age'].isin(all_groups)]
    ctrl_reps = gene_data[gene_data['Age'] == ctrl_name]['Bio_Rep'].unique()
    if len(ctrl_reps) < k_reps: return None
    all_scenarios_list = []
    for d0_combo in list(combinations(ctrl_reps, k_reps)):
        d0_base_df = gene_data[(gene_data['Age'] == ctrl_name) & (gene_data['Bio_Rep'].isin(d0_combo))].drop_duplicates(subset=['Bio_Rep', 'Delta_Ct_Mean']).sort_values('Bio_Rep')
        d0_ids, d0_dCts = d0_base_df['Bio_Rep'].tolist(), d0_base_df['Delta_Ct_Mean'].values
        current_scenario_data, current_scenario_score, valid_scenario = [], 0, True
        for grp in all_groups:
            age_df = gene_data[gene_data['Age'] == grp]
            if grp == ctrl_name:
                res = age_df[age_df['Bio_Rep'].isin(d0_combo)].copy()
                res['ddCt'] = res['Delta_Ct'] - res['Delta_Ct_Mean']
                res['Bio_RelExp'] = res.groupby('Bio_Rep')['Delta_Ct'].transform(lambda x: 2 ** -(x - d0_dCts.mean()))
                res['Pairing'] = "Control"
                current_scenario_data.append(res[['Gene', 'Age', 'Bio_Rep', 'Bio_RelExp', 'Pairing']].drop_duplicates())
                current_scenario_score += res['Bio_RelExp'].std()
            else:
                reps_in_age = age_df['Bio_Rep'].unique()
                if len(reps_in_age) >= k_reps:
                    best_age_sd, best_age_df = float('inf'), None
                    for ac in list(combinations(reps_in_age, k_reps)):
                        test_df = age_df[age_df['Bio_Rep'].isin(ac)].drop_duplicates(subset=['Bio_Rep', 'Delta_Ct_Mean'])
                        p_dCts_list = list(permutations(test_df['Delta_Ct_Mean']))
                        p_ids_list = list(permutations(test_df['Bio_Rep']))
                        for i in range(len(p_dCts_list)):
                            rel_exps = 2 ** -(np.array(p_dCts_list[i]) - d0_dCts)
                            if np.std(rel_exps) < best_age_sd:
                                best_age_sd = np.std(rel_exps)
                                best_age_df = pd.DataFrame({'Gene': target_gene, 'Age': grp, 'Bio_Rep': p_ids_list[i], 'Bio_RelExp': rel_exps, 'Pairing': "Matched"})
                    current_scenario_data.append(best_age_df); current_scenario_score += best_age_sd
                else: valid_scenario = False
        if valid_scenario: all_scenarios_list.append({'score': current_scenario_score, 'data': pd.concat(current_scenario_data)})
    if not all_scenarios_list: return None
    all_scenarios_list.sort(key=lambda x: x['score'])
    final_dfs = []
    for rank, scenario in enumerate(all_scenarios_list[:top_n]):
        df_rank = scenario['data'].copy(); df_rank['Rank'] = rank + 1; final_dfs.append(df_rank)
    return pd.concat(final_dfs)

def calc_simple_logic(df, target_gene, calc_type, mode, trend_ctrl, paired_ctrl, paired_treat):
    gene_data = df[df['Gene'] == target_gene].copy()
    if gene_data.empty: return None
    ctrl_name, all_groups = (trend_ctrl, gene_data['Age'].unique()) if mode in ["trend", "grouped"] else (paired_ctrl, [paired_ctrl, paired_treat])
    if calc_type == "raw":
        res = gene_data.copy(); res['Rank'], res['Pairing'] = 0, "Raw"
        return res[['Gene', 'Age', 'Bio_Rep', 'Bio_RelExp', 'Rank', 'Pairing']]
    if calc_type == "strict":
        gene_data_collapsed = gene_data.drop_duplicates(subset=['Age', 'Bio_Rep'])
        ctrl_df = gene_data_collapsed[gene_data_collapsed['Age'] == ctrl_name][['Bio_Rep', 'Delta_Ct_Mean']].rename(columns={'Delta_Ct_Mean': 'Ctrl_dCt'})
        res_list = []
        for grp in all_groups:
            joined = pd.merge(gene_data_collapsed[gene_data_collapsed['Age'] == grp], ctrl_df, on='Bio_Rep', how='inner')
            joined['Bio_RelExp'] = 2 ** -(joined['Delta_Ct_Mean'] - joined['Ctrl_dCt'])
            joined['Rank'], joined['Pairing'] = 0, "Strict"
            res_list.append(joined[['Gene', 'Age', 'Bio_Rep', 'Bio_RelExp', 'Rank', 'Pairing']])
        return pd.concat(res_list) if res_list else None

# ==========================================
# 2. 数据解析
# ==========================================
def prepare_data(file, calc_type, raw_cols):
    raw_df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    if calc_type == "raw":
        df = raw_df.rename(columns={raw_cols['gene']: 'Gene', raw_cols['group']: 'Age', raw_cols['val']: 'Bio_RelExp'})
        df['Bio_Rep'] = df[raw_cols['rep']].astype(str) if raw_cols['rep'] != "自动生成" else df.groupby(['Gene', 'Age']).cumcount().apply(lambda x: f"Rep{x+1}")
        return df.dropna(subset=['Bio_RelExp'])
    cols = raw_df.columns
    def find_col(keys):
        for col in cols:
            if any(k.lower() in col.lower() for k in keys): return col
        return None
    col_s, col_t, col_c = find_col(["Sample", "Name"]), find_col(["Target", "Gene"]), find_col(["CT", "Ct"])
    df = raw_df[[col_s, col_t, col_c]].copy(); df.columns = ['SN', 'TN', 'CTV']
    df['Ct_V'] = pd.to_numeric(df['CTV'], errors='coerce')
    df = df.dropna(subset=['Ct_V']).copy()
    extracted = df['SN'].astype(str).str.extract(r"(.*?)(\d+)$")
    df['Age'], df['Bio_Rep'] = extracted[0], extracted[1]
    df['Is_Ref'] = df['TN'].str.contains("-")
    df['Gene'] = df.apply(lambda row: re.sub(r".*-", "", str(row['TN'])) if row['Is_Ref'] else str(row['TN']), axis=1)
    df['Tech_Idx'] = df.groupby(['Gene', 'Age', 'Bio_Rep', 'Is_Ref']).cumcount()
    df_pivot = df.pivot(index=['Gene', 'Age', 'Bio_Rep', 'Tech_Idx'], columns='Is_Ref', values='Ct_V').reset_index()
    df_pivot = df_pivot.rename(columns={False: 'Ct_Target', True: 'Ct_Ref', 0: 'Ct_Target', 1: 'Ct_Ref'})
    df = df_pivot.dropna(subset=['Ct_Target', 'Ct_Ref']).copy()
    df['Ref_Mean'] = df.groupby(['Gene', 'Age', 'Bio_Rep'])['Ct_Ref'].transform('mean')
    df['Delta_Ct'] = df['Ct_Target'] - df['Ref_Mean']
    df['Delta_Ct_Mean'] = df.groupby(['Gene', 'Age', 'Bio_Rep'])['Delta_Ct'].transform('mean')
    return df

# ==========================================
# UI 侧边栏
# ==========================================
with st.sidebar:
    st.markdown("### 数据源配置")
    uploaded_file = st.file_uploader("上传文件 (CSV/Excel)", type=["csv", "xlsx"])
    calc_type = st.radio("处理模式", ["opt", "strict", "raw"], format_func=lambda x: {"opt": "智能寻优", "strict": "严格配对", "raw": "原始数据"}[x])
    raw_cols = {}
    if calc_type == "raw" and uploaded_file:
        temp_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        c_cols = temp_df.columns.tolist()
        raw_cols = {'gene': st.selectbox("基因列", c_cols, 0), 'group': st.selectbox("分组列", c_cols, 1), 'val': st.selectbox("数值列", c_cols, 2), 'rep': st.selectbox("重复列", ["自动生成"]+c_cols)}
    work_mode = st.radio("实验设计", ["trend", "paired", "grouped"], format_func=lambda x: {"trend": "单因素 (趋势)", "paired": "单因素 (配对)", "grouped": "多因素 (分组)"}[x])
    df_prepared, unique_ages = None, []
    if uploaded_file:
        try:
            df_prepared = prepare_data(uploaded_file, calc_type, raw_cols)
            unique_ages = sorted(df_prepared['Age'].astype(str).unique())
        except Exception as e: st.error(f"解析错误: {e}")
    group_mapping, group_ref_leg, ctrl_name, paired_ctrl, paired_treat, dynamic_xs = {}, None, None, None, None, []
    if work_mode == "grouped" and unique_ages:
        for grp in unique_ages:
            c1, c2 = st.columns(2); p = str(grp).replace("_", " ").split()
            map_x = c1.text_input(f"X轴({grp})", p[1] if len(p)>1 else grp, key=f"x_{grp}")
            map_leg = c2.text_input(f"图例({grp})", p[0] if len(p)>1 else grp, key=f"leg_{grp}")
            group_mapping[grp] = {'PLOT_X': map_x, 'PLOT_FILL': map_leg}; dynamic_xs.append(map_x)
        dynamic_xs = list(dict.fromkeys(dynamic_xs))
        all_legs = list(set([v['PLOT_FILL'] for v in group_mapping.values()]))
        if all_legs: group_ref_leg = st.selectbox("统计对照基准", all_legs)
    elif unique_ages:
        dynamic_xs = unique_ages
        if work_mode == "trend": ctrl_name = st.selectbox("归一化对照", unique_ages)
        else: paired_ctrl = st.selectbox("对照组", unique_ages); paired_treat = st.selectbox("实验组", unique_ages, 1); ctrl_name = paired_ctrl
    x_rename_dict, x_order = {}, []
    if dynamic_xs:
        with st.expander("排序与重命名"):
            x_order = st.multiselect("物理顺序", options=dynamic_xs, default=dynamic_xs)
            for v in x_order: x_rename_dict[v] = st.text_input(f"'{v}' 改名为:", v, key=f"rn_{v}")
    k_reps = st.slider("重复数极限", 3, 5, 3) if calc_type == "opt" else 3
    run_btn = st.button("运行分析", use_container_width=True, type="primary")

# ==========================================
# 核心渲染区
# ==========================================
if run_btn and df_prepared is not None:
    genes, all_results = df_prepared['Gene'].unique(), {}
    for g in genes:
        if work_mode == "grouped":
            sub_df = df_prepared[df_prepared['Gene'] == g].copy()
            sub_df['PLOT_X'] = sub_df['Age'].map(lambda x: group_mapping[x]['PLOT_X'])
            sub_df['PLOT_FILL'] = sub_df['Age'].map(lambda x: group_mapping[x]['PLOT_FILL'])
            res_list = []
            for ux in sub_df['PLOT_X'].unique():
                c_age = next((o for o, m in group_mapping.items() if m['PLOT_X'] == ux and m['PLOT_FILL'] == group_ref_leg), None)
                if c_age:
                    res = process_gene_logic(sub_df[sub_df['PLOT_X']==ux], g, "trend", c_age, None, None, 1, k_reps) if calc_type=="opt" else calc_simple_logic(sub_df[sub_df['PLOT_X']==ux], g, calc_type, "trend", c_age, None, None)
                    if res is not None: res_list.append(res)
            if res_list: all_results[g] = pd.concat(res_list)
        else:
            res = process_gene_logic(df_prepared, g, work_mode, ctrl_name, paired_ctrl, paired_treat, 12, k_reps) if calc_type=="opt" else calc_simple_logic(df_prepared, g, calc_type, work_mode, ctrl_name, paired_ctrl, paired_treat)
            if res is not None: all_results[g] = res
    st.session_state['results'] = all_results

if 'results' in st.session_state:
    results = st.session_state['results']
    c_src, c_view = st.columns([1, 2.5], gap="large")
    with c_src:
        st.markdown("### 图表设置")
        sel_g = st.selectbox("分析基因", list(results.keys()))
        t_g, t_t, t_s = st.tabs(["图形", "坐标", "标注"])
        with t_g:
            p_type = st.selectbox("类别", ["bar", "box", "violin"])
            pal = st.selectbox("色系", ["npg", "aaas", "lancet", "nejm", "jama", "d3"])
            b_w = st.slider("宽度", 0.2, 1.0, 0.7)
            pts = st.checkbox("显示散点", True)
        with t_t:
            xlab, ylab = st.text_input("X轴名", "Groups"), st.text_input("Y轴名", "Relative Expression")
            f_size = st.slider("字号", 8, 30, 14); x_rot = st.slider("旋转", 0, 90, 0, 15)
        with t_s:
            sig_z = st.slider("符号大小", 8, 30, 16); sig_y = st.slider("高度偏移", 0.0, 0.5, 0.05)

    with c_view:
        st.markdown("### 结果预览")
        dat = results[sel_g].copy()
        if work_mode == "grouped":
            dat['PLOT_X'] = dat['Age'].map(lambda x: group_mapping.get(x, {}).get('PLOT_X', x))
            dat['PLOT_FILL'] = dat['Age'].map(lambda x: group_mapping.get(x, {}).get('PLOT_FILL', x))
        else: dat['PLOT_X'], dat['PLOT_FILL'] = dat['Age'], dat['Age']
        dat = dat.dropna(subset=['PLOT_X', 'PLOT_FILL'])
        safe_x = [str(x) for x in x_order if str(x) in dat['PLOT_X'].values]
        hue_lv = list(dat['PLOT_FILL'].unique())
        if work_mode == "grouped" and group_ref_leg in hue_lv: hue_lv.insert(0, hue_lv.pop(hue_lv.index(group_ref_leg)))
        dat['PLOT_X'] = pd.Categorical(dat['PLOT_X'], categories=safe_x, ordered=True)
        dat['PLOT_FILL'] = pd.Categorical(dat['PLOT_FILL'], categories=hue_lv, ordered=True)
        dat = dat.sort_values(['PLOT_X', 'PLOT_FILL'])
        
        plt.rcParams['font.sans-serif'] = ['Arial']; fig, ax = plt.subplots(figsize=(6, 5))
        is_g = (work_mode == "grouped")
        sns.barplot(data=dat, x='PLOT_X', y='Bio_RelExp', hue='PLOT_FILL', dodge=is_g, width=b_w, ax=ax, palette=pal, edgecolor="black", errcolor="black", capsize=0.1)
        if pts: sns.stripplot(data=dat, x='PLOT_X', y='Bio_RelExp', hue='PLOT_FILL', dodge=is_g, ax=ax, color="white", edgecolor="black", linewidth=1, size=4, jitter=0.1)
        
        # 显著性计算与防遮挡标注
        stats = dat.groupby(['PLOT_X', 'PLOT_FILL'])['Bio_RelExp'].agg(['mean', 'std', 'max']).reset_index()
        y_max_all = stats['max'].max()
        for i, px in enumerate(safe_x):
            sub = dat[dat['PLOT_X'] == px]
            if len(hue_lv) > 2: # 多组 Tukey
                tukey = pairwise_tukeyhsd(sub['Bio_RelExp'], sub['PLOT_FILL'])
                let_map = get_tukey_letters(tukey, sub.groupby('PLOT_FILL')['Bio_RelExp'].mean())
                for j, h in enumerate(hue_lv):
                    x_p = i + (j - (len(hue_lv)-1)/2)*(b_w/len(hue_lv)) if is_g else i
                    y_p = stats[(stats['PLOT_X']==px)&(stats['PLOT_FILL']==h)]['max'].values[0]
                    ax.text(x_p, y_p + y_max_all*sig_y, let_map.get(h, ""), ha='center', fontsize=sig_z)
            elif len(hue_lv) == 2: # 两组 T-Test
                g1, g2 = sub[sub['PLOT_FILL']==hue_lv[0]]['Bio_RelExp'], sub[sub['PLOT_FILL']==hue_lv[1]]['Bio_RelExp']
                p_v = ttest_ind(g1, g2)[1]; sig = "***" if p_v<0.001 else "**" if p_v<0.01 else "*" if p_v<0.05 else "ns"
                y_p = max(g1.max(), g2.max()) + y_max_all*sig_y
                x1, x2 = i + (0 - 0.5)*(b_w/2) if is_g else 0, i + (1 - 0.5)*(b_w/2) if is_g else 1
                ax.plot([x1, x1, x2, x2], [y_p, y_p + y_max_all*0.05, y_p + y_max_all*0.05, y_p], color="black")
                ax.text((x1+x2)/2, y_p + y_max_all*0.06, sig, ha='center', fontsize=sig_z)

        ax.set_xticklabels([x_rename_dict.get(x, x) for x in safe_x], rotation=x_rot, ha="right" if x_rot>0 else "center")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); sns.despine()
        st.pyplot(fig)
        
        # 导出
        c1, c2, c3 = st.columns(3)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=600, bbox_inches='tight')
        c1.download_button("下载 PNG", buf.getvalue(), "plot.png", "image/png")
        buf_pdf = io.BytesIO(); fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
        c2.download_button("下载 PDF", buf_pdf.getvalue(), "plot.pdf", "application/pdf")
        
    st.markdown("---"); st.markdown("### 数据明细")
    st.dataframe(dat, use_container_width=True)
else:
    st.info("请在左侧上传数据并点击「运行分析」。")