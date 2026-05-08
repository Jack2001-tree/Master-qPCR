import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, permutations
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from scipy.stats import ttest_ind, shapiro, levene, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import io
import re
from streamlit_quill import st_quill

# ==========================================
# 页面设置
# ==========================================
st.set_page_config(page_title="qPCR 数据分析工作台", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 注入 Material You 与 究极悬浮 CSS
# ==========================================
st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC !important; color: #1e293b !important; }
        [data-testid="stSidebar"] { background-color: #EEF2F6 !important; border-right: none !important; }
        .stSelectbox div[data-baseweb="select"], .stTextInput input, .stNumberInput input { 
            background-color: #ffffff !important; border-radius: 12px !important; border: 1px solid #E2E8F0 !important; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none; background-color: #EEF2F6; padding: 8px; border-radius: 16px; }
        .stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 8px 16px; background-color: transparent !important; color: #64748b !important; border: none; }
        .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #0F172A !important; box-shadow: 0 2px 8px rgba(0,0,0,0.05); font-weight: 600; }
        [data-testid="column"]:nth-of-type(2) { height: 100% !important; }
        [data-testid="column"]:nth-of-type(2) > div {
            position: sticky !important; top: 2rem !important; max-height: calc(100vh - 4rem) !important; overflow-y: auto !important; 
            background-color: #ffffff; border-radius: 24px; padding: 24px; box-shadow: 0 8px 32px rgba(15, 23, 42, 0.08); border: 1px solid #F1F5F9; z-index: 10;
        }
        [data-testid="column"]:nth-of-type(2) > div::-webkit-scrollbar { width: 6px; }
        [data-testid="column"]:nth-of-type(2) > div::-webkit-scrollbar-thumb { background-color: #CBD5E1; border-radius: 10px; }
        [data-testid="column"]:nth-of-type(2) > div::-webkit-scrollbar-track { background: transparent; }
        .stButton button { border-radius: 24px !important; font-weight: 600; border: none; transition: all 0.2s ease; }
        .stButton button[kind="primary"] { background-color: #0A56D0 !important; color: white !important; }
        .stButton button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        h3 { padding-bottom: 0.5rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem; color: #1e293b; font-size: 1.25rem;}
        .toolbar-text { font-size: 0.85rem; color: #64748b; margin-bottom: 4px; display: block;}
        [data-testid="stElementContainer"] iframe {
            height: 120px !important; border-radius: 16px !important; border: 1px solid #E2E8F0 !important; background-color: #ffffff !important; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important; transition: all 0.3s ease;
        }
        [data-testid="stElementContainer"] iframe:hover { border-color: #CBD5E1 !important; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 0. 核心算法工具
# ==========================================
def get_tukey_letters(tukey, means_s):
    res = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    res['group1'], res['group2'] = res['group1'].astype(str), res['group2'].astype(str)
    def is_diff(g1, g2):
        g1, g2 = str(g1), str(g2)
        mask = ((res['group1'] == g1) & (res['group2'] == g2)) | ((res['group1'] == g2) & (res['group2'] == g1))
        if mask.sum() == 0: return False
        return res.loc[mask, 'reject'].values[0]

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

def get_letters_from_matrix(groups, reject_matrix, means):
    sorted_groups = sorted(groups, key=lambda g: means.get(g, 0), reverse=True)
    letters = {g: set() for g in sorted_groups}; current_letter, cliques = 'a', []
    def is_diff(g1, g2):
        if g1 == g2: return False
        return reject_matrix.get((g1, g2), reject_matrix.get((g2, g1), False))
    for g in sorted_groups:
        added = False
        for clique in cliques:
            if all(not is_diff(g, member) for member in clique['members']):
                clique['members'].add(g); letters[g].add(clique['letter']); added = True
        if not added:
            cliques.append({'letter': current_letter, 'members': {g}})
            letters[g].add(current_letter); current_letter = chr(ord(current_letter) + 1)
    return {g: "".join(sorted(list(ls))) for g, ls in letters.items()}

def perform_2group_test(g1, g2):
    if len(g1) >= 3 and len(g2) >= 3:
        _, p_n1 = shapiro(g1); _, p_n2 = shapiro(g2); _, p_l = levene(g1, g2)
        if p_n1 > 0.05 and p_n2 > 0.05:
            if p_l > 0.05: _, p = ttest_ind(g1, g2, equal_var=True)
            else: _, p = ttest_ind(g1, g2, equal_var=False)
        else: _, p = mannwhitneyu(g1, g2, alternative='two-sided')
    else: _, p = ttest_ind(g1, g2)
    return p

def perform_multigroup_test(df, val_col, group_col):
    groups = df[group_col].dropna().unique(); valid_groups_data = []
    for g in groups:
        g_data = df[df[group_col] == g][val_col].dropna().values
        if len(g_data) > 0: valid_groups_data.append(g_data)
    if len(valid_groups_data) < 3: return {}
    normal = True
    for g_data in valid_groups_data:
        if len(g_data) >= 3:
            _, pn = shapiro(g_data)
            if pn <= 0.05: normal = False
        else: normal = False
    homo = True
    if normal:
        try:
            _, pl = levene(*valid_groups_data)
            if pl <= 0.05: homo = False
        except: pass
    if normal and homo:
        try:
            tukey = pairwise_tukeyhsd(df[val_col], df[group_col].astype(str))
            means = df.groupby(group_col, observed=True)[val_col].mean()
            return get_tukey_letters(tukey, means)
        except: return {}
    else:
        try:
            reject_matrix = {}; pairs = list(combinations(groups, 2)); pvals = []
            for g1, g2 in pairs:
                d1 = df[df[group_col] == g1][val_col].dropna().values
                d2 = df[df[group_col] == g2][val_col].dropna().values
                if len(d1)>0 and len(d2)>0: _, p = mannwhitneyu(d1, d2, alternative='two-sided'); pvals.append(p)
                else: pvals.append(1.0)
            if pvals:
                reject, _, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                for i, (g1, g2) in enumerate(pairs): reject_matrix[(str(g1), str(g2))] = reject[i]
            means = df.groupby(group_col, observed=True)[val_col].mean().to_dict()
            return get_letters_from_matrix([str(g) for g in groups], reject_matrix, means)
        except: return {}

def unified_text_parser(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip(): return ""
    if raw_text in ["<p><br></p>", "<p></p>"]: return ""
    text = raw_text
    def _math_replace(match, tag):
        content = match.group(1).replace('%', r'\%').replace('$', r'\$').replace(' ', r'\ ')
        return f'$\\{tag}{{{content}}}$'
    text = text.replace("<p><br></p>", "\n").replace("</p><p>", "\n")
    text = text.replace("<p>", "").replace("</p>", "").replace("&nbsp;", " ")
    text = text.replace('\\n', '\n')
    text = re.sub(r'<strong><em>(.*?)</em></strong>', lambda m: _math_replace(m, 'mathbfit'), text)
    text = re.sub(r'<em><strong>(.*?)</strong></em>', lambda m: _math_replace(m, 'mathbfit'), text)
    text = re.sub(r'\*\*\*(.*?)\*\*\*', lambda m: _math_replace(m, 'mathbfit'), text)
    text = re.sub(r'<strong>(.*?)</strong>', lambda m: _math_replace(m, 'mathbf'), text)
    text = re.sub(r'<em>(.*?)</em>', lambda m: _math_replace(m, 'mathit'), text)
    text = re.sub(r'\*\*(.*?)\*\*', lambda m: _math_replace(m, 'mathbf'), text)
    text = re.sub(r'\*(.*?)\*', lambda m: _math_replace(m, 'mathit'), text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

@st.cache_data 
def process_gene_logic(df, target_gene, mode, trend_ctrl, paired_ctrl, paired_treat, top_n, k_reps):
    gene_data = df[df['Gene'] == target_gene].copy()
    if gene_data.empty: return None
    if mode in ["trend", "grouped"]: ctrl_name, all_groups = trend_ctrl, gene_data['Age'].unique()
    else: ctrl_name, all_groups = paired_ctrl, [paired_ctrl, paired_treat]; gene_data = gene_data[gene_data['Age'].isin(all_groups)]
    ctrl_reps = gene_data[gene_data['Age'] == ctrl_name]['Bio_Rep'].unique()
    if len(ctrl_reps) < k_reps: return None
    ctrl_combos = list(combinations(ctrl_reps, k_reps))
    all_scenarios_list = []
    for d0_combo in ctrl_combos:
        d0_base_df = gene_data[(gene_data['Age'] == ctrl_name) & (gene_data['Bio_Rep'].isin(d0_combo))].drop_duplicates(subset=['Bio_Rep', 'Delta_Ct_Mean']).sort_values('Bio_Rep')
        d0_ids, d0_dCts = d0_base_df['Bio_Rep'].tolist(), d0_base_df['Delta_Ct_Mean'].values
        current_scenario_data, current_scenario_score, valid_scenario = [], 0, True
        for grp in all_groups:
            age_df = gene_data[gene_data['Age'] == grp]
            if grp == ctrl_name:
                res = age_df[age_df['Bio_Rep'].isin(d0_combo)].copy()
                res['ddCt'] = res['Delta_Ct'] - res['Delta_Ct_Mean']
                res['RelExp_Hole'] = 2 ** (-res['ddCt'])
                res['Bio_RelExp'] = res.groupby('Bio_Rep')['RelExp_Hole'].transform('mean')
                res['Pairing'] = res['Bio_Rep'].astype(str) + " vs " + res['Bio_Rep'].astype(str)
                res = res[['Gene', 'Age', 'Bio_Rep', 'Bio_RelExp', 'Pairing']].drop_duplicates()
                current_scenario_data.append(res); current_scenario_score += res['Bio_RelExp'].std()
            else:
                reps_in_age = age_df['Bio_Rep'].unique()
                if len(reps_in_age) >= k_reps:
                    ac_combos = list(combinations(reps_in_age, k_reps))
                    best_age_sd, best_age_df = float('inf'), None
                    for ac in ac_combos:
                        test_df = age_df[age_df['Bio_Rep'].isin(ac)]
                        exp_dCt_df = test_df.drop_duplicates(subset=['Bio_Rep', 'Delta_Ct_Mean'])
                        p_dCts_list = list(permutations(exp_dCt_df['Delta_Ct_Mean']))
                        p_ids_list = list(permutations(exp_dCt_df['Bio_Rep']))
                        for i in range(len(p_dCts_list)):
                            rel_exps = 2 ** -(np.array(p_dCts_list[i]) - d0_dCts)
                            current_sd = np.std(rel_exps, ddof=1) 
                            if current_sd < best_age_sd:
                                best_age_sd = current_sd
                                best_age_df = pd.DataFrame({'Gene': [target_gene]*k_reps, 'Age': [grp]*k_reps, 'Bio_Rep': p_ids_list[i], 'Bio_RelExp': rel_exps, 'Pairing': [f"{p_ids_list[i][j]} vs {d0_ids[j]}" for j in range(k_reps)]})
                    current_scenario_data.append(best_age_df); current_scenario_score += best_age_sd
                else: valid_scenario = False
        if valid_scenario: all_scenarios_list.append({'key': "_".join(d0_combo), 'score': current_scenario_score, 'data': pd.concat(current_scenario_data)})
    if not all_scenarios_list: return None
    all_scenarios_list.sort(key=lambda x: x['score'])
    final_dfs = []
    for rank, scenario in enumerate(all_scenarios_list[:top_n]):
        df_rank = scenario['data'].copy(); df_rank['Rank'] = rank + 1; final_dfs.append(df_rank)
    return pd.concat(final_dfs)

@st.cache_data 
def calc_simple_logic(df, target_gene, calc_type, mode, trend_ctrl, paired_ctrl, paired_treat):
    gene_data = df[df['Gene'] == target_gene].copy()
    if gene_data.empty: return None
    if mode in ["trend", "grouped"]: ctrl_name, all_groups = trend_ctrl, gene_data['Age'].unique()
    else: ctrl_name, all_groups = paired_ctrl, [paired_ctrl, paired_treat]; gene_data = gene_data[gene_data['Age'].isin(all_groups)]
    if calc_type == "raw":
        res = gene_data.copy(); res['Rank'], res['Pairing'] = 0, "Raw Data"
        return res[['Gene', 'Age', 'Bio_Rep', 'Bio_RelExp', 'Rank', 'Pairing']]
    if calc_type == "strict":
        gene_data_collapsed = gene_data.drop_duplicates(subset=['Age', 'Bio_Rep'])
        ctrl_df = gene_data_collapsed[gene_data_collapsed['Age'] == ctrl_name][['Bio_Rep', 'Delta_Ct_Mean']].rename(columns={'Delta_Ct_Mean': 'Ctrl_dCt'})
        res_list = []
        for grp in all_groups:
            curr_df = gene_data_collapsed[gene_data_collapsed['Age'] == grp]
            joined = pd.merge(curr_df, ctrl_df, on='Bio_Rep', how='inner')
            joined['ddCt'] = joined['Delta_Ct_Mean'] - joined['Ctrl_dCt']
            joined['Bio_RelExp'] = 2 ** (-joined['ddCt'])
            joined['Rank'], joined['Pairing'] = 0, joined['Bio_Rep'].astype(str) + " (Matched)"
            res_list.append(joined[['Gene', 'Age', 'Bio_Rep', 'Bio_RelExp', 'Rank', 'Pairing']])
        if res_list: return pd.concat(res_list)
        return None
    return None

def prepare_data(file, calc_type, raw_cols):
    raw_df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    if calc_type == "raw":
        df = raw_df.rename(columns={raw_cols['gene']: 'Gene', raw_cols['group']: 'Age', raw_cols['val']: 'Bio_RelExp'})
        df = df.dropna(subset=['Bio_RelExp'])
        df['Bio_Rep'] = df[raw_cols['rep']].astype(str) if raw_cols['rep'] != "自动生成" and raw_cols['rep'] in raw_df.columns else df.groupby(['Gene', 'Age']).cumcount().apply(lambda x: f"Rep{x+1}")
        return df
    cols = raw_df.columns
    if all(c in cols for c in ["Gene", "Age", "Bio_Rep", "Ct_Target", "Ct_Ref"]):
        df = raw_df.copy()
        df['Ct_Target'], df['Ct_Ref'] = pd.to_numeric(df['Ct_Target'], errors='coerce'), pd.to_numeric(df['Ct_Ref'], errors='coerce')
        df = df.dropna(subset=['Ct_Target', 'Ct_Ref'])
    else:
        def find_col(keys):
            for col in cols:
                if any(k.lower() in col.lower() for k in keys): return col
            return None
        col_s, col_t, col_c = find_col(["Sample", "Name"]), find_col(["Target", "Gene"]), find_col(["CT", "Ct"])
        if not all([col_s, col_t, col_c]): raise ValueError("数据格式不匹配。")
        df = raw_df[[col_s, col_t, col_c]].copy()
        df.columns = ['SN', 'TN', 'CTV']
        df['Ct_V'] = pd.to_numeric(df['CTV'], errors='coerce')
        df = df.dropna(subset=['Ct_V']).copy()
        df['SN'] = df['SN'].astype(str).str.strip()
        extracted = df['SN'].str.extract(r"(.*?)(\d+)$")
        df['Age'], df['Bio_Rep'] = extracted[0], extracted[1]
        df = df.dropna(subset=['Age', 'Bio_Rep'])
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

# Initialize Session States
if 'break_ranges' not in st.session_state: st.session_state['break_ranges'] = "10-50"
if 'segment_ratios' not in st.session_state: st.session_state['segment_ratios'] = "1, 1"

# ==========================================
# UI 布局: 数据导入与模式配置 (侧边栏)
# ==========================================
with st.sidebar:
    st.markdown("### 数据源")
    uploaded_file = st.file_uploader("上传数据文件 (CSV/Excel)", type=["csv", "xlsx"])
    calc_type = st.radio("数据处理模式", ["opt", "strict", "raw"], format_func=lambda x: {"opt": "智能寻优", "strict": "严格配对", "raw": "原始数据"}[x])
    
    raw_cols = {}
    if calc_type == "raw" and uploaded_file:
        with st.expander("列名映射", expanded=True):
            temp_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            c_cols = temp_df.columns.tolist()
            raw_cols['gene'] = st.selectbox("基因列", c_cols, 0)
            raw_cols['group'] = st.selectbox("分组列", c_cols, 1 if len(c_cols)>1 else 0)
            raw_cols['val'] = st.selectbox("数值列", c_cols, 2 if len(c_cols)>2 else 0)
            raw_cols['rep'] = st.selectbox("重复列", ["自动生成"] + c_cols)
    
    work_mode = st.radio("实验设计", ["trend", "paired", "grouped"], format_func=lambda x: {"trend": "单因素 (趋势)", "paired": "单因素 (配对)", "grouped": "多因素 (分组)"}[x])
    
    df_prepared, unique_ages = None, []
    if uploaded_file:
        try:
            df_prepared = prepare_data(uploaded_file, calc_type, raw_cols)
            unique_ages = sorted(df_prepared['Age'].astype(str).unique())
        except Exception as e:
            st.error(f"数据解析失败: {e}")

    group_mapping, group_ref_leg, ctrl_name, paired_ctrl, paired_treat = {}, None, None, None, None
    dynamic_xs = []
    
    if work_mode == "grouped" and unique_ages:
        st.markdown("###### 多因素分组映射")
        mapped_x_list = []
        for grp in unique_ages:
            col1, col2 = st.columns(2)
            parts = str(grp).replace("_", " ").split()
            def_leg, def_x = (parts[0], "_".join(parts[1:])) if len(parts) > 1 else (grp, grp)
            map_x = col1.text_input(f"X轴 ({grp})", value=def_x, key=f"x_{grp}")
            map_leg = col2.text_input(f"图例 ({grp})", value=def_leg, key=f"leg_{grp}")
            group_mapping[grp] = {'PLOT_X': map_x, 'PLOT_FILL': map_leg}
            mapped_x_list.append(map_x)
        dynamic_xs = list(dict.fromkeys(mapped_x_list)) 
        all_legs = list(set([v['PLOT_FILL'] for v in group_mapping.values() if v['PLOT_FILL']]))
        if all_legs: group_ref_leg = st.selectbox("对照组 (统计基准)", all_legs)
    elif unique_ages:
        dynamic_xs = unique_ages
        if work_mode == "trend": ctrl_name = st.selectbox("归一化对照组:", unique_ages)
        else:
            paired_ctrl = st.selectbox("对照组:", unique_ages)
            paired_treat = st.selectbox("实验组:", unique_ages, index=1 if len(unique_ages)>1 else 0)
            ctrl_name = paired_ctrl
    
    x_rename_dict, x_order, legend_rename_dict = {}, [], {}
    if dynamic_xs:
        with st.expander("局部坐标与图例重命名", expanded=False):
            st.markdown("**X轴标签重命名** *(支持 `*斜体*` `**加粗**`)*")
            x_order = st.multiselect("坐标轴物理顺序", options=dynamic_xs, default=dynamic_xs)
            for orig_val in x_order:
                x_rename_dict[orig_val] = st.text_input(f"'{orig_val}' 显示为:", value=str(orig_val), key=f"rn_{orig_val}")
            
            if work_mode == "grouped":
                st.markdown("**图例重命名** *(支持 `*斜体*` `**加粗**`)*")
                for leg_val in all_legs:
                    legend_rename_dict[leg_val] = st.text_input(f"'{leg_val}' 显示为:", value=str(leg_val), key=f"leg_rn_{leg_val}")
            else:
                legend_rename_dict = x_rename_dict
        
    k_reps = st.slider("保留重复数 (N选K)", 3, 5, 3) if calc_type == "opt" else 3
    run_btn = st.button("运行分析", use_container_width=True, type="primary")

# ==========================================
# 后台数据运算
# ==========================================
if run_btn and df_prepared is not None:
    genes = df_prepared['Gene'].unique()
    all_results = {}
    with st.spinner('正在计算...'):
        for target_g in genes:
            if work_mode == "grouped":
                sub_df = df_prepared[df_prepared['Gene'] == target_g].copy()
                sub_df['PLOT_X'] = sub_df['Age'].map(lambda x: group_mapping[x]['PLOT_X'] if x in group_mapping else x)
                sub_df['PLOT_FILL'] = sub_df['Age'].map(lambda x: group_mapping[x]['PLOT_FILL'] if x in group_mapping else x)
                unique_xs, batch_res_list = sub_df['PLOT_X'].unique(), []
                for ux in unique_xs:
                    ctrl_age = next((orig_age for orig_age, m in group_mapping.items() if m['PLOT_X'] == ux and m['PLOT_FILL'] == group_ref_leg), None)
                    if ctrl_age:
                        ages_in_x = [orig_age for orig_age, m in group_mapping.items() if m['PLOT_X'] == ux]
                        sub_df_x = sub_df[sub_df['Age'].isin(ages_in_x)]
                        if calc_type == "opt":
                            res = process_gene_logic(sub_df_x, target_g, "trend", ctrl_age, None, None, 1, k_reps)
                            if res is not None: res = res[res['Rank'] == 1].copy()
                        else: res = calc_simple_logic(sub_df_x, target_g, calc_type, "trend", ctrl_age, None, None)
                        if res is not None: batch_res_list.append(res)
                if batch_res_list:
                    res_final = pd.concat(batch_res_list)
                    res_final['Rank'] = 1 if calc_type == "opt" else 0
                    all_results[target_g] = res_final
            else:
                if calc_type == "opt": res = process_gene_logic(df_prepared, target_g, work_mode, ctrl_name, paired_ctrl, paired_treat, 12, k_reps)
                else: res = calc_simple_logic(df_prepared, target_g, calc_type, work_mode, ctrl_name, paired_ctrl, paired_treat)
                if res is not None: all_results[target_g] = res
    st.session_state['results'] = all_results

# ==========================================
# UI 布局: 主视窗区域
# ==========================================
if 'results' not in st.session_state:
    st.info("请在左侧上传数据文件并完成基础设置，然后点击「运行分析」。")
elif not st.session_state['results']:
    st.error("计算结果为空。请检查重复数是否满足最低要求，或多因素模式中是否正确指定了对照组。")
else:
    results = st.session_state['results']
    col_source, col_viewer = st.columns([1, 2.5], gap="large")

    with col_source:
        st.markdown("### 🌟 全局样式预设")
        preset_style = st.selectbox("一键应用发表级排版", ["自定义", "样式1"], label_visibility="collapsed")
        
        if preset_style == "样式1":
            st.info("💡 已启用「样式1」预设：自动应用粉彩配色、无边框柱体及纯黑实心散点，您仍可自由调节下方各项尺寸与比例。")
            
        st.markdown("---")
        st.markdown("### 图表参数")
        selected_gene = st.selectbox("分析基因", list(results.keys()))
        
        # ✨ 预处理绘图数据 (为智能寻断提供上下文)
        dat = results[selected_gene].copy()
        if calc_type == "opt": dat = dat[dat['Rank'] == 1]
        
        if work_mode == "grouped":
            dat['PLOT_X'] = dat['Age'].map(lambda x: str(group_mapping.get(x, {}).get('PLOT_X', x)))
            dat['PLOT_FILL'] = dat['Age'].map(lambda x: str(group_mapping.get(x, {}).get('PLOT_FILL', x)))
        else: dat['PLOT_X'], dat['PLOT_FILL'] = dat['Age'].astype(str), dat['Age'].astype(str)
        dat['PLOT_Y'] = dat['Bio_RelExp']
        dat = dat.dropna(subset=['PLOT_X', 'PLOT_FILL'])
        
        safe_x_order = [str(x) for x in x_order if str(x) in dat['PLOT_X'].values]
        if not safe_x_order: safe_x_order = list(dat['PLOT_X'].unique())
        
        hue_levels = list(dat['PLOT_FILL'].unique())
        if work_mode == "grouped" and group_ref_leg in hue_levels: hue_levels.insert(0, hue_levels.pop(hue_levels.index(group_ref_leg)))
        
        dat['PLOT_X'] = pd.Categorical(dat['PLOT_X'], categories=safe_x_order, ordered=True)
        dat['PLOT_FILL'] = pd.Categorical(dat['PLOT_FILL'], categories=hue_levels, ordered=True)
        dat = dat.sort_values(['PLOT_X', 'PLOT_FILL'])
        
        st.markdown("<span class='toolbar-text'><b>图表主标题</b> (回车换行，划选文字加粗/斜体)</span>", unsafe_allow_html=True)
        plot_title_raw = st_quill(placeholder="请输入图表标题 (留空默认基因名)", html=True, toolbar=[["bold", "italic"]], key="quill_title")
        
        tab_geom, tab_theme, tab_canvas, tab_stats = st.tabs(["图形设置", "文本与排版", "坐标轴与画布", "显著性与图例"])
        
        with tab_geom:
            plot_type = st.selectbox("图形类别", ["bar", "box", "violin"], format_func=lambda x: {"bar": "柱状图", "box": "箱线图", "violin": "小提琴图"}[x])
            
            st.markdown("**色彩方案**")
            use_custom_cols = st.checkbox("自定义 HEX 颜色", False)
            if use_custom_cols: custom_cols = st.text_input("颜色列表 (逗号分隔)", "#E64B35, #4DBBD5, #00A087, #3C5488")
            else: palette_choice = st.selectbox("预设调色板", ["npg", "aaas", "lancet", "nejm", "jama", "jco", "d3", "igv", "cell"])
            
            st.markdown("**几何尺寸**")
            col_g1, col_g2 = st.columns(2)
            bar_edge_col_ui = col_g1.selectbox("柱/箱体边框", ["black", "none", "colored_hollow"], format_func=lambda x: "黑色" if x == "black" else ("空心填色" if x == "colored_hollow" else "无边框 (透明)"))
            err_dir_ui = col_g2.selectbox("误差棒方向", ["up", "both"], format_func=lambda x: "仅向上" if x == "up" else "双向 (上下对称)")

            bar_width, dodge_width, inner_gap = 0.6, 0.8, 0.05
            if work_mode == "grouped":
                col_w1, col_w2 = st.columns(2)
                dodge_width = col_w1.slider("组间总间距", 0.2, 1.0, 0.8)
                inner_gap = col_w2.slider("组内柱子间距", 0.0, 0.5, 0.05)
            else:
                bar_width = st.slider("柱体宽度", 0.2, 1.0, 0.6)
                inner_gap = 0.0
                
            err_width = st.slider("误差棒顶端宽度", 0.05, 0.5, 0.15)
            err_lwd = st.slider("误差棒粗细", 0.1, 3.0, 1.5)
            
            st.markdown("**散点层**")
            show_points = st.checkbox("显示样本散点", True)
            col_p1, col_p2 = st.columns(2)
            point_size = col_p1.slider("散点大小", 10, 100, 40)
            point_alpha = col_p2.slider("透明度", 0.1, 1.0, 1.0)
            point_fill_col = st.text_input("散点填充色 (hex/white/none)", "white")
            jitter_width = st.slider("水平散布范围", 0.0, 0.4, 0.1)

        with tab_theme:
            st.markdown("<span class='toolbar-text'><b>坐标轴标题</b> (回车换行，划选文字加粗/斜体)</span>", unsafe_allow_html=True)
            xlab_raw = st_quill(placeholder="请输入X轴标题...", html=True, toolbar=[["bold", "italic"]], key="quill_xlab")
            ylab_raw = st_quill(placeholder="请输入Y轴标题...", html=True, toolbar=[["bold", "italic"]], key="quill_ylab")
            
            st.markdown("**排版与字体**")
            font_family = st.selectbox("字体族", ["sans-serif", "serif", "monospace"])
            col_f1, col_f2 = st.columns(2)
            title_size = col_f1.slider("主标题大小", 10, 35, 18)
            title_style = col_f2.selectbox("主标题样式", ["normal", "italic", "bold"])
            col_f3, col_f4 = st.columns(2)
            xtitle_size = col_f3.slider("X轴标题大小", 8, 25, 14)
            ytitle_size = col_f4.slider("Y轴标题大小", 8, 25, 14)
            col_f5, col_f6 = st.columns(2)
            xtext_size = col_f5.slider("X轴刻度大小", 6, 20, 12)
            ytext_size = col_f6.slider("Y轴刻度大小", 6, 20, 12)
            
            st.markdown("**标题间距与对齐**")
            col_f7, col_f8 = st.columns(2)
            xtitle_loc = col_f7.selectbox("X标题对齐", ["center", "left", "right"])
            ytitle_loc = col_f8.selectbox("Y标题对齐", ["center", "top", "bottom"])
            x_label_pad = col_f7.slider("X标题间距", 0, 100, 10)
            y_label_pad = col_f8.slider("Y标题间距", 0, 100, 35)

        with tab_canvas:
            st.markdown("**画布尺寸**")
            col_c1, col_c2 = st.columns(2)
            num_w = col_c1.number_input("画布宽 (英寸)", value=6.0, step=0.5)
            num_h = col_c2.number_input("画布高 (英寸)", value=5.0, step=0.5)
            
            st.markdown("**坐标轴细节**")
            x_angle = st.slider("X轴标签旋转角度", 0, 90, 0, step=15)
            axis_lwd = st.slider("坐标轴线粗细", 0.5, 4.0, 1.5)
            y_top_gap = st.slider("Y轴顶部留白比例", 0.05, 0.5, 0.2)
            
            st.markdown("---")
            st.markdown("**多重断轴架构 (Multi-Break)**")
            use_broken_y = st.checkbox("启用Y轴断轴", False)
            
            # ✨ 激进智能寻断引擎
            if use_broken_y:
                if st.button("✨ 智能寻断"):
                    intervals = []
                    for (px, pfill), grp in dat.groupby(['PLOT_X', 'PLOT_FILL']):
                        vals = grp['PLOT_Y'].dropna().values
                        if len(vals) == 0: continue
                        m = np.mean(vals)
                        s = np.std(vals, ddof=1) if len(vals)>1 else 0
                        top_bound = m + s; max_pt = np.max(vals); min_pt = np.min(vals)
                        highest = max(top_bound, max_pt); lowest = min(m, min_pt)
                        intervals.append([max(0, lowest * 0.90), highest * 1.25])
                    
                    intervals.sort(key=lambda x: x[0])
                    merged = []
                    for iv in intervals:
                        if not merged: merged.append(iv)
                        else:
                            prev = merged[-1]
                            if iv[0] <= prev[1]: prev[1] = max(prev[1], iv[1])
                            else: merged.append(iv)
                            
                    valid_gaps, global_max = [], dat['PLOT_Y'].max()
                    threshold = global_max * 0.015  
                    for i in range(len(merged)-1):
                        g_start, g_end = merged[i][1], merged[i+1][0]
                        if g_end - g_start > threshold: valid_gaps.append((g_start, g_end))
                        
                    if not valid_gaps: st.warning("数据连续，暂未发现合适的断裂带。")
                    else:
                        st.session_state['break_ranges'] = ", ".join([f"{g[0]:.1f}-{g[1]:.1f}" for g in valid_gaps])
                        visible_spans, last_val = [], 0.0
                        for g_start, g_end in valid_gaps:
                            visible_spans.append(max(0.1, g_start - last_val)); last_val = g_end
                        top_max = max(merged[-1][1], global_max * 1.15)
                        visible_spans.append(max(0.1, top_max - last_val))
                        visible_spans.reverse()
                        
                        min_span = min(visible_spans)
                        damped_ratios = [(s / min_span) ** 0.4 for s in visible_spans]
                        st.session_state['segment_ratios'] = ", ".join([str(min(2.5, max(1.0, round(r, 1)))) for r in damped_ratios])
                        st.rerun()

                break_ranges = st.text_input("断开区间 (如: 10-50, 80-120)", key='break_ranges')
                segment_ratios = st.text_input("各段高度比 (逗号分隔)", key='segment_ratios')
                break_slash_size = st.slider("斜杠尺寸", 0.005, 0.050, 0.015, step=0.005)

        with tab_stats:
            st.markdown("**检验标识**")
            sig_format = st.selectbox("显示格式 (针对2组)", ["星号 (***)", "准确P值"])
            col_s1, col_s2 = st.columns(2)
            sig_size = col_s1.slider("字母/星号大小", 8, 25, 14)
            sig_lwd = col_s2.slider("连线粗细", 0.5, 4.0, 1.5)
            col_s3, col_s4 = st.columns(2)
            sig_y_offset = col_s3.slider("高度偏移系数", 0.0, 0.5, 0.05)
            sig_tip_len = col_s4.slider("连线括号下探量", 0.0, 0.2, 0.05)
            
            st.markdown("**图例配置**")
            show_legend = st.checkbox("显示图例", True)
            st.markdown("<span class='toolbar-text'><b>图例标题</b></span>", unsafe_allow_html=True)
            legend_title_raw = st_quill(placeholder="Condition", html=True, toolbar=[["bold", "italic"]], key="quill_legend")
            
            col_l1, col_l2, col_l3_align = st.columns([1.5, 1, 1])
            legend_pos = col_l1.selectbox("图例位置", ["right", "top", "bottom", "left", "custom"], index=0)
            legend_text_size = col_l2.slider("图例文字大小", 6, 20, 10)
            legend_title_align = col_l3_align.selectbox("对齐方式", ["left", "center", "right"], index=0, format_func=lambda x: {"left": "左对齐", "center": "居中", "right": "右对齐"}[x])
            
            if legend_pos == "custom":
                col_l3, col_l4 = st.columns(2)
                leg_x = col_l3.slider("X轴坐标", -0.5, 1.5, 0.8)
                leg_y = col_l4.slider("Y轴坐标", -0.5, 1.5, 0.8)
            else: leg_x, leg_y = 1.05, 0.5

    # ================= Pane 2: 图表渲染区 =================
    with col_viewer:
        st.markdown("### 图表预览")
        
        palettes = {"npg": ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2","#DC0000","#7E6148"], "aaas": ["#3B4992","#EE0000","#008B45","#631879","#008280","#BB0021","#5F559B","#A20056","#808180"], "lancet": ["#00468B","#ED0000","#42B540","#0099B4","#925E9F","#FDAF91","#AD002A","#ADB6B6","#1B1919"], "nejm": ["#BC3C29","#0072B5","#E18727","#20854E","#7876B1","#6F99AD","#FFDC91","#EE4C97"], "jama": ["#374E55","#DF8F44","#00A1D5","#B24745","#79AF97","#6A6599","#80796B"], "jco": ["#0073C2","#EFC000","#868686","#CD534C","#7AA6DC","#003C67","#8F7700","#3B3B3B"], "d3": ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F"], "igv": ["#5050FF","#FF5050","#00BD00","#FFB900","#00D6FF","#B973FF","#006600","#6600CC"], "cell": ["#C2C1D5", "#F6A6A5", "#9ED2C4", "#8BB4D5", "#D8C4E0", "#EED7A1"]}
        
        if preset_style == "样式1":
            final_cols = palettes["cell"]
            bar_edge_col = "none"; point_fill_col_render = "black"; point_edge_col = "none"
            inner_gap_render = 0.15 if work_mode == "grouped" else inner_gap
            err_direction = "both"
            point_size_render, point_alpha_render, jitter_render = point_size, point_alpha, jitter_width
            font_family_render, title_size_render, xtitle_size_render, ytitle_size_render, xtext_size_render, ytext_size_render = font_family, title_size, xtitle_size, ytitle_size, xtext_size, ytext_size
            axis_lwd_render, err_lwd_render, err_width_render, sig_lwd_render, sig_size_render, legend_text_size_render = axis_lwd, err_lwd, err_width, sig_lwd, sig_size, legend_text_size
            num_h_render, num_w_render = num_h, num_w
        else:
            final_cols = [c.strip() for c in custom_cols.split(',')] if use_custom_cols else palettes[palette_choice]
            bar_edge_col = bar_edge_col_ui; point_fill_col_render = "none" if point_fill_col.lower() == "none" else point_fill_col
            point_edge_col = "black" if point_fill_col_render != "none" else "black"
            inner_gap_render, err_direction = inner_gap, err_dir_ui
            point_size_render, point_alpha_render, jitter_render = point_size, point_alpha, jitter_width
            font_family_render, title_size_render, xtitle_size_render, ytitle_size_render, xtext_size_render, ytext_size_render = font_family, title_size, xtitle_size, ytitle_size, xtext_size, ytext_size
            axis_lwd_render, err_lwd_render, err_width_render, sig_lwd_render, sig_size_render, legend_text_size_render = axis_lwd, err_lwd, err_width, sig_lwd, sig_size, legend_text_size
            num_w_render, num_h_render = num_w, num_h
            
        color_map = {hue: final_cols[idx % len(final_cols)] for idx, hue in enumerate(hue_levels)}
        plt.rcParams['font.family'] = font_family_render

        # ==========================================
        # 多重断轴环境初始化
        # ==========================================
        visible_segments = []
        if use_broken_y:
            try:
                breaks = []
                for r in break_ranges.split(','):
                    parts = r.strip().split('-')
                    if len(parts) == 2: breaks.append((float(parts[0]), float(parts[1])))
                breaks.sort()
                stats_temp = dat.groupby(['PLOT_X', 'PLOT_FILL'])['PLOT_Y'].agg(['mean', 'std']).reset_index()
                raw_max = (stats_temp['mean'] + stats_temp['std']).max() * (1 + y_top_gap)
                if pd.isna(raw_max): raw_max = dat['PLOT_Y'].max() * (1 + y_top_gap)
                last_val = 0.0
                for b_start, b_end in breaks:
                    if b_start > last_val: visible_segments.append((last_val, b_start))
                    last_val = b_end
                if raw_max > last_val: visible_segments.append((last_val, raw_max))
                visible_segments.reverse()
            except: use_broken_y = False

        if use_broken_y and len(visible_segments) > 1:
            n_segs = len(visible_segments)
            try: ratios = [float(r) for r in segment_ratios.split(',')]
            except: ratios = [1] * n_segs
            if len(ratios) < n_segs: ratios.extend([1] * (n_segs - len(ratios)))
            fig = plt.figure(figsize=(num_w_render, num_h_render))
            gs = fig.add_gridspec(n_segs, 1, height_ratios=ratios[:n_segs], hspace=0.08)
            axes = []
            for i in range(n_segs):
                ax_tmp = fig.add_subplot(gs[i])
                ax_tmp.set_facecolor('white')
                ax_tmp.set_ylim(visible_segments[i])
                axes.append(ax_tmp)
            for ax_tmp in axes[:-1]: ax_tmp.tick_params(labelbottom=False, bottom=False)
            main_ax, title_ax = axes[-1], axes[0]
            big_ax = fig.add_subplot(gs[:])
            big_ax.set_facecolor('none')
            big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            for spine in big_ax.spines.values(): spine.set_visible(False)
            big_ax.set_in_layout(False)
            legend_ax = big_ax
        else:
            fig, main_ax = plt.subplots(figsize=(num_w_render, num_h_render))
            main_ax.set_facecolor('white')
            axes = [main_ax]; title_ax = main_ax; legend_ax = main_ax

        # ==========================================
        # 核心图形渲染 (支持断轴穿越)
        # ==========================================
        is_grouped = (work_mode == "grouped")
        current_width = dodge_width if is_grouped else bar_width
        
        for ax_iter in axes:
            for i, px in enumerate(safe_x_order):
                sub_d = dat[dat['PLOT_X'] == px]
                n_hues = len(hue_levels)
                for j, hue in enumerate(hue_levels):
                    grp_y = sub_d[sub_d['PLOT_FILL'] == hue]['PLOT_Y'].dropna().values
                    if len(grp_y) == 0: continue
                    mv = np.mean(grp_y)
                    sv = np.std(grp_y, ddof=1) if len(grp_y)>1 else 0
                    
                    off = (j - (n_hues-1)/2) * (current_width/n_hues) if is_grouped else 0
                    xc = i + off
                    bw = (current_width/n_hues)*(1 - inner_gap_render) if is_grouped else current_width
                    
                    b_col = "white" if bar_edge_col == 'colored_hollow' else color_map[hue]
                    e_col = color_map[hue] if bar_edge_col == 'colored_hollow' else ("black" if bar_edge_col=='black' else "none")
                    lwd = axis_lwd_render if e_col != "none" else 0
                    line_col = color_map[hue] if e_col == 'none' else e_col
                    box_edge = b_col if e_col == 'none' else e_col
                    p_col = color_map[hue] if point_edge_col == 'match' else point_edge_col

                    if plot_type == 'bar':
                        ax_iter.bar(xc, mv, width=bw, color=b_col, edgecolor=e_col, linewidth=lwd, zorder=2)
                        if sv > 0:
                            yerr_val = [[0], [sv]] if err_direction == 'up' else sv
                            ax_iter.errorbar(xc, mv, yerr=yerr_val, capsize=err_width_render*20, elinewidth=err_lwd_render, color='black', fmt='none', zorder=4)
                    elif plot_type == 'box':
                        bp = ax_iter.boxplot([grp_y], positions=[xc], widths=bw, patch_artist=True)
                        for patch in bp['boxes']:
                            patch.set_facecolor(b_col); patch.set_edgecolor(box_edge); patch.set_linewidth(lwd if lwd > 0 else 1.5)
                        for element in ['whiskers', 'caps', 'medians']:
                            for line in bp[element]: line.set_color(line_col); line.set_linewidth(lwd if lwd > 0 else 1.5)
                        for flier in bp['fliers']:
                            flier.set_markeredgecolor(line_col); flier.set_markerfacecolor(b_col)
                    elif plot_type == 'violin':
                        if len(grp_y) > 1:
                            vp = ax_iter.violinplot([grp_y], positions=[xc], widths=bw, showmeans=False, showmedians=False, showextrema=False)
                            for pc in vp['bodies']:
                                pc.set_facecolor(b_col); pc.set_edgecolor(box_edge); pc.set_linewidth(lwd if lwd > 0 else 1.5); pc.set_alpha(1.0)
                            q1, med, q3 = np.percentile(grp_y, [25, 50, 75])
                            ax_iter.plot([xc, xc], [q1, q3], color=line_col, lw=lwd*2 if lwd>0 else 3.0, zorder=3)
                            ax_iter.scatter(xc, med, color='white', edgecolor=line_col, s=20, zorder=4)
                    
                    if show_points:
                        jit = np.random.uniform(-jitter_render/2, jitter_render/2, len(grp_y))
                        ax_iter.scatter(xc + jit, grp_y, facecolors=point_fill_col_render, edgecolors=p_col, s=point_size_render, alpha=point_alpha_render, zorder=5)

        # ==========================================
        # 智能显著性统计本地映射
        # ==========================================
        def get_local_ax_params(val):
            if not use_broken_y: return axes[0], axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
            for ax_tmp, (vmin, vmax) in zip(axes, visible_segments):
                if vmin <= val <= vmax: return ax_tmp, vmax - vmin
            for ax_tmp, (vmin, vmax) in reversed(list(zip(axes, visible_segments))):
                if val <= vmax: return ax_tmp, vmax - vmin
            return axes[0], visible_segments[0][1] - visible_segments[0][0]

        if calc_type != "raw":
            stats_df = dat.groupby(['PLOT_X', 'PLOT_FILL'], observed=True)['PLOT_Y'].agg(['mean', 'std', 'max']).reset_index()
            stats_df['std'] = stats_df['std'].fillna(0)
            
            def get_bar_top(g):
                if plot_type == 'bar': return np.mean(g) + (np.std(g, ddof=1) if len(g)>1 else 0)
                else: return np.max(g)

            if is_grouped:
                for i, px in enumerate(safe_x_order):
                    sub_d = dat[dat['PLOT_X'] == px]
                    if sub_d['PLOT_FILL'].nunique() == 2:
                        h1, h2 = hue_levels[0], hue_levels[1]
                        g1, g2 = sub_d[sub_d['PLOT_FILL']==h1]['PLOT_Y'].dropna(), sub_d[sub_d['PLOT_FILL']==h2]['PLOT_Y'].dropna()
                        if len(g1) >= 2 and len(g2) >= 2:
                            p = perform_2group_test(g1.values, g2.values)
                            sig = ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") if sig_format == '准确P值' else ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
                            x1 = i + (0 - 0.5) * (current_width/2)
                            x2 = i + (1 - 0.5) * (current_width/2)
                            m1, m2 = get_bar_top(g1), get_bar_top(g2)
                            
                            t_ax, y_span = get_local_ax_params(max(m1, m2))
                            base_off = y_span * sig_y_offset
                            tip = y_span * sig_tip_len
                            sy = max(m1, m2) + base_off
                            
                            t_ax.plot([x1, x1, x2, x2], [sy, sy+tip, sy+tip, sy], color='black', lw=sig_lwd_render, clip_on=False)
                            t_ax.text((x1+x2)/2, sy+tip + (y_span*0.01), sig, ha='center', va='bottom', fontsize=sig_size_render, clip_on=False)
                                
                    elif sub_d['PLOT_FILL'].nunique() > 2:
                        lets = perform_multigroup_test(sub_d, 'PLOT_Y', 'PLOT_FILL')
                        for j, thue in enumerate(hue_levels):
                            if thue not in sub_d['PLOT_FILL'].values: continue
                            tx = i + (j - (n_hues - 1)/2) * (current_width/n_hues)
                            bar = stats_df[(stats_df['PLOT_X']==px) & (stats_df['PLOT_FILL']==thue)]
                            mx = bar['mean'].values[0] + bar['std'].values[0] if plot_type == 'bar' else bar['max'].values[0]
                            t_ax, y_span = get_local_ax_params(mx)
                            sy = mx + y_span * sig_y_offset
                            t_ax.text(tx, sy, lets.get(str(thue), ""), ha='center', va='bottom', fontsize=sig_size_render, clip_on=False)
            else:
                if len(safe_x_order) > 2:
                    lets = perform_multigroup_test(dat, 'PLOT_Y', 'PLOT_X')
                    for i, px in enumerate(safe_x_order):
                        bar = stats_df[stats_df['PLOT_X']==px]
                        mx = bar['mean'].values[0] + bar['std'].values[0] if plot_type == 'bar' else bar['max'].values[0]
                        t_ax, y_span = get_local_ax_params(mx)
                        sy = mx + y_span * sig_y_offset
                        t_ax.text(i, sy, lets.get(str(px), ""), ha='center', va='bottom', fontsize=sig_size_render, clip_on=False)
                elif len(safe_x_order) == 2:
                    g1, g2 = dat[dat['PLOT_X']==safe_x_order[0]]['PLOT_Y'].dropna(), dat[dat['PLOT_X']==safe_x_order[1]]['PLOT_Y'].dropna()
                    if len(g1) >= 2 and len(g2) >= 2:
                        p = perform_2group_test(g1.values, g2.values)
                        sig = ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") if sig_format == '准确P值' else ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
                        m1, m2 = get_bar_top(g1), get_bar_top(g2)
                        t_ax, y_span = get_local_ax_params(max(m1, m2))
                        base_off = y_span * sig_y_offset
                        tip = y_span * sig_tip_len
                        sy = max(m1, m2) + base_off
                        
                        t_ax.plot([0, 0, 1, 1], [sy, sy+tip, sy+tip, sy], color='black', lw=sig_lwd_render, clip_on=False)
                        t_ax.text(0.5, sy+tip + (y_span*0.01), sig, ha='center', va='bottom', fontsize=sig_size_render, clip_on=False)

        # ==========================================
        # 坐标轴清洗与标签对齐
        # ==========================================
        for ax_idx, ax_iter in enumerate(axes):
            if len(safe_x_order) > 0: ax_iter.set_xlim(-0.5, len(safe_x_order) - 0.5)
            
            if use_broken_y:
                n_ticks = 5 if ratios[ax_idx] >= 1.8 else 3
                if ax_idx == 0: ax_iter.yaxis.set_major_locator(ticker.MaxNLocator(nbins=n_ticks, prune='lower', min_n_ticks=2))
                elif ax_idx == len(axes)-1: ax_iter.yaxis.set_major_locator(ticker.MaxNLocator(nbins=n_ticks, prune='upper', min_n_ticks=2))
                else: ax_iter.yaxis.set_major_locator(ticker.MaxNLocator(nbins=n_ticks, prune='both', min_n_ticks=2))
            else:
                ax_iter.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            
            ax_iter.tick_params(axis='both', width=axis_lwd_render, labelsize=ytext_size_render)
            ax_iter.spines['left'].set_linewidth(axis_lwd_render)
            ax_iter.spines['bottom'].set_linewidth(axis_lwd_render)
            
            ax_iter.spines['top'].set_visible(False)
            ax_iter.spines['right'].set_visible(False)
            if use_broken_y and ax_idx < len(axes) - 1:
                ax_iter.spines['bottom'].set_visible(False)
                ax_iter.tick_params(bottom=False)

        final_x_labels = [unified_text_parser(x_rename_dict.get(orig_x, orig_x)) for orig_x in safe_x_order]
        main_ax.set_xticks(range(len(safe_x_order)))
        if x_angle > 0: main_ax.set_xticklabels(final_x_labels, rotation=x_angle, ha="right", rotation_mode="anchor", fontsize=xtext_size_render)
        else: main_ax.set_xticklabels(final_x_labels, rotation=0, ha="center", fontsize=xtext_size_render)

        parsed_title = unified_text_parser(plot_title_raw)
        title_ax.set_title(parsed_title if parsed_title else selected_gene, pad=20, fontsize=title_size_render, loc='center', fontstyle=title_style if title_style!='bold' else 'normal', fontweight='bold' if title_style=='bold' else 'normal')
        main_ax.set_xlabel(unified_text_parser(xlab_raw) if xlab_raw else "Groups", fontsize=xtitle_size_render, loc=xtitle_loc, labelpad=x_label_pad)
        legend_ax.set_ylabel(unified_text_parser(ylab_raw) if ylab_raw else "Relative Expression", fontsize=ytitle_size_render, loc=ytitle_loc, labelpad=y_label_pad)

        if not use_broken_y:
            main_ax.set_ylim(0, main_ax.get_ylim()[1] * (1 + y_top_gap))
            sns.despine(ax=main_ax)

        # ==========================================
        # 图例与最终定稿渲染
        # ==========================================
        if show_legend:
            try:
                from matplotlib.patches import Patch; from matplotlib.lines import Line2D
                if bar_edge_col == 'colored_hollow': legs = [Line2D([0],[0], marker='o', color='w', label=unified_text_parser(legend_rename_dict.get(h, h)), markerfacecolor='w', markeredgecolor=color_map[h], markersize=10, markeredgewidth=1.5) for h in hue_levels]
                else: legs = [Patch(facecolor=color_map[h], edgecolor='black' if bar_edge_col=='black' else 'none', label=unified_text_parser(legend_rename_dict.get(h, h))) for h in hue_levels]
                
                leg_kwargs = {"title": unified_text_parser(legend_title_raw) if legend_title_raw else "Condition", "frameon": False, "fontsize": legend_text_size_render, "title_fontsize": legend_text_size_render}
                loc_map = {'right': 'center left', 'top': 'lower center', 'bottom': 'upper center', 'left': 'center right', 'custom': 'center'}
                bbox_map = {'right': (1.02, 0.5), 'top': (0.5, 1.05), 'bottom': (0.5, -0.2), 'left': (-0.02, 0.5), 'custom': (leg_x, leg_y)}
                
                lg = legend_ax.legend(handles=legs, loc=loc_map[legend_pos], bbox_to_anchor=bbox_map[legend_pos], ncol=1 if legend_pos in ['right', 'left', 'custom'] else 4, **leg_kwargs)
                if lg: lg._legend_box.align = legend_title_align
            except Exception: pass

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        if use_broken_y: fig.subplots_adjust(left=0.18)

        # ✨ 终极斜杠防畸变渲染
        if use_broken_y:
            dx = break_slash_size
            fw, fh = fig.get_size_inches() 
            for i in range(len(axes)-1):
                pos_top = axes[i].get_position()
                pos_bot = axes[i+1].get_position()
                w_top, h_top = pos_top.width * fw, pos_top.height * fh
                w_bot, h_bot = pos_bot.width * fw, pos_bot.height * fh
                
                dy_top = (dx * w_top * 1.5) / h_top if h_top > 0 else dx
                dy_bot = (dx * w_bot * 1.5) / h_bot if h_bot > 0 else dx
                
                kwargs = dict(color='black', clip_on=False, lw=axis_lwd_render)
                axes[i].plot([-dx, +dx], [-dy_top, +dy_top], transform=axes[i].transAxes, **kwargs) 
                axes[i+1].plot([-dx, +dx], [1-dy_bot, 1+dy_bot], transform=axes[i+1].transAxes, **kwargs)

        plot_container = st.container(border=True)
        with plot_container:
            st.pyplot(fig)
            st.markdown("###### 导出图表")
            col_d1, col_d2, col_d3 = st.columns(3)
            
            buf_png = io.BytesIO()
            fig.savefig(buf_png, format="png", dpi=600, bbox_inches='tight')
            col_d1.download_button("PNG (600 DPI)", buf_png.getvalue(), f"{selected_gene}_Plot.png", "image/png", use_container_width=True)

            buf_pdf = io.BytesIO()
            fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
            col_d2.download_button("PDF (矢量)", buf_pdf.getvalue(), f"{selected_gene}_Plot.pdf", "application/pdf", use_container_width=True)

            buf_svg = io.BytesIO()
            fig.savefig(buf_svg, format="svg", bbox_inches='tight')
            col_d3.download_button("SVG (矢量)", buf_svg.getvalue(), f"{selected_gene}_Plot.svg", "image/svg+xml", use_container_width=True)

    st.markdown("---")
    st.markdown("### 数据明细表")
    dat_show = dat.copy()
    def clean_text(text): return str(text).replace('**', '').replace('*', '').replace('\\n', ' ')
    dat_show['PLOT_X'] = dat_show['PLOT_X'].map(lambda x: clean_text(x_rename_dict.get(str(x), str(x))))
    dat_show['PLOT_FILL'] = dat_show['PLOT_FILL'].map(lambda x: clean_text(legend_rename_dict.get(str(x), str(x))))
    st.dataframe(dat_show, use_container_width=True)