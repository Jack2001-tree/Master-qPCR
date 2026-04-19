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
from streamlit_quill import st_quill

# ==========================================
# 页面设置
# ==========================================
st.set_page_config(page_title="qPCR 数据分析工作台", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 注入 Material You 与 Sticky (吸顶) CSS
# ==========================================
st.markdown("""
    <style>
        /* Material You 全局背景与字体色 */
        .stApp { background-color: #F8FAFC !important; color: #1e293b !important; }
        
        /* 侧边栏 Material 风格化 */
        [data-testid="stSidebar"] { 
            background-color: #EEF2F6 !important; 
            border-right: none !important; 
        }
        
        /* 原生输入框与选择框的 Material You 大圆角质感 */
        .stSelectbox div[data-baseweb="select"], .stTextInput input, .stNumberInput input { 
            background-color: #ffffff !important; 
            border-radius: 12px !important; 
            border: 1px solid #E2E8F0 !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        }

        /* Material You 风格的 Tabs */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 8px; 
            border-bottom: none; 
            background-color: #EEF2F6;
            padding: 8px;
            border-radius: 16px;
        }
        .stTabs [data-baseweb="tab"] { 
            border-radius: 12px; 
            padding: 8px 16px; 
            background-color: transparent !important; 
            color: #64748b !important; 
            border: none; 
        }
        .stTabs [aria-selected="true"] { 
            background-color: #ffffff !important; 
            color: #0F172A !important; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            font-weight: 600; 
        }

        /* 👑 核心逻辑：右侧预览区完美悬浮 (Sticky)，解除等高拉伸 */
        [data-testid="column"]:nth-of-type(2) {
            position: sticky !important;
            top: 4rem !important; 
            align-self: flex-start !important; /* 关键核心：解除 Streamlit 列等高拉伸 */
            z-index: 10;
        }

        /* 预览区卡片化包装 */
        [data-testid="column"]:nth-of-type(2) > div {
            background-color: #ffffff;
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.08);
            border: 1px solid #F1F5F9;
        }

        /* Material 药丸形按钮 */
        .stButton button { 
            border-radius: 24px !important; 
            font-weight: 600; 
            border: none;
            transition: all 0.2s ease;
        }
        .stButton button[kind="primary"] {
            background-color: #0A56D0 !important;
            color: white !important;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* 标题与辅助文本微调 */
        h3 { padding-bottom: 0.5rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem; color: #1e293b; font-size: 1.25rem;}
        .toolbar-text { font-size: 0.85rem; color: #64748b; margin-bottom: 4px; display: block;}
        
        /* 强制压缩 Quill 富文本编辑器的 iframe，并注入 Material You 美学 */
        [data-testid="stElementContainer"] iframe {
            height: 120px !important; /* 强制压缩总高度 */
            border-radius: 16px !important; /* 统一的 MY 大圆角 */
            border: 1px solid #E2E8F0 !important;
            background-color: #ffffff !important;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
            transition: all 0.3s ease;
        }
        [data-testid="stElementContainer"] iframe:hover {
            border-color: #CBD5E1 !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 0. 显著性字母生成算法 (Tukey HSD)
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

# ==========================================
# 0.5 全能富文本解析引擎 (增强防御)
# ==========================================
def unified_text_parser(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip(): 
        return ""
    if raw_text in ["<p><br></p>", "<p></p>"]: 
        return ""

    text = raw_text
    def _math_replace(match, tag):
        content = match.group(1)
        content = content.replace('%', r'\%').replace('$', r'\$')
        content = content.replace(' ', r'\ ')
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

# ==========================================
# 1. 核心计算引擎
# ==========================================
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
                current_scenario_data.append(res)
                current_scenario_score += res['Bio_RelExp'].std()
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
                                best_age_df = pd.DataFrame({
                                    'Gene': [target_gene]*k_reps, 'Age': [grp]*k_reps, 
                                    'Bio_Rep': p_ids_list[i], 'Bio_RelExp': rel_exps,
                                    'Pairing': [f"{p_ids_list[i][j]} vs {d0_ids[j]}" for j in range(k_reps)]
                                })
                    current_scenario_data.append(best_age_df)
                    current_scenario_score += best_age_sd
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

# ==========================================
# 2. 数据解析
# ==========================================
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
        if not all([col_s, col_t, col_c]): raise ValueError("数据格式不匹配，缺少必要的 Sample, Target 或 CT 列。")
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

# ==========================================
# UI 布局: 数据导入与模式配置 (侧边栏)
# ==========================================
with st.sidebar:
    st.markdown("### 数据源")
    uploaded_file = st.file_uploader("上传数据文件 (CSV/Excel)", type=["csv", "xlsx"])
    calc_type = st.radio("数据处理模式", ["opt", "智能寻优", "strict", "严格配对", "raw", "原始数据"] if False else ["opt", "strict", "raw"], format_func=lambda x: {"opt": "智能寻优", "strict": "严格配对", "raw": "原始数据"}[x])
    
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

    # ================= Pane 1: 参数配置区 =================
    with col_source:
        
        # 👑 样式独立层：置于最上方级全局控制
        st.markdown("### 🌟 全局样式预设")
        preset_style = st.selectbox("一键应用发表级排版", ["自定义", "样式1"], label_visibility="collapsed")
        
        if preset_style == "样式1":
            st.info("💡 已启用「样式1」预设：自动应用粉彩配色、无边框柱体及纯黑实心散点，您仍可自由调节下方各项尺寸与比例。")
            
        st.markdown("---")
        
        st.markdown("### 图表参数")
        selected_gene = st.selectbox("分析基因", list(results.keys()))
        
        st.markdown("<span class='toolbar-text'><b>图表主标题</b> (回车换行，划选文字加粗/斜体)</span>", unsafe_allow_html=True)
        plot_title_raw = st_quill(placeholder="请输入图表标题 (留空默认基因名)", html=True, toolbar=[["bold", "italic"]], key="quill_title")
        
        # 👑 独立出画布与坐标轴标签页
        tab_geom, tab_theme, tab_canvas, tab_stats = st.tabs(["图形设置", "文本与排版", "坐标轴与画布", "显著性与图例"])
        
        with tab_geom:
            plot_type = st.selectbox("图形类别", ["bar", "box", "violin"], format_func=lambda x: {"bar": "柱状图", "box": "箱线图", "violin": "小提琴图"}[x])
            
            st.markdown("**色彩方案**")
            use_custom_cols = st.checkbox("自定义 HEX 颜色", False)
            if use_custom_cols: custom_cols = st.text_input("颜色列表 (逗号分隔)", "#E64B35, #4DBBD5, #00A087, #3C5488")
            else: palette_choice = st.selectbox("预设调色板", ["npg", "aaas", "lancet", "nejm", "jama", "jco", "d3", "igv", "cell"])
            
            st.markdown("**几何尺寸**")
            col_g1, col_g2 = st.columns(2)
            bar_edge_col_ui = col_g1.selectbox("柱/箱体边框", ["black", "none"], format_func=lambda x: "黑色" if x == "black" else "无边框 (透明)")
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
            
        with tab_canvas:
            st.markdown("**画布尺寸**")
            col_c1, col_c2 = st.columns(2)
            num_w = col_c1.number_input("画布宽 (英寸)", value=6.0, step=0.5)
            num_h = col_c2.number_input("画布高 (英寸)", value=5.0, step=0.5)
            
            st.markdown("**坐标轴细节**")
            x_angle = st.slider("X轴标签旋转角度", 0, 90, 0, step=15)
            axis_lwd = st.slider("坐标轴线粗细", 0.5, 4.0, 1.5)
            y_top_gap = st.slider("Y轴顶部留白比例", 0.05, 0.5, 0.2)

        with tab_stats:
            st.markdown("**检验标识**")
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

    # ================= Pane 2: 图表渲染区 (右侧悬浮) =================
    with col_viewer:
        st.markdown("### 图表预览")
        
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
        
        # 👑 样式引擎覆盖系统 (Style Override Engine)
        palettes = {"npg": ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2","#DC0000","#7E6148"], "aaas": ["#3B4992","#EE0000","#008B45","#631879","#008280","#BB0021","#5F559B","#A20056","#808180"], "lancet": ["#00468B","#ED0000","#42B540","#0099B4","#925E9F","#FDAF91","#AD002A","#ADB6B6","#1B1919"], "nejm": ["#BC3C29","#0072B5","#E18727","#20854E","#7876B1","#6F99AD","#FFDC91","#EE4C97"], "jama": ["#374E55","#DF8F44","#00A1D5","#B24745","#79AF97","#6A6599","#80796B"], "jco": ["#0073C2","#EFC000","#868686","#CD534C","#7AA6DC","#003C67","#8F7700","#3B3B3B"], "d3": ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F"], "igv": ["#5050FF","#FF5050","#00BD00","#FFB900","#00D6FF","#B973FF","#006600","#6600CC"], "cell": ["#C2C1D5", "#F6A6A5", "#9ED2C4", "#8BB4D5", "#D8C4E0", "#EED7A1"]}
        
        if preset_style == "样式1":
            final_cols = palettes["cell"]
            # 视觉组件覆写（保持特有风格）
            bar_edge_col = "none"
            point_fill_col_render = "black"
            point_edge_col = "none"
            inner_gap_render = 0.15 if work_mode == "grouped" else inner_gap
            err_direction = "both"
            
            # 放开限制，将尺寸变量重新绑定回 UI 的当前输入值
            point_size_render = point_size
            point_alpha_render = point_alpha
            jitter_render = jitter_width
            font_family_render = font_family
            title_size_render = title_size
            xtitle_size_render = xtitle_size
            ytitle_size_render = ytitle_size
            xtext_size_render = xtext_size
            ytext_size_render = ytext_size
            axis_lwd_render = axis_lwd
            err_lwd_render = err_lwd
            err_width_render = err_width
            sig_lwd_render = sig_lwd
            sig_size_render = sig_size
            legend_text_size_render = legend_text_size
            num_h_render = num_h
            num_w_render = num_w
        else:
            final_cols = [c.strip() for c in custom_cols.split(',')] if use_custom_cols else palettes[palette_choice]
            # 自定义模式直通
            bar_edge_col = bar_edge_col_ui
            point_fill_col_render = "none" if point_fill_col.lower() == "none" else point_fill_col
            point_edge_col = "black" if point_fill_col_render != "none" else "black"
            inner_gap_render = inner_gap
            err_direction = err_dir_ui
            point_size_render = point_size
            point_alpha_render = point_alpha
            jitter_render = jitter_width
            font_family_render = font_family
            title_size_render = title_size
            xtitle_size_render = xtitle_size
            ytitle_size_render = ytitle_size
            xtext_size_render = xtext_size
            ytext_size_render = ytext_size
            axis_lwd_render = axis_lwd
            err_lwd_render = err_lwd
            err_width_render = err_width
            sig_lwd_render = sig_lwd
            sig_size_render = sig_size
            legend_text_size_render = legend_text_size
            num_w_render = num_w
            num_h_render = num_h
            
        color_map = {hue: final_cols[idx % len(final_cols)] for idx, hue in enumerate(hue_levels)}

        leg_kwargs = {"title": unified_text_parser(legend_title_raw) if legend_title_raw else "Condition", "frameon": False, "fontsize": legend_text_size_render, "title_fontsize": legend_text_size_render}
        if legend_pos == "right": leg_kwargs.update({"loc": "center left", "bbox_to_anchor": (1.02, 0.5)})
        elif legend_pos == "left": leg_kwargs.update({"loc": "center right", "bbox_to_anchor": (-0.02, 0.5)})
        elif legend_pos == "top": leg_kwargs.update({"loc": "lower center", "bbox_to_anchor": (0.5, 1.05), "ncol": 4})
        elif legend_pos == "bottom": leg_kwargs.update({"loc": "upper center", "bbox_to_anchor": (0.5, -0.2), "ncol": 4})
        elif legend_pos == "custom": leg_kwargs.update({"loc": "center", "bbox_to_anchor": (leg_x, leg_y)})

        plt.rcParams['font.family'] = font_family_render
        fig, ax = plt.subplots(figsize=(num_w_render, num_h_render))
        
        is_grouped = (work_mode == "grouped")
        current_width = dodge_width if is_grouped else bar_width
        
        # 绘图层
        if plot_type == "bar":
            for i, px in enumerate(safe_x_order):
                sub_d = dat[dat['PLOT_X'] == px]
                n_hues = len(hue_levels)
                for j, hue in enumerate(hue_levels):
                    group_data = sub_d[sub_d['PLOT_FILL'] == hue]['PLOT_Y']
                    if len(group_data) > 0:
                        mean_val, std_val = group_data.mean(), group_data.std() if len(group_data) > 1 else np.nan
                        x_center = i + (j - (n_hues - 1) / 2) * (current_width / n_hues) if is_grouped else i
                        actual_bar_width = (current_width / n_hues) * (1 - inner_gap_render) if is_grouped else current_width
                        
                        ax.bar(x_center, mean_val, width=actual_bar_width, color=color_map[hue], edgecolor=bar_edge_col, linewidth=0 if bar_edge_col=='none' else axis_lwd_render, zorder=2)
                        
                        if pd.notna(std_val) and std_val > 0:
                            yerr_val = std_val if err_direction == "both" else [[0], [std_val]]
                            ax.errorbar(x_center, mean_val, yerr=yerr_val, capsize=err_width_render*20, elinewidth=err_lwd_render, color='black', fmt='none', zorder=4)
                            
                        if show_points:
                            jitter = np.random.uniform(-jitter_render/2, jitter_render/2, size=len(group_data))
                            ax.scatter(x_center + jitter, group_data, facecolors=point_fill_col_render, edgecolors=point_edge_col, linewidth=1.2, s=point_size_render, alpha=point_alpha_render, zorder=5)
            if show_legend:
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=color_map[hue], edgecolor='black' if bar_edge_col=='black' else 'none', label=unified_text_parser(legend_rename_dict.get(hue, hue))) for hue in hue_levels]
                leg = ax.legend(handles=legend_elements, **leg_kwargs)
                if leg: leg._legend_box.align = legend_title_align

        else:
            if plot_type == "box": sns.boxplot(data=dat, x='PLOT_X', y='PLOT_Y', hue='PLOT_FILL', order=safe_x_order, hue_order=hue_levels, dodge=is_grouped, width=current_width, boxprops={'edgecolor': bar_edge_col, 'linewidth': 0 if bar_edge_col=='none' else axis_lwd_render}, ax=ax, palette=final_cols)
            else: sns.violinplot(data=dat, x='PLOT_X', y='PLOT_Y', hue='PLOT_FILL', order=safe_x_order, hue_order=hue_levels, dodge=is_grouped, width=current_width, inner="quartile", ax=ax, palette=final_cols)
            if show_points:
                if is_grouped:
                    n_hues = len(hue_levels)
                    for i, px in enumerate(safe_x_order):
                        for j, treat_hue in enumerate(hue_levels):
                            group_data = dat[(dat['PLOT_X'] == px) & (dat['PLOT_FILL'] == treat_hue)]['PLOT_Y']
                            if len(group_data) > 0:
                                treat_x_pos = i + (j - (n_hues - 1) / 2) * (current_width / n_hues)
                                jitter = np.random.uniform(-jitter_render/2, jitter_render/2, size=len(group_data))
                                ax.scatter(treat_x_pos + jitter, group_data, facecolors=point_fill_col_render, edgecolors=point_edge_col, linewidth=1.2, s=point_size_render, alpha=point_alpha_render, zorder=5)
                else: sns.stripplot(data=dat, x='PLOT_X', y='PLOT_Y', hue='PLOT_FILL', order=safe_x_order, hue_order=hue_levels, dodge=False, edgecolor=point_edge_col, linewidth=0 if point_edge_col=='none' else 1.2, size=point_size_render/5, ax=ax, palette=[point_fill_col_render]*len(hue_levels), alpha=point_alpha_render, jitter=jitter_render)
            handles, labels = ax.get_legend_handles_labels()
            if show_legend and handles: 
                new_labels = [unified_text_parser(legend_rename_dict.get(l, l)) for l in labels[:len(hue_levels)]]
                leg = ax.legend(handles[:len(hue_levels)], new_labels, **leg_kwargs)
                if leg: leg._legend_box.align = legend_title_align
            elif ax.get_legend(): ax.get_legend().remove()

        # 防重叠核心层
        if calc_type != "raw":
            n_hues = len(hue_levels)
            stats_df = dat.groupby(['PLOT_X', 'PLOT_FILL'], observed=True)['PLOT_Y'].agg(['mean', 'std', 'max']).reset_index()
            max_y_overall = stats_df['mean'].max() + stats_df['std'].max() if plot_type=='bar' else stats_df['max'].max()
            if pd.isna(max_y_overall): max_y_overall = dat['PLOT_Y'].max()
            base_offset = max_y_overall * sig_y_offset
            tip = max_y_overall * sig_tip_len
            
            if is_grouped:
                for i, px in enumerate(safe_x_order):
                    sub_d = dat[dat['PLOT_X'] == px]
                    if sub_d['PLOT_FILL'].nunique() > 1:
                        if sub_d['PLOT_FILL'].nunique() > 2: 
                            try:
                                tukey = pairwise_tukeyhsd(endog=sub_d['PLOT_Y'], groups=sub_d['PLOT_FILL'], alpha=0.05)
                                means_s = sub_d.groupby('PLOT_FILL', observed=True)['PLOT_Y'].mean().dropna()
                                letters_dict = get_tukey_letters(tukey, means_s)
                                for j, treat_hue in enumerate(hue_levels):
                                    if treat_hue not in sub_d['PLOT_FILL'].values: continue
                                    treat_x_pos = i + (j - (n_hues - 1) / 2) * (current_width / n_hues)
                                    bar_stat = stats_df[(stats_df['PLOT_X']==px) & (stats_df['PLOT_FILL']==treat_hue)]
                                    safe_top = bar_stat['mean'].values[0] + (bar_stat['std'].values[0] if pd.notna(bar_stat['std'].values[0]) else 0) if plot_type == 'bar' else bar_stat['max'].values[0]
                                    ax.text(treat_x_pos, safe_top + base_offset, letters_dict.get(str(treat_hue), ""), ha='center', va='bottom', color='black', fontsize=sig_size_render)
                            except Exception: pass
                        else:
                            try:
                                ref_hue, treat_hue = hue_levels[0], hue_levels[1]
                                ref_x_pos, treat_x_pos = i + (0 - (n_hues - 1) / 2) * (current_width / n_hues), i + (1 - (n_hues - 1) / 2) * (current_width / n_hues)
                                
                                bar_stat_ref = stats_df[(stats_df['PLOT_X']==px) & (stats_df['PLOT_FILL']==ref_hue)]
                                bar_stat_treat = stats_df[(stats_df['PLOT_X']==px) & (stats_df['PLOT_FILL']==treat_hue)]
                                max_ref = bar_stat_ref['mean'].values[0] + (bar_stat_ref['std'].values[0] if pd.notna(bar_stat_ref['std'].values[0]) else 0) if plot_type == 'bar' else bar_stat_ref['max'].values[0]
                                max_treat = bar_stat_treat['mean'].values[0] + (bar_stat_treat['std'].values[0] if pd.notna(bar_stat_treat['std'].values[0]) else 0) if plot_type == 'bar' else bar_stat_treat['max'].values[0]
                                
                                local_max = max(max_ref, max_treat)
                                stack_y = local_max + base_offset
                                
                                g1, g2 = sub_d[sub_d['PLOT_FILL'] == ref_hue]['PLOT_Y'], sub_d[sub_d['PLOT_FILL'] == treat_hue]['PLOT_Y']
                                if len(g1)>1 and len(g2)>1:
                                    stat, p_val = ttest_ind(g1, g2)
                                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                                    
                                    ax.plot([ref_x_pos, ref_x_pos, treat_x_pos, treat_x_pos], [stack_y, stack_y+tip, stack_y+tip, stack_y], lw=sig_lwd_render, c='black')
                                    ax.text((ref_x_pos+treat_x_pos)/2, stack_y+tip + (max_y_overall*0.01), sig, ha='center', va='bottom', color='black', fontsize=sig_size_render)
                            except Exception: pass
            else: 
                if len(safe_x_order) > 2:
                    try:
                        tukey = pairwise_tukeyhsd(endog=dat['PLOT_Y'], groups=dat['PLOT_X'], alpha=0.05)
                        means_s = dat.groupby('PLOT_X', observed=True)['PLOT_Y'].mean().dropna()
                        letters_dict = get_tukey_letters(tukey, means_s)
                        for i, px in enumerate(safe_x_order):
                            bar_stat = stats_df[stats_df['PLOT_X']==px]
                            safe_top = bar_stat['mean'].values[0] + (bar_stat['std'].values[0] if pd.notna(bar_stat['std'].values[0]) else 0) if plot_type == 'bar' else bar_stat['max'].values[0]
                            ax.text(i, safe_top + base_offset, letters_dict.get(str(px), ""), ha='center', va='bottom', color='black', fontsize=sig_size_render)
                    except Exception: pass
                elif len(safe_x_order) == 2:
                    try:
                        g1, g2 = dat[dat['PLOT_X'] == safe_x_order[0]]['PLOT_Y'], dat[dat['PLOT_X'] == safe_x_order[1]]['PLOT_Y']
                        bar_stat_ref = stats_df[stats_df['PLOT_X']==safe_x_order[0]]
                        bar_stat_treat = stats_df[stats_df['PLOT_X']==safe_x_order[1]]
                        max_ref = bar_stat_ref['mean'].values[0] + (bar_stat_ref['std'].values[0] if pd.notna(bar_stat_ref['std'].values[0]) else 0) if plot_type == 'bar' else bar_stat_ref['max'].values[0]
                        max_treat = bar_stat_treat['mean'].values[0] + (bar_stat_treat['std'].values[0] if pd.notna(bar_stat_treat['std'].values[0]) else 0) if plot_type == 'bar' else bar_stat_treat['max'].values[0]
                        local_max = max(max_ref, max_treat)
                        
                        if len(g1)>1 and len(g2)>1:
                            stat, p_val = ttest_ind(g1, g2)
                            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                            stack_y = local_max + base_offset
                            ax.plot([0, 0, 1, 1], [stack_y, stack_y+tip, stack_y+tip, stack_y], lw=sig_lwd_render, c='black')
                            ax.text(0.5, stack_y+tip + (max_y_overall*0.01), sig, ha='center', va='bottom', color='black', fontsize=sig_size_render)
                    except Exception: pass

        # 图表坐标与边框美化 
        parsed_title = unified_text_parser(plot_title_raw)
        final_title = parsed_title if parsed_title else selected_gene
        ax.set_title(final_title, pad=20, fontsize=title_size_render, loc='center', fontstyle=title_style if title_style!='bold' else 'normal', fontweight='bold' if title_style=='bold' else 'normal')
        
        ax.set_xlabel(unified_text_parser(xlab_raw) if xlab_raw else "Groups", fontsize=xtitle_size_render)
        ax.set_ylabel(unified_text_parser(ylab_raw) if ylab_raw else "Relative Expression", fontsize=ytitle_size_render)
        
        final_x_labels = [unified_text_parser(x_rename_dict.get(orig_x, orig_x)) for orig_x in safe_x_order]
        ax.set_xticks(range(len(safe_x_order)))
        if x_angle > 0: ax.set_xticklabels(final_x_labels, rotation=x_angle, ha="right", rotation_mode="anchor", fontsize=xtext_size_render)
        else: ax.set_xticklabels(final_x_labels, rotation=0, ha="center", fontsize=xtext_size_render)
            
        ax.tick_params(axis='y', labelsize=ytext_size_render)
        sns.despine()
        ax.spines['bottom'].set_linewidth(axis_lwd_render)
        ax.spines['left'].set_linewidth(axis_lwd_render)
        ax.tick_params(width=axis_lwd_render)
        y_max_plot = ax.get_ylim()[1]
        ax.set_ylim(0, y_max_plot * (1 + y_top_gap))

        # ================= 导出模块 =================
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

    # ================= Pane 4: 数据明细 =================
    st.markdown("---")
    st.markdown("### 数据明细表")
    dat_show = dat.copy()
    def clean_text(text): return str(text).replace('**', '').replace('*', '').replace('\\n', ' ')
    dat_show['PLOT_X'] = dat_show['PLOT_X'].map(lambda x: clean_text(x_rename_dict.get(str(x), str(x))))
    dat_show['PLOT_FILL'] = dat_show['PLOT_FILL'].map(lambda x: clean_text(legend_rename_dict.get(str(x), str(x))))
    st.dataframe(dat_show, use_container_width=True)