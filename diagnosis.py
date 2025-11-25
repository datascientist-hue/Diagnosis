import streamlit as st
import pandas as pd
import plotly.express as px
from ftplib import FTP
import io
import os
from functools import reduce

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Intelligent DB Growth Engine")

# --- SECURE FTP DATA LOADING FUNCTION ---
@st.cache_data
def load_data_from_ftp(_ftp_creds):
    """
    Securely loads primary and category data from an FTP server using Streamlit secrets,
    merges them, and preprocesses.
    """
    try:
        def download_file_from_ftp(ftp, full_path):
            directory = os.path.dirname(full_path)
            filename = os.path.basename(full_path)
            ftp.cwd(directory)
            in_memory_file = io.BytesIO()
            ftp.retrbinary(f"RETR {filename}", in_memory_file.write)
            in_memory_file.seek(0)
            return in_memory_file

        ftp = FTP(_ftp_creds['host'])
        ftp.login(user=_ftp_creds['user'], passwd=_ftp_creds['password'])
        st.success("✅ Successfully connected to FTP server.")

        primary_file_obj = download_file_from_ftp(ftp, _ftp_creds['primary_path'])
        ftp.cwd("/")
        ctg_file_obj = download_file_from_ftp(ftp, _ftp_creds['category_path'])
        
        ftp.quit()

        df_primary = pd.read_parquet(primary_file_obj)
        df_ctg_map = pd.read_parquet(ctg_file_obj)
        
        df = pd.merge(df_primary, df_ctg_map, on='ProductCategory', how='left')
        df['ProductCategory_Updated'].fillna(df['ProductCategory'], inplace=True)
        df.drop('ProductCategory', axis=1, inplace=True)
        df.rename(columns={'ProductCategory_Updated': 'ProductCategory'}, inplace=True)
        
        df['Inv Date'] = pd.to_datetime(df['Inv Date'], dayfirst=True, errors='coerce')
        key_cols = ['Cust Code', 'Cust Name', 'JCPeriod', 'WeekNum', 'DSM', 'ASM', 'CustomerClass', 'ProductCategory', 'Inv Num']
        df.dropna(subset=key_cols, inplace=True)
        
        df['JCPeriod'] = df['JCPeriod'].astype(int)
        df['WeekNum'] = df['WeekNum'].astype(int)
        df['Qty in Ltrs/Kgs'] = pd.to_numeric(df['Qty in Ltrs/Kgs'], errors='coerce').fillna(0)
        df['Volume in Tonnes'] = df['Qty in Ltrs/Kgs'] / 1000
        df['Net Value'] = pd.to_numeric(df['Net Value'], errors='coerce').fillna(0)
        df['Fin Year'] = df['Fin Year'].astype(str)
        return df

    except Exception as e:
        st.error(f"Error connecting to FTP or loading data: {e}")
        st.error("Please check your credentials and file paths in the .streamlit/secrets.toml file.")
        return None

# --- MAIN APP ---
df_original = load_data_from_ftp(st.secrets["ftp"])

if df_original is not None:
    st.title("💡 Intelligent Distributor Growth Engine")
    st.markdown("_From Data to Decisions: Identify Gaps, Analyze Investment Patterns, and Unlock Potential._")
    
    st.sidebar.header("Control Panel")
    
    available_fin_years = sorted(df_original['Fin Year'].unique(), reverse=True)
    selected_fin_year = st.sidebar.selectbox('Select Financial Year', options=available_fin_years)
    df_fy_filtered = df_original[df_original['Fin Year'] == selected_fin_year]
    available_jc_periods = sorted(df_fy_filtered['JCPeriod'].unique())
    selected_jc_period = st.sidebar.multiselect('Filter by JC Period(s)', options=available_jc_periods, default=[])
    
    df_pre_filter = df_original.copy()
    if selected_fin_year:
        df_pre_filter = df_pre_filter[df_pre_filter['Fin Year'] == selected_fin_year]

    available_asms = sorted(df_pre_filter['ASM'].unique())
    selected_asm = st.sidebar.multiselect('Filter by ASM', options=available_asms, default=[])
    if selected_asm:
        df_pre_filter = df_pre_filter[df_pre_filter['ASM'].isin(selected_asm)]

    available_dsms = sorted(df_pre_filter['DSM'].unique())
    selected_dsm = st.sidebar.multiselect('Filter by DSM', options=available_dsms, default=[])
    if selected_dsm:
        df_pre_filter = df_pre_filter[df_pre_filter['DSM'].isin(selected_dsm)]

    available_cust_class = sorted(df_pre_filter['CustomerClass'].unique())
    selected_cust_class = st.sidebar.multiselect('Filter by Customer Class', options=available_cust_class, default=[])
    if selected_cust_class:
        df_pre_filter = df_pre_filter[df_pre_filter['CustomerClass'].isin(selected_cust_class)]

    available_distributors = sorted(df_pre_filter['Cust Name'].unique())
    selected_distributor = st.sidebar.multiselect('Filter by Distributor', options=available_distributors, default=[])
    if selected_distributor:
        df_pre_filter = df_pre_filter[df_pre_filter['Cust Name'].isin(selected_distributor)]

    df_universe_base = df_pre_filter
    db_universe_dynamic = set(df_universe_base['Cust Code'].unique())
    
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive", "📉 Gaps & Frequency", "🧠 Investment Analysis", "✨ DB Evolution & Potential", "📦 Product Deep Dive"])
    
    with tab0:
        if not selected_jc_period:
            st.warning("Please select at least one JC Period from the sidebar for analysis.")
        else:
            # --- HELPER FUNCTIONS ---
            def get_metrics(df):
                if df.empty: return {'value': 0, 'volume': 0, 'db_count': 0}
                total_value = df['Net Value'].sum()
                total_volume = df['Volume in Tonnes'].sum()
                num_dbs = df['Cust Code'].nunique()
                return {'value': total_value, 'volume': total_volume, 'db_count': num_dbs}

            # --- DATA PREPARATION FOR ALL SECTIONS ---
            yoy_years = sorted(df_original['Fin Year'].unique(), reverse=True)[:3]
            max_jc = max(selected_jc_period)
            ytd_jcs = list(range(1, max_jc + 1))
            
            # Base DataFrame for YoY calculations, respecting all sidebar filters
            df_yoy_base = df_original.copy()
            if selected_asm: df_yoy_base = df_yoy_base[df_yoy_base['ASM'].isin(selected_asm)]
            if selected_dsm: df_yoy_base = df_yoy_base[df_yoy_base['DSM'].isin(selected_dsm)]
            if selected_cust_class: df_yoy_base = df_yoy_base[df_yoy_base['CustomerClass'].isin(selected_cust_class)]
            if selected_distributor: df_yoy_base = df_yoy_base[df_yoy_base['Cust Name'].isin(selected_distributor)]
            
            # --- LAYOUT DEFINITION (MAIN AREA + BILLBOARD) ---
            main_col, billboard_col = st.columns([2.5, 1])

            with billboard_col:
                st.markdown("<h3 style='text-align: center;'>Performance Billboard</h3>", unsafe_allow_html=True)
                ytd_metrics_data_billboard = []
                for year in yoy_years:
                    df_ytd = df_yoy_base[(df_yoy_base['Fin Year'] == year) & (df_yoy_base['JCPeriod'].isin(ytd_jcs))]
                    metrics = get_metrics(df_ytd)
                    metrics['Financial Year'] = year
                    ytd_metrics_data_billboard.append(metrics)
                
                ytd_df_billboard = pd.DataFrame(ytd_metrics_data_billboard).sort_values('Financial Year', ascending=False).reset_index(drop=True)

                if len(ytd_df_billboard) > 1:
                    current_vol = ytd_df_billboard.iloc[0]['volume']
                    prev_vol = ytd_df_billboard.iloc[1]['volume']
                    
                    if prev_vol > 0:
                        ytd_growth_percentage = ((current_vol - prev_vol) / prev_vol) * 100
                        delta_color = "green" if ytd_growth_percentage >= 0 else "red"
                        arrow = "▲" if ytd_growth_percentage >= 0 else "▼"
                        growth_text = f"{ytd_growth_percentage:.1f}%"
                    else:
                        delta_color = "green"
                        arrow = "▲"
                        growth_text = "New"

                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center;">
                        <p style="font-size: 16px; margin-bottom: 5px; font-weight: bold;">YTD Volume Growth</p>
                        <p style="font-size: 12px; margin-bottom: 5px;">(JCs 1-{max_jc} vs. Same Period LY)</p>
                        <p style="font-size: 48px; font-weight: bold; color: {delta_color}; margin: 0;">
                            {arrow} {growth_text}
                        </p>
                        <p style="font-size: 16px; margin-top: 10px;">
                            CY: {current_vol:,.2f} T | PY: {prev_vol:,.2f} T
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Not enough data for YoY comparison billboard.")

            with main_col:
                st.header("📊 Descriptive Performance Analysis")
                # =========================================================================
                # 1. INTRA-YEAR MOMENTUM
                # =========================================================================
                st.subheader(f"1. Intra-Year Momentum ({selected_fin_year})")
                min_jc_selected = min(selected_jc_period)
                prior_jcs = list(range(1, min_jc_selected))

                if prior_jcs:
                    kpi3_col1, kpi3_col2 = st.columns(2)
                    df_selected_period = df_universe_base[(df_universe_base['Fin Year'] == selected_fin_year) & (df_universe_base['JCPeriod'].isin(selected_jc_period))]
                    metrics_selected = get_metrics(df_selected_period)
                    df_prior_cumulative = df_universe_base[(df_universe_base['Fin Year'] == selected_fin_year) & (df_universe_base['JCPeriod'].isin(prior_jcs))]
                    metrics_prior_total = get_metrics(df_prior_cumulative)
                    num_prior_jcs = len(prior_jcs)
                    metrics_avg_prior = {
                        'volume': metrics_prior_total['volume'] / num_prior_jcs if num_prior_jcs > 0 else 0,
                        'value': metrics_prior_total['value'] / num_prior_jcs if num_prior_jcs > 0 else 0,
                        'db_count': metrics_prior_total['db_count'] / num_prior_jcs if num_prior_jcs > 0 else 0
                    }
                    def calculate_delta_for_kpi3(current, average_previous):
                         if average_previous > 0: return f"{((current - average_previous) / average_previous * 100):.1f}%"
                         return "New"
                    with kpi3_col1:
                        st.markdown(f"**Selected Period (JC(s) {', '.join(map(str, selected_jc_period))})**")
                        st.metric("Volume (T)", f"{metrics_selected['volume']:,.2f}")
                        st.metric("Value", f"₹ {metrics_selected['value']:,.0f}")
                        st.metric("DBs Billed", f"{metrics_selected['db_count']:,}")
                    with kpi3_col2:
                        st.markdown(f"**Avg. per Prior JC (JCs 1-{min_jc_selected-1})**")
                        st.metric("Avg. Volume (T)", f"{metrics_avg_prior['volume']:,.2f}", delta=calculate_delta_for_kpi3(metrics_selected['volume'], metrics_avg_prior['volume']))
                        st.metric("Avg. Value", f"₹ {metrics_avg_prior['value']:,.0f}", delta=calculate_delta_for_kpi3(metrics_selected['value'], metrics_avg_prior['value']))
                        st.metric("Avg. DBs Billed", f"{metrics_avg_prior['db_count']:,.1f}", delta=calculate_delta_for_kpi3(metrics_selected['db_count'], metrics_avg_prior['db_count']))
                else:
                    st.info(f"JC {min_jc_selected} is the first period. No prior JCs in this financial year to compare against for momentum.")
                st.markdown("---")

                # =========================================================================
                # 2. TREND CHARTS
                # =========================================================================
                st.subheader(f"2. Detailed Trends for {selected_fin_year}")
                trend1, trend2 = st.columns(2)
                with trend1:
                    st.markdown("##### Volume Trend Across JCs")
                    jc_trend_data = df_universe_base[df_universe_base['Fin Year'] == selected_fin_year].groupby('JCPeriod')['Volume in Tonnes'].sum().reset_index()
                    fig_jc_trend = px.line(jc_trend_data, x='JCPeriod', y='Volume in Tonnes', text='Volume in Tonnes', markers=True, title=f"JC-wise Volume Trend for {selected_fin_year}")
                    fig_jc_trend.update_traces(texttemplate='%{y:,.2f}', textposition='top center')
                    st.plotly_chart(fig_jc_trend, use_container_width=True)
                with trend2:
                    st.markdown("##### Weekly Volume Breakdown")
                    weekly_data = df_universe_base[(df_universe_base['Fin Year'] == selected_fin_year) & (df_universe_base['JCPeriod'].isin(selected_jc_period))]
                    weekly_trend_data = weekly_data.groupby('WeekNum')['Volume in Tonnes'].sum().reset_index().sort_values('WeekNum')
                    fig_weekly_trend = px.bar(weekly_trend_data, x='WeekNum', y='Volume in Tonnes', title=f"Weekly Volume in Selected JCs", text_auto='.2f')
                    st.plotly_chart(fig_weekly_trend, use_container_width=True)
                st.markdown("---")

                # =========================================================================
                # 3. PERIOD VS SAME PERIOD
                # =========================================================================
                st.subheader(f"3. Period vs. Same Period Performance (YoY for JC(s) {', '.join(map(str, selected_jc_period))})")
                period_metrics_data = []
                for year in yoy_years:
                    df_period = df_yoy_base[(df_yoy_base['Fin Year'] == year) & (df_yoy_base['JCPeriod'].isin(selected_jc_period))]
                    metrics = get_metrics(df_period)
                    metrics['Financial Year'] = year
                    period_metrics_data.append(metrics)
                kpi1_cols = st.columns(len(period_metrics_data))
                for i, data in enumerate(period_metrics_data):
                    with kpi1_cols[i]:
                        st.markdown(f"**{data['Financial Year']}**")
                        st.metric("Total Volume (T)", f"{data['volume']:,.2f}")
                        st.metric("Total Value", f"₹ {data['value']:,.0f}")
                        st.metric("DBs Billed", f"{data['db_count']:,}")
                st.markdown("---")

                # =========================================================================
                # 4. CUMULATIVE YTD PERFORMANCE
                # =========================================================================
                st.subheader(f"4. Cumulative YTD Performance (YoY for JCs 1-{max_jc})")
                ytd_metrics_data = []
                for year in yoy_years:
                    df_ytd = df_yoy_base[(df_yoy_base['Fin Year'] == year) & (df_yoy_base['JCPeriod'].isin(ytd_jcs))]
                    metrics = get_metrics(df_ytd)
                    metrics['Financial Year'] = year
                    ytd_metrics_data.append(metrics)
                kpi2_cols = st.columns(len(ytd_metrics_data))
                for i, data in enumerate(ytd_metrics_data):
                    with kpi2_cols[i]:
                        st.markdown(f"**{data['Financial Year']}**")
                        st.metric("YTD Volume (T)", f"{data['volume']:,.2f}")
                        st.metric("YTD Value", f"₹ {data['value']:,.0f}")
                        st.metric("YTD DBs Billed", f"{data['db_count']:,}")
                st.markdown("---")

                # =========================================================================
                # 5. CUMULATIVE YTD PRODUCT CATEGORY
                # =========================================================================
                st.subheader(f"5. Cumulative YTD Product Category Performance (JCs 1-{max_jc})")
                ytd_df_for_table = pd.DataFrame(ytd_metrics_data).sort_values('Financial Year', ascending=False).reset_index(drop=True)
                category_dfs = []
                for year in yoy_years:
                    df_ytd_cat = df_yoy_base[(df_yoy_base['Fin Year'] == year) & (df_yoy_base['JCPeriod'].isin(ytd_jcs))]
                    if not df_ytd_cat.empty:
                        cat_vol = df_ytd_cat.groupby('ProductCategory')['Volume in Tonnes'].sum().reset_index()
                        cat_vol.rename(columns={'Volume in Tonnes': f'Volume {year}'}, inplace=True)
                        category_dfs.append(cat_vol)
                if len(category_dfs) > 1:
                    final_cat_df = reduce(lambda left, right: pd.merge(left, right, on='ProductCategory', how='outer'), category_dfs).fillna(0)
                    current_year_col, prev_year_col = f'Volume {yoy_years[0]}', f'Volume {yoy_years[1]}'
                    if prev_year_col in final_cat_df.columns:
                        def calc_growth(row):
                            if row[prev_year_col] > 0: return (row[current_year_col] - row[prev_year_col]) / row[prev_year_col] * 100
                            elif row[current_year_col] > 0: return float('inf')
                            return 0
                        final_cat_df['Growth_Numeric'] = final_cat_df.apply(calc_growth, axis=1)
                        final_cat_df['Growth %'] = final_cat_df['Growth_Numeric'].map(lambda x: f"{x:.1f}%" if x != float('inf') else "New")
                        display_cols = ['ProductCategory'] + sorted([col for col in final_cat_df if 'Volume' in col], reverse=True) + ['Growth %']
                        st.dataframe(final_cat_df[display_cols], use_container_width=True)
    
    with tab1:
        st.subheader("Distributor Billing Gap, Frequency & Efficiency")
        df_tab1_active = df_universe_base[(df_universe_base['Fin Year'] == selected_fin_year) & (df_universe_base['JCPeriod'].isin(selected_jc_period))] if selected_jc_period else pd.DataFrame()
        
        ### <<< FIX START >>> ###
        # Add a check to ensure the dataframe is not empty before proceeding.
        # This prevents the KeyError when no JC Period is selected.
        if df_tab1_active.empty:
            st.warning("Please select at least one JC Period from the sidebar to view this analysis.")
        else:
            billed_dbs_active = set(df_tab1_active['Cust Code'].unique())
            gap_dbs_active = db_universe_dynamic - billed_dbs_active
            billing_efficiency = (len(billed_dbs_active) / len(db_universe_dynamic) * 100) if db_universe_dynamic else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total DBs (in selection)", len(db_universe_dynamic))
            col2.metric("DBs Billed (in selected JCs)", len(billed_dbs_active))
            col3.metric("DB Gaps (Not Billed)", len(gap_dbs_active))
            col4.metric("Billing Efficiency", f"{billing_efficiency:.1f}%")
            
            st.markdown("---")
            
            col_gap, col_freq = st.columns(2)
            with col_gap:
                st.subheader(f"JC-wise Gaps")
                st.info("This chart shows gaps for all JC periods in the FY for your base selection.")
                gap_data_dynamic = []
                for period in sorted(available_jc_periods):
                    billed_in_jc_for_selection = df_universe_base[(df_universe_base['Fin Year'] == selected_fin_year) & (df_universe_base['JCPeriod'] == period)]['Cust Code'].nunique()
                    gap_count = len(db_universe_dynamic) - billed_in_jc_for_selection
                    gap_data_dynamic.append({'JC Period': period, 'Gap Count': gap_count})
                if gap_data_dynamic:
                    gap_df_dynamic = pd.DataFrame(gap_data_dynamic)
                    fig_gap_dynamic = px.bar(gap_df_dynamic, x='JC Period', y='Gap Count', title='Unbilled Distributors per JC Period', text_auto=True)
                    st.plotly_chart(fig_gap_dynamic, use_container_width=True)
            
            with col_freq:
                st.subheader("Top Billed DBs - 360° View")
                st.info("Ranked by invoice count, with their volume and portfolio width for the selected period.")
                # This check is now slightly redundant due to the main wrapper, but it's harmless.
                if not df_tab1_active.empty:
                    freq_df_enhanced = df_tab1_active.groupby('Cust Name').agg(Invoice_Count=('Inv Num', 'nunique'),Volume_Tonnes=('Volume in Tonnes', 'sum'),Category_Count=('ProductCategory', 'nunique')).reset_index()
                    freq_df_enhanced = freq_df_enhanced.sort_values('Invoice_Count', ascending=False)
                    freq_df_enhanced['Volume_Tonnes'] = freq_df_enhanced['Volume_Tonnes'].map('{:,.2f}'.format)
                    st.dataframe(freq_df_enhanced.head(10), use_container_width=True)
                else: 
                    st.info("Select JC Period(s) to see the analysis.") # This line is unlikely to be reached now.
        ### <<< FIX END >>> ###

    with tab2:
        st.subheader("Distributor Investment Pattern Analysis")
        st.info("Analyzes a DB's billing within your filtered selection to understand their focus.")
        df_tab2_active = df_universe_base[(df_universe_base['Fin Year'] == selected_fin_year) & (df_universe_base['JCPeriod'].isin(selected_jc_period))] if selected_jc_period else pd.DataFrame()
        if df_tab2_active.empty:
            st.warning("No billing data for the current selections. Please adjust your filters and select a JC Period.")
        else:
            db_analysis_df = df_tab2_active.groupby(['Cust Name', 'City', 'DSM']).agg(Total_Volume_Tonnes=('Volume in Tonnes', 'sum'),Unique_Categories_Billed=('ProductCategory', 'nunique'),Product_Categories=('ProductCategory', lambda x: ', '.join(sorted(x.unique()))),).reset_index()
            freq_df = df_tab2_active.groupby('Cust Name')['WeekNum'].nunique().reset_index()
            freq_df.rename(columns={'WeekNum': 'Billing Frequency'}, inplace=True)
            categories_per_jc = df_tab2_active.groupby(['Cust Name', 'JCPeriod'])['ProductCategory'].nunique().reset_index()
            avg_cat_df = categories_per_jc.groupby('Cust Name')['ProductCategory'].mean().reset_index()
            avg_cat_df.rename(columns={'ProductCategory': 'Avg Categories Billed'}, inplace=True)
            db_analysis_df = pd.merge(db_analysis_df, freq_df, on='Cust Name', how='left')
            db_analysis_df = pd.merge(db_analysis_df, avg_cat_df, on='Cust Name', how='left')
            db_analysis_df.fillna(0, inplace=True)
            total_categories_available = df_tab2_active['ProductCategory'].nunique()
            def assign_investment_profile(row):
                if total_categories_available > 0 and (row['Unique_Categories_Billed'] / total_categories_available) >= 0.8: return "⭐ Portfolio Champion"
                elif row['Avg Categories Billed'] > 4: return "🚀 Expanding Player"
                elif row['Unique_Categories_Billed'] == 1: return "🎯 Category Loyalist"
                elif row['Avg Categories Billed'] > 1: return "🌱 Focused Buyer"
                else: return "❓ Occasional Buyer"
            db_analysis_df['Investment Profile'] = db_analysis_df.apply(assign_investment_profile, axis=1)
            db_analysis_df['Avg Categories Billed'] = db_analysis_df['Avg Categories Billed'].map('{:,.2f}'.format)
            display_cols = ['Cust Name', 'City', 'DSM', 'Investment Profile', 'Total_Volume_Tonnes', 'Unique_Categories_Billed', 'Avg Categories Billed', 'Billing Frequency', 'Product_Categories']
            st.dataframe(db_analysis_df.sort_values('Total_Volume_Tonnes', ascending=False)[display_cols], use_container_width=True)
            with st.expander("What do these Investment Profiles mean?"):
                st.markdown("""- **⭐ Portfolio Champion**: A top-tier distributor who invests in almost all available product categories (**80%+ of total unique categories**).\n- **🚀 Expanding Player**: A significant growth driver who buys a wide range of products, billing **more than 4 categories on average** per active JC.\n- **🎯 Category Loyalist**: A highly consistent distributor who focuses exclusively on a **single product category**.\n- **🌱 Focused Buyer**: A distributor who concentrates on a specific set of products, billing **1 to 4 categories on average**. Potential to cross-sell.\n- **❓ Occasional Buyer**: A distributor who doesn't fit the other patterns.""")
            st.markdown("---")
            st.subheader("Investment Profile Summary")
            col1_chart, col2_chart = st.columns(2)
            with col1_chart:
                profile_db_counts = db_analysis_df['Investment Profile'].value_counts().reset_index(); profile_db_counts.columns = ['Investment Profile', 'DB Count']; fig_db_count = px.bar(profile_db_counts, x='Investment Profile', y='DB Count', title='Count of DBs by Investment Profile', text_auto=True); st.plotly_chart(fig_db_count, use_container_width=True)
            with col2_chart:
                merged_for_chart = pd.merge(df_tab2_active, db_analysis_df[['Cust Name', 'Investment Profile']], on='Cust Name', how='left'); profile_cat_counts = merged_for_chart.groupby('Investment Profile')['ProductCategory'].nunique().reset_index(); profile_cat_counts.columns = ['Investment Profile', 'Unique Product Categories']; fig_cat_count = px.bar(profile_cat_counts, x='Investment Profile', y='Unique Product Categories', title='Product Categories Touched by Profile', text_auto=True); st.plotly_chart(fig_cat_count, use_container_width=True)
            st.markdown("---")
            st.subheader("Actionable Segments")
            profile_filter = st.selectbox("Filter by Investment Profile to find opportunities:", options=['All'] + sorted(list(db_analysis_df['Investment Profile'].unique())))
            if profile_filter != 'All':
                filtered_profile_df = db_analysis_df[db_analysis_df['Investment Profile'] == profile_filter]
                st.dataframe(filtered_profile_df[display_cols], use_container_width=True)

    with tab3:
        st.subheader("Distributor Evolution & Onboarding Analysis")
        if not selected_jc_period:
            st.warning("Please select at least one JC Period from the sidebar to run the evolution analysis.")
        else:
            target_jc, comparison_jc = max(selected_jc_period), max(selected_jc_period) - 1
            st.markdown(f"#### 📈 **Portfolio Changers Analysis: JC {target_jc} vs JC {comparison_jc}**")
            st.info(f"This analysis identifies distributors who changed focus in **JC {target_jc}** compared to **JC {comparison_jc}**.")
            df_tab3_universe = df_universe_base[df_universe_base['Fin Year'] == selected_fin_year]
            true_onboard_jc_map = df_tab3_universe.groupby('Cust Name')['JCPeriod'].min().to_dict()
            df_past_activity = df_tab3_universe[df_tab3_universe['JCPeriod'] < comparison_jc]
            last_active_jc_map = df_past_activity.groupby('Cust Name')['JCPeriod'].max().to_dict()
            df_target = df_tab3_universe[df_tab3_universe['JCPeriod'] == target_jc]
            df_comparison = df_tab3_universe[df_tab3_universe['JCPeriod'] == comparison_jc]
            current_portfolios = df_target.groupby('Cust Name')['ProductCategory'].unique().apply(set).to_dict()
            previous_portfolios = df_comparison.groupby('Cust Name')['ProductCategory'].unique().apply(set).to_dict()
            change_analysis_results = []
            active_dbs_target_jc = set(df_target['Cust Name'].unique())
            for db_name in active_dbs_target_jc:
                current_set = current_portfolios.get(db_name, set())
                previous_set = previous_portfolios.get(db_name, set())
                onboard_jc = true_onboard_jc_map.get(db_name, 0)
                added_categories, dropped_categories = current_set - previous_set, previous_set - current_set
                change_type = "Unchanged"
                if not previous_set:
                    if onboard_jc == target_jc: change_type = '🚀 New this JC'
                    else:
                        last_jc = last_active_jc_map.get(db_name)
                        change_type = f'🔄 Re-Engaged ({int(last_jc)})' if last_jc else '🔄 Re-Engaged'
                elif added_categories and not dropped_categories: change_type = "➕ Expansion"
                elif added_categories and dropped_categories: change_type = "🔄 Switch"
                elif dropped_categories and not added_categories: change_type = "➖ Contraction"
                if change_type != "Unchanged":
                    change_analysis_results.append({"Distributor": db_name,"ProductCategory Before": ', '.join(sorted(list(previous_set))) if previous_set else "None","Newly Added": ', '.join(sorted(list(added_categories))) if added_categories else "None","Dropped": ', '.join(sorted(list(dropped_categories))) if dropped_categories else "None","Change Type": change_type,"Onboard JC": onboard_jc})
            if change_analysis_results:
                change_df = pd.DataFrame(change_analysis_results)
                st.markdown("##### **Table 1: Portfolio Change Details**")
                st.dataframe(change_df[['Distributor', 'ProductCategory Before', 'Newly Added', 'Dropped', 'Change Type']], use_container_width=True)
                st.markdown("---")
                st.markdown("##### **Table 2: Portfolio Change Summary (Volume & Counts)**")
                change_df_summary = change_df.copy()
                volume_comparison_df = df_comparison.groupby('Cust Name')['Volume in Tonnes'].sum().reset_index().rename(columns={'Volume in Tonnes': f'Volume JC {comparison_jc} (T)'})
                volume_target_df = df_target.groupby('Cust Name')['Volume in Tonnes'].sum().reset_index().rename(columns={'Volume in Tonnes': f'Volume JC {target_jc} (T)'})
                change_df_summary = pd.merge(change_df_summary, volume_comparison_df, left_on='Distributor', right_on='Cust Name', how='left').fillna(0)
                change_df_summary = pd.merge(change_df_summary, volume_target_df, left_on='Distributor', right_on='Cust Name', how='left').fillna(0)
                change_df_summary.drop(columns=['Cust Name_x', 'Cust Name_y'], inplace=True, errors='ignore')
                change_df_summary['Ctg Count Before'] = change_df_summary['ProductCategory Before'].apply(lambda x: len(x.split(', ')) if x != "None" else 0)
                change_df_summary['Ctg Count After'] = change_df_summary['Ctg Count Before'] + (change_df_summary['Newly Added'].apply(lambda x: len(x.split(', ')) if x != "None" else 0)) - (change_df_summary['Dropped'].apply(lambda x: len(x.split(', ')) if x != "None" else 0))
                display_cols_summary = ['Distributor', f'Volume JC {comparison_jc} (T)', f'Volume JC {target_jc} (T)', 'Ctg Count Before', 'Ctg Count After', 'Change Type', 'Onboard JC']
                for col in [f'Volume JC {comparison_jc} (T)', f'Volume JC {target_jc} (T)']:
                    change_df_summary[col] = change_df_summary[col].map('{:,.2f}'.format)
                st.dataframe(change_df_summary[display_cols_summary], use_container_width=True)
                st.markdown("---")
                st.markdown("##### **Summary Chart: Distributor Counts by Change Type**")
                chart_df = change_df.copy()
                chart_df['Change Type'] = chart_df['Change Type'].apply(lambda x: '🔄 Re-Engaged' if 'Re-Engaged' in x else x)
                change_type_counts = chart_df['Change Type'].value_counts().reset_index()
                change_type_counts.columns = ['Change Type', 'Number of Distributors']
                fig_change_type = px.bar(change_type_counts, x='Change Type', y='Number of Distributors',title=f'Distributor Activity in JC {target_jc}', text_auto=True, color='Change Type',color_discrete_map={'🚀 New this JC': 'green', '🔄 Re-Engaged': 'orange', '➕ Expansion': 'royalblue','🔄 Switch': 'goldenrod', '➖ Contraction': 'firebrick'})
                st.plotly_chart(fig_change_type, use_container_width=True)
            else:
                st.info(f"No new distributors or portfolio changes found in JC {target_jc} compared to JC {comparison_jc} for the current selection.")

    with tab4:
        st.subheader("Product Category Deep Dive")
        df_tab4_active = df_universe_base[(df_universe_base['Fin Year'] == selected_fin_year) & (df_universe_base['JCPeriod'].isin(selected_jc_period))] if selected_jc_period else pd.DataFrame()
        if df_tab4_active.empty:
            st.warning("Please select JC Period(s) to analyze the product portfolio.")
        else:
            all_categories = sorted(df_tab4_active['ProductCategory'].unique())
            st.markdown("#### Overall Category Gap Snapshot")
            st.info("This chart shows the number of distributors in your selection who did NOT bill each product category in the selected period.")
            gap_data = []
            for cat in all_categories:
                billed_cat_dbs = set(df_tab4_active[df_tab4_active['ProductCategory'] == cat]['Cust Code'].unique())
                gap_count = len(db_universe_dynamic) - len(billed_cat_dbs)
                gap_data.append({'Product Category': cat, 'Gap DB Count': gap_count})
            if gap_data:
                gap_df = pd.DataFrame(gap_data).sort_values('Gap DB Count', ascending=False)
                fig_gap_chart = px.bar(gap_df, x='Product Category', y='Gap DB Count', title='Distributor Gaps per Product Category', text_auto=True); fig_gap_chart.update_xaxes(tickangle=45); st.plotly_chart(fig_gap_chart, use_container_width=True)
            st.markdown("---")
            st.markdown("#### Deep Dive Analysis for a Selected Category")
            selected_category = st.selectbox("Select a Product Category to analyze its performance and gaps", options=all_categories)
            if selected_category:
                billed_this_cat_set = set(df_tab4_active[df_tab4_active['ProductCategory'] == selected_category]['Cust Code'].unique())
                not_billed_this_cat_set = db_universe_dynamic - billed_this_cat_set
                billed_count, gap_count = len(billed_this_cat_set), len(not_billed_this_cat_set)
                total_volume_for_cat = df_tab4_active[df_tab4_active['ProductCategory'] == selected_category]['Volume in Tonnes'].sum()
                gap_percentage = (gap_count / len(db_universe_dynamic) * 100) if db_universe_dynamic else 0
                st.markdown(f"##### Metrics for **{selected_category}**")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("DBs Who Billed This", f"{billed_count}"); m_col2.metric("Total Volume (Tonnes)", f"{total_volume_for_cat:,.2f}"); m_col3.metric("Opportunity Gaps (DBs)", f"{gap_count}"); m_col4.metric("Gap Percentage", f"{gap_percentage:.1f}%")
                st.markdown("---")
                col_billed, col_gap = st.columns(2)
                with col_billed:
                    st.markdown(f"##### ✅ Billed '{selected_category}' ({billed_count} DBs)")
                    if billed_count > 0:
                        billed_df = df_tab4_active[df_tab4_active['Cust Code'].isin(billed_this_cat_set) & (df_tab4_active['ProductCategory'] == selected_category)]
                        billed_summary = billed_df.groupby(['Cust Name', 'City', 'DSM']).agg(Volume_of_Category_Tonnes=('Volume in Tonnes', 'sum'), Invoice_Count=('Inv Num', 'nunique')).reset_index().sort_values('Volume_of_Category_Tonnes', ascending=False)
                        st.dataframe(billed_summary, use_container_width=True)
                    else: st.info(f"No distributors billed '{selected_category}' in the current selection.")
                with col_gap:
                    st.markdown(f"##### 🚫 Gaps for '{selected_category}' ({gap_count} DBs)")
                    if gap_count > 0:
                        not_billed_df_base = df_universe_base[~df_universe_base['Cust Code'].isin(billed_this_cat_set)][['Cust Code', 'Cust Name', 'City', 'DSM']].drop_duplicates()
                        other_billed_data = df_tab4_active[df_tab4_active['Cust Code'].isin(not_billed_this_cat_set)]
                        if not other_billed_data.empty:
                            other_billed_summary = other_billed_data.groupby('Cust Name').agg(Other_Categories_Billed=('ProductCategory', lambda x: ', '.join(sorted(x.unique()))), Other_Volume_Tonnes=('Volume in Tonnes', 'sum')).reset_index().sort_values('Other_Volume_Tonnes', ascending=False)
                            final_gap_report = pd.merge(not_billed_df_base, other_billed_summary, on='Cust Name', how='left')
                            final_gap_report.fillna({'Other_Categories_Billed': 'None Billed in Selection', 'Other_Volume_Tonnes': 0}, inplace=True)
                            final_gap_report['Other_Volume_Tonnes'] = final_gap_report['Other_Volume_Tonnes'].map('{:,.2f}'.format)
                            st.dataframe(final_gap_report[['Cust Name', 'City', 'DSM', 'Other_Categories_Billed', 'Other_Volume_Tonnes']], use_container_width=True)
                        else:
                            not_billed_df_base['Other_Categories_Billed'] = 'None Billed in Selection'; not_billed_df_base['Other_Volume_Tonnes'] = '0.00'
                            st.dataframe(not_billed_df_base[['Cust Name', 'City', 'DSM', 'Other_Categories_Billed', 'Other_Volume_Tonnes']], use_container_width=True)
                    else: st.success(f"Excellent! All distributors in your selection have billed '{selected_category}'.")


