import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Intelligent DB Growth Engine")

# --- DATA LOADING AND MAPPING (ENHANCED) ---
@st.cache_data
def load_and_map_data(primary_path, ctg_path):
    """Loads primary and category data, merges them, and preprocesses."""
    try:
        df_primary = pd.read_csv(primary_path, encoding='latin1')
        df_ctg_map = pd.read_csv(ctg_path, encoding='latin1')
        df = pd.merge(df_primary, df_ctg_map, on='Prod Ctg', how='left')
        df['Prod Ctg_Updated'].fillna(df['Prod Ctg'], inplace=True)
        df.drop('Prod Ctg', axis=1, inplace=True)
        df.rename(columns={'Prod Ctg_Updated': 'Prod Ctg'}, inplace=True)
        
        df['Inv Date'] = pd.to_datetime(df['Inv Date'], dayfirst=True, errors='coerce')
        key_cols = ['Cust Code', 'Cust Name', 'JCPeriod', 'DSM', 'CustomerClass', 'Prod Ctg', 'Inv Num']
        df.dropna(subset=key_cols, inplace=True)
        
        df['JCPeriod'] = df['JCPeriod'].astype(int)
        df['Qty in Ltrs/Kgs'] = pd.to_numeric(df['Qty in Ltrs/Kgs'], errors='coerce').fillna(0)
        df['Volume in Tonnes'] = df['Qty in Ltrs/Kgs'] / 1000
        df['Fin Year'] = df['Fin Year'].astype(str)
        return df
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please make sure both CSV files are in the correct path.")
        return None
    except Exception as e:
        st.error(f"An error occurred during data loading and mapping: {e}")
        return None

# --- MAIN APP ---
primary_file_path = r'E:\Automation\primary.csv'
ctg_file_path = r'E:\Automation\prod_ctg.csv'
df_original = load_and_map_data(primary_file_path, ctg_file_path)

if df_original is not None:
    st.title("💡 Intelligent Distributor Growth Engine")
    st.markdown("_From Data to Decisions: Identify Gaps, Analyze Investment Patterns, and Unlock Potential._")
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Control Panel")
    
    available_fin_years = sorted(df_original['Fin Year'].unique(), reverse=True)
    selected_fin_year = st.sidebar.selectbox('Select Financial Year', options=available_fin_years)
    
    df_fy_filtered = df_original[df_original['Fin Year'] == selected_fin_year]

    available_jc_periods = sorted(df_fy_filtered['JCPeriod'].unique())
    selected_jc_period = st.sidebar.multiselect('Filter by JC Period(s)', options=available_jc_periods, default=[])
    available_dsms = sorted(df_fy_filtered['DSM'].unique())
    selected_dsm = st.sidebar.multiselect('Filter by DSM', options=available_dsms, default=[])
    available_cust_class = sorted(df_fy_filtered['CustomerClass'].unique())
    selected_cust_class = st.sidebar.multiselect('Filter by Customer Class', options=available_cust_class, default=[])

    # --- DYNAMIC FILTERING LOGIC ---
    df_universe_base = df_fy_filtered.copy()
    if selected_dsm:
        df_universe_base = df_universe_base[df_universe_base['DSM'].isin(selected_dsm)]
    if selected_cust_class:
        df_universe_base = df_universe_base[df_universe_base['CustomerClass'].isin(selected_cust_class)]
        
    db_universe_dynamic = set(df_universe_base['Cust Code'].unique())

    df_active = df_universe_base.copy()
    if selected_jc_period:
        df_active = df_active[df_active['JCPeriod'].isin(selected_jc_period)]
    
    # --- TAB LAYOUT ---
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Gaps & Frequency", "🧠 Investment Analysis", "✨ DB Evolution & Potential", "📦 Product Deep Dive"])

    with tab1: # --- MODIFIED AS PER REQUEST ---
        st.subheader("Distributor Billing Gap, Frequency & Efficiency")
        billed_dbs_active = set(df_active['Cust Code'].unique())
        if not selected_jc_period: 
             billed_dbs_active = db_universe_dynamic
        
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
            # --- START OF MODIFICATION ---
            st.info("This chart shows gaps for all JC periods in the FY. The counts are dynamic to your DSM & Customer Class selection.")

            gap_data_dynamic = []
            
            # The universe for the chart is the one filtered by DSM and Class (db_universe_dynamic)
            total_dbs_in_selection = db_universe_dynamic
            
            # Loop through ALL available JC periods to keep the chart's x-axis static
            for period in sorted(available_jc_periods):
                # Count billed DBs for each period, but only from within the selected universe (df_universe_base)
                billed_in_jc_for_selection = df_universe_base[df_universe_base['JCPeriod'] == period]['Cust Code'].nunique()
                
                # Calculate the gap based on the selected universe size
                gap_count = len(total_dbs_in_selection) - billed_in_jc_for_selection
                
                gap_data_dynamic.append({'JC Period': period, 'Gap Count': gap_count})
                
            if gap_data_dynamic:
                gap_df_dynamic = pd.DataFrame(gap_data_dynamic)
                fig_gap_dynamic = px.bar(gap_df_dynamic, x='JC Period', y='Gap Count', title='Unbilled Distributors per JC Period (in selection)', text_auto=True)
                st.plotly_chart(fig_gap_dynamic, use_container_width=True)
            # --- END OF MODIFICATION ---

        with col_freq:
            st.subheader("Top Billed DBs - 360° View")
            st.info("Ranked by invoice count, with their volume and portfolio width.")
            if not df_active.empty:
                freq_df_enhanced = df_active.groupby('Cust Name').agg(Invoice_Count=('Inv Num', 'nunique'),Volume_Tonnes=('Volume in Tonnes', 'sum'),Category_Count=('Prod Ctg', 'nunique')).reset_index()
                freq_df_enhanced = freq_df_enhanced.sort_values('Invoice_Count', ascending=False)
                freq_df_enhanced['Volume_Tonnes'] = freq_df_enhanced['Volume_Tonnes'].map('{:,.2f}'.format)
                st.dataframe(freq_df_enhanced.head(10), use_container_width=True)
            else:
                st.info("Select JC Period(s) to see the analysis.")

    with tab2: # UNCHANGED
        st.subheader("Distributor Investment Pattern Analysis")
        st.info("Analyzes a DB's billing within your filtered selection to understand their focus.")
        
        if df_active.empty:
            st.warning("No billing data for the current selections. Please select JC Period(s) to analyze investment patterns.")
        else:
            db_analysis_df = df_active.groupby(['Cust Name', 'City', 'DSM']).agg(
                Total_Volume_Tonnes=('Volume in Tonnes', 'sum'),
                Unique_Categories_Billed=('Prod Ctg', 'nunique'),
                Product_Categories=('Prod Ctg', lambda x: ', '.join(sorted(x.unique()))),
                Total_Billing_JCs=('JCPeriod', 'nunique')
            ).reset_index()
            if not df_active.empty and len(db_universe_dynamic) > 0:
                total_categories_available = df_active['Prod Ctg'].nunique()
                total_jcs_selected = df_active['JCPeriod'].nunique() if selected_jc_period else 0
                avg_volume_per_db = df_active['Volume in Tonnes'].sum() / len(db_universe_dynamic)
                def assign_investment_profile(row):
                    cat_diversity,jc_consistency = row['Unique_Categories_Billed'],row['Total_Billing_JCs']
                    if cat_diversity == 1 and jc_consistency == total_jcs_selected and total_jcs_selected > 0: return "🎯 Category Loyalist"
                    elif total_categories_available > 0 and cat_diversity / total_categories_available >= 0.8: return "⭐ Portfolio Champion"
                    elif total_categories_available > 0 and cat_diversity / total_categories_available <= 0.3: return "🌱 Focused Buyer"
                    elif total_categories_available > 0 and cat_diversity / total_categories_available > 0.3 and row['Total_Volume_Tonnes'] > avg_volume_per_db: return "🚀 Expanding Player"
                    else: return "❓ Occasional Buyer"
                db_analysis_df['Investment Profile'] = db_analysis_df.apply(assign_investment_profile, axis=1)
            else: db_analysis_df['Investment Profile'] = "N/A"
            display_cols = ['Cust Name', 'City', 'DSM', 'Investment Profile', 'Total_Volume_Tonnes', 'Unique_Categories_Billed', 'Product_Categories', 'Total_Billing_JCs']
            st.dataframe(db_analysis_df.sort_values('Total_Volume_Tonnes', ascending=False)[display_cols], use_container_width=True)
            with st.expander("What do these Investment Profiles mean?"):
                st.markdown("""- **⭐ Portfolio Champion**: A top-tier distributor who invests in almost all available product categories (80%+).\n- **🚀 Expanding Player**: A significant distributor who buys a wide range of products and has an above-average billing volume. A key growth driver.\n- **🎯 Category Loyalist**: A highly consistent distributor who focuses exclusively on a single product category across all selected periods.\n- **🌱 Focused Buyer**: A distributor who concentrates on a small, specific set of products (less than 30% of the portfolio). Potential to cross-sell.\n- **❓ Occasional Buyer**: A distributor who doesn't fit the other patterns, often with lower volume or inconsistent buying behavior.""")
            st.markdown("---")
            st.subheader("Investment Profile Summary")
            col1_chart, col2_chart = st.columns(2)
            with col1_chart:
                profile_db_counts = db_analysis_df['Investment Profile'].value_counts().reset_index(); profile_db_counts.columns = ['Investment Profile', 'DB Count']; fig_db_count = px.bar(profile_db_counts, x='Investment Profile', y='DB Count', title='Count of DBs by Investment Profile', text_auto=True); st.plotly_chart(fig_db_count, use_container_width=True)
            with col2_chart:
                merged_for_chart = pd.merge(df_active, db_analysis_df[['Cust Name', 'Investment Profile']], on='Cust Name', how='left'); profile_cat_counts = merged_for_chart.groupby('Investment Profile')['Prod Ctg'].nunique().reset_index(); profile_cat_counts.columns = ['Investment Profile', 'Unique Product Categories']; fig_cat_count = px.bar(profile_cat_counts, x='Investment Profile', y='Unique Product Categories', title='Product Categories Touched by Profile', text_auto=True); st.plotly_chart(fig_cat_count, use_container_width=True)
            st.markdown("---")
            st.subheader("Actionable Segments")
            profile_filter = st.selectbox("Filter by Investment Profile to find opportunities:", options=['All'] + sorted(list(db_analysis_df['Investment Profile'].unique())))
            if profile_filter != 'All':
                filtered_profile_df = db_analysis_df[db_analysis_df['Investment Profile'] == profile_filter]
                st.dataframe(filtered_profile_df[display_cols], use_container_width=True)
    
    with tab3: # UNCHANGED
        st.subheader("Distributor Evolution & Onboarding Analysis")
        if not selected_jc_period:
            jc_periods_for_analysis = available_jc_periods
        else:
            jc_periods_for_analysis = selected_jc_period
        if len(jc_periods_for_analysis) < 2:
            st.warning("Evolution analysis requires at least two JC periods to compare. Clear the JC Period filter to analyze the full year, or select two or more periods.")
        else:
            st.markdown(f"#### 📈 **Portfolio Changers Analysis**")
            st.info("Distributors who are new, expanded, or switched their focus in the target JC period.")
            target_jc = max(jc_periods_for_analysis)
            st.info(f"Comparing portfolio changes in **JC {target_jc}** against preceding selected periods.")
            analysis_data = df_universe_base[df_universe_base['JCPeriod'].isin(jc_periods_for_analysis)]
            db_onboard_jc_map = analysis_data.groupby('Cust Name')['JCPeriod'].min().to_dict()
            previous_portfolios = analysis_data[analysis_data['JCPeriod'] < target_jc].groupby('Cust Name')['Prod Ctg'].unique().apply(set).to_dict()
            current_portfolios = analysis_data[analysis_data['JCPeriod'] == target_jc].groupby('Cust Name')['Prod Ctg'].unique().apply(set).to_dict()
            change_analysis_results = []
            for db_name, current_set in current_portfolios.items():
                previous_set = previous_portfolios.get(db_name, set())
                added_categories,dropped_categories = current_set - previous_set,previous_set - current_set
                onboard_jc = db_onboard_jc_map.get(db_name, 0)
                if not previous_set:
                    change_type = '🔥 Recent' if onboard_jc == target_jc or onboard_jc == target_jc - 1 else '🚀 New'
                elif added_categories and not dropped_categories:
                    change_type = "➕ Expansion"
                elif added_categories and dropped_categories:
                    change_type = "🔄 Switch"
                else:
                    continue
                change_analysis_results.append({"Distributor": db_name, "Existing Prod Ctg": ', '.join(sorted(list(previous_set))) if previous_set else "None", "Newly Added": ', '.join(sorted(list(added_categories))), "Dropped": ', '.join(sorted(list(dropped_categories))) if dropped_categories else "None", "Change Type": change_type, "Onboard JC": onboard_jc})
            if change_analysis_results:
                change_df = pd.DataFrame(change_analysis_results)
                st.markdown("##### **Table 1: Portfolio Change Details**")
                st.dataframe(change_df[['Distributor', 'Existing Prod Ctg', 'Newly Added', 'Dropped', 'Change Type', 'Onboard JC']], use_container_width=True)
                st.markdown("##### **Table 2: Portfolio Change Summary (Counts & Volume)**")
                volume_before_df = analysis_data[analysis_data['JCPeriod'] < target_jc].groupby('Cust Name')['Volume in Tonnes'].sum().reset_index().rename(columns={'Volume in Tonnes': 'Volume Before (Tonnes)'})
                volume_target_jc_df = analysis_data[analysis_data['JCPeriod'] == target_jc].groupby('Cust Name')['Volume in Tonnes'].sum().reset_index().rename(columns={'Volume in Tonnes': 'Volume in Target JC (Tonnes)'})
                change_df_summary = change_df.copy()
                change_df_summary = pd.merge(change_df_summary, volume_before_df, left_on='Distributor', right_on='Cust Name', how='left').fillna(0)
                change_df_summary = pd.merge(change_df_summary, volume_target_jc_df, left_on='Distributor', right_on='Cust Name', how='left').fillna(0)
                change_df_summary.drop(columns=['Cust Name_x', 'Cust Name_y'], inplace=True, errors='ignore')
                change_df_summary['Existing_Count'] = change_df_summary['Existing Prod Ctg'].apply(lambda x: len(x.split(', ')) if x != "None" else 0)
                change_df_summary['Added_Count'] = change_df_summary['Newly Added'].apply(lambda x: len(x.split(', ')))
                change_df_summary['Dropped_Count'] = change_df_summary['Dropped'].apply(lambda x: len(x.split(', ')) if x != "None" else 0)
                change_df_summary['Final_Prod_Ctg_Count'] = change_df_summary['Existing_Count'] + change_df_summary['Added_Count'] - change_df_summary['Dropped_Count']
                display_cols_summary = ['Distributor', 'Volume Before (Tonnes)', 'Volume in Target JC (Tonnes)', 'Existing_Count', 'Added_Count', 'Dropped_Count', 'Final_Prod_Ctg_Count', 'Change Type', 'Onboard JC']
                st.dataframe(change_df_summary[display_cols_summary].rename(columns={'Existing_Count': 'Existing Ctg', 'Added_Count': 'Added Ctg', 'Dropped_Count': 'Dropped Ctg', 'Final_Prod_Ctg_Count': 'Final Ctg Count'}), use_container_width=True)
                st.markdown("---")
                st.markdown("##### **Summary Chart: Distributor Counts by Change Type**")
                change_type_counts = change_df['Change Type'].value_counts().reset_index(); change_type_counts.columns = ['Change Type', 'Number of Distributors']
                fig_change_type = px.bar(change_type_counts, x='Change Type', y='Number of Distributors', title=f'Distributor Activity in JC {target_jc}', text_auto=True, color='Change Type', color_discrete_map={'🚀 New': 'green', '🔥 Recent': 'orange', '➕ Expansion': 'royalblue', '🔄 Switch': 'goldenrod'})
                st.plotly_chart(fig_change_type, use_container_width=True)
            else:
                st.info(f"No new distributors or portfolio changes found in JC {target_jc} for the current selection.")

    with tab4: # UNCHANGED
        st.subheader("Product Category Deep Dive")
        if df_active.empty:
            st.warning("Please select JC Period(s) to analyze the product portfolio.")
        else:
            all_categories = sorted(df_active['Prod Ctg'].unique())
            st.markdown("#### Overall Category Gap Snapshot")
            st.info("This chart shows the number of distributors who did NOT bill each product category. Use this to identify the biggest opportunity areas.")
            gap_data = []
            for cat in all_categories:
                billed_cat_dbs = set(df_active[df_active['Prod Ctg'] == cat]['Cust Code'].unique())
                gap_count = len(db_universe_dynamic - billed_cat_dbs)
                gap_data.append({'Product Category': cat, 'Gap DB Count': gap_count})
            if gap_data:
                gap_df = pd.DataFrame(gap_data).sort_values('Gap DB Count', ascending=False)
                fig_gap_chart = px.bar(gap_df, x='Product Category', y='Gap DB Count', title='Distributor Gaps per Product Category', text_auto=True); fig_gap_chart.update_xaxes(tickangle=45); st.plotly_chart(fig_gap_chart, use_container_width=True)
            st.markdown("---")
            st.markdown("#### Deep Dive Analysis for a Selected Category")
            selected_category = st.selectbox("Select a Product Category to analyze its performance and gaps", options=all_categories)
            if selected_category:
                billed_this_cat_set = set(df_active[df_active['Prod Ctg'] == selected_category]['Cust Code'].unique())
                not_billed_this_cat_set = db_universe_dynamic - billed_this_cat_set
                billed_count, gap_count = len(billed_this_cat_set), len(not_billed_this_cat_set)
                total_volume_for_cat = df_active[df_active['Prod Ctg'] == selected_category]['Volume in Tonnes'].sum()
                gap_percentage = (gap_count / len(db_universe_dynamic) * 100) if db_universe_dynamic else 0
                st.markdown(f"##### Metrics for **{selected_category}**")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("DBs Who Billed This", f"{billed_count}"); m_col2.metric("Total Volume (Tonnes)", f"{total_volume_for_cat:,.2f}"); m_col3.metric("Opportunity Gaps (DBs)", f"{gap_count}"); m_col4.metric("Gap Percentage", f"{gap_percentage:.1f}%")
                st.markdown("---")
                col_billed, col_gap = st.columns(2)
                with col_billed:
                    st.markdown(f"##### ✅ Billed '{selected_category}' ({billed_count} DBs)")
                    if billed_count > 0:
                        billed_df = df_active[df_active['Cust Code'].isin(billed_this_cat_set) & (df_active['Prod Ctg'] == selected_category)]
                        billed_summary = billed_df.groupby(['Cust Name', 'City', 'DSM']).agg(Volume_of_Category_Tonnes=('Volume in Tonnes', 'sum'), Invoice_Count=('Inv Num', 'nunique')).reset_index().sort_values('Volume_of_Category_Tonnes', ascending=False)
                        st.dataframe(billed_summary, use_container_width=True)
                    else: st.info(f"No distributors billed '{selected_category}' in the current selection.")
                with col_gap:
                    st.markdown(f"##### 🚫 Gaps for '{selected_category}' ({gap_count} DBs)")
                    if gap_count > 0:
                        not_billed_df_base = df_universe_base[df_universe_base['Cust Code'].isin(not_billed_this_cat_set)][['Cust Code', 'Cust Name', 'City', 'DSM']].drop_duplicates()
                        other_billed_data = df_active[df_active['Cust Code'].isin(not_billed_this_cat_set)]
                        if not other_billed_data.empty:
                            other_billed_summary = other_billed_data.groupby('Cust Name').agg(Other_Categories_Billed=('Prod Ctg', lambda x: ', '.join(sorted(x.unique()))), Other_Volume_Tonnes=('Volume in Tonnes', 'sum')).reset_index().sort_values('Other_Volume_Tonnes', ascending=False)
                            final_gap_report = pd.merge(not_billed_df_base, other_billed_summary, on='Cust Name', how='left'); final_gap_report.fillna({'Other_Categories_Billed': 'None Billed in Selection', 'Other_Volume_Tonnes': 0}, inplace=True)
                            final_gap_report['Other_Volume_Tonnes'] = final_gap_report['Other_Volume_Tonnes'].map('{:,.2f}'.format)
                            st.dataframe(final_gap_report[['Cust Name', 'City', 'DSM', 'Other_Categories_Billed', 'Other_Volume_Tonnes']], use_container_width=True)
                        else:
                            not_billed_df_base['Other_Categories_Billed'] = 'None Billed in Selection'; not_billed_df_base['Other_Volume_Tonnes'] = '0.00'
                            st.dataframe(not_billed_df_base[['Cust Name', 'City', 'DSM', 'Other_Categories_Billed', 'Other_Volume_Tonnes']], use_container_width=True)
                    else: st.success(f"Excellent! All distributors in your selection have billed '{selected_category}'.")