import streamlit as st
import os
# Unga original file la irundhu indha function ah import pannikonga
# Unga function irukura file peru correct ah irukanum (diagnosis.py)
from diagnosis import load_data_from_ftp

st.set_page_config(layout="wide")

st.title("Render Secrets & Environment Debugger 🕵️‍♂️")
st.warning("Indha page debugging ku mattum thaan. Problem solve aana aprom idha remove pannidalam.")

# ==============================================================================
st.header("Part 1: Environment Variables ah List Pannalam")
st.write("Render la set panna ella Environment Variables um inga list aaganum.")

# Get all environment variables
all_vars = dict(os.environ)
st.json(all_vars) # Displaying as a JSON for better readability

st.subheader("Naan Thedura `ST_FTP_...` variables iruka?")
ftp_vars_found = {
    "ST_FTP_HOST": os.environ.get("ST_FTP_HOST"),
    "ST_FTP_USER": os.environ.get("ST_FTP_USER"),
    "ST_FTP_PASSWORD": "Password irundha 'Exists' nu kaatum, illana 'MISSING'" if os.environ.get("ST_FTP_PASSWORD") else "MISSING",
    "ST_FTP_PRIMARY_PATH": os.environ.get("ST_FTP_PRIMARY_PATH"),
    "ST_FTP_CATEGORY_PATH": os.environ.get("ST_FTP_CATEGORY_PATH"),
}
st.write(ftp_vars_found)
# ==============================================================================

st.header("Part 2: `st.secrets` ah Bypass Panni Test Pannalam")
st.write("Namba ippo `st.secrets` use pannama, direct ah FTP details ah pass panni test pannuvom.")

# Unga FTP details ah inga hardcode panrom (for testing only)
hardcoded_ftp_details = {
    "host": "82.112.232.31",
    "user": "u363812745.vvd.in",
    "password": "5:qQCD]lwfaTsx?S",
    "primary_path": "/public_html/VVD_Hic/data_storage/primary.csv",
    "category_path": "/public_html/VVD_Hic/data_storage/prod_ctg.csv"
}

st.write("Naan use panna pora hardcoded FTP details:")
st.json(hardcoded_ftp_details)

if st.button("FTP Data Load Test ah Run Pannu"):
    st.info("Test starting... Konjam wait pannunga...")
    try:
        # Unga original function ah inga call panrom
        df = load_data_from_ftp(hardcoded_ftp_details)
        st.success("SUCCESS! FTP la irundhu data va load panna mudinjidhu!")
        st.write("Loaded Data ( முதல் 5 rows):")
        st.dataframe(df.head())
    except Exception as e:
        st.error("FAILURE! FTP data load pannum podhu error vandhuruchu.")
        st.exception(e)
# ==============================================================================
