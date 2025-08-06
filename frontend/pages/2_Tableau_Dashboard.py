# pages/2_Tableau_Dashboard.py
import streamlit as st

st.set_page_config(page_title="Tableau Dashboard", layout="wide")

st.title("Loan Default Analysis Dashboard")

st.markdown("""
""")

# Use the extracted embed URL here
tableau_embed_url = "https://public.tableau.com/views/CDAC_Project_Dashboard/Dashboard1?:language=en-US&:display_count=n&:origin=viz_share_link&:embed=y&:showVizHome=no&:tabs=yes&:toolbar=yes"

st.components.v1.iframe(tableau_embed_url, height=800, scrolling=True)

st.markdown("---")
st.markdown("Dashboard created using Tableau.")
