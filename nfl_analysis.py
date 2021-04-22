import pandas as pd
import numpy as np
import streamlit as st

data = pd.read_excel('C:/Users/Darragh/Documents/Python/NFL/NFL_2019_Data.xlsx')

st.table(data.head())

# here we go, create a column with Last Game Turnover
# create the season to date cover for every team
# data['Home_Last_Turnover']=test test TEST