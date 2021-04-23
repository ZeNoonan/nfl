import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")
data = pd.read_excel('C:/Users/Darragh/Documents/Python/NFL/NFL_2019_Data.xlsx')

# st.table(data.head())
data['home_win']=data['Home Points'] - data['Away Points']
data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
data['home_cover']=np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1, np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0))
data['home_season_cover'] = data.groupby (['Home ID'])['home_cover'].cumsum()

st.dataframe(data)
# here we go, create a column with Last Game Turnover
# create the season to date cover for every team
# data['Home_Last_Turnover']=test test TEST