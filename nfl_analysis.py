import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

@st.cache
def read_data():
    return pd.read_excel('C:/Users/Darragh/Documents/Python/NFL/NFL_2019_Data.xlsx') 
data = read_data()

# st.table(data.head())
data['home_win']=data['Home Points'] - data['Away Points']
data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
data['home_cover']=np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1, np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0))
# data['home_season_cover'] = data.groupby (['Home ID'])['home_cover'].cumsum()
season_cover_df=(data.set_index('Week').loc[0:,:]).copy()
new = season_cover_df.loc[:,['Home ID','Away ID','home_cover']].copy()
# st.write(new)
test=new.reset_index()
# st.write(test)
test_test = pd.melt(test,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
# test_test=set_index('Week').drop('index',axis=1)
# st.write(test_test) #Looks like this works

test_test['home_season_cover'] = test_test.groupby (['ID'])['home_cover'].transform(lambda x: x.cumsum().shift())
st.write(test_test)
st.write(test_test.reset_index().sort_values(by=['Week','ID'],ascending=True))
# season_cover_df['home_season_cover'] = season_cover_df.groupby (['Home ID'])['home_cover'].transform(lambda x: x.cumsum().shift())
# https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
# don't think this is going to work, need away season to date cover as well need a specific df is what i'm thinking



st.dataframe(data)

st.write(season_cover_df)

# temp = pd.DataFrame(data=[['a',1],['a',1],['a',1],['b',1],['b',1],['b',1],['c',1],['c',1]], columns=['ID','X'])
# st.write(temp)
# here we go, create a column with Last Game Turnover
# create the season to date cover for every team
# data['Home_Last_Turnover']=test test TEST