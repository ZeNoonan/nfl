import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

@st.cache
def read_data(file):
    return pd.read_excel(file) 
data_2019 = read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2019_Data.xlsx').copy()
data_2020=read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2020_Data.xlsx').copy()

# st.table(data.head())
def spread_workings(data):
    data['home_win']=data['Home Points'] - data['Away Points']
    data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
    data['home_cover']=np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1,
    np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0))
    data['away_cover'] = -data['home_cover']
    return data

spread=spread_workings(data_2020)
# st.write(spread)

def season_cover_workings(data):
    season_cover_df=(data.set_index('Week').loc[1:,:]).reset_index()
    home_cover_df = (season_cover_df.loc[:,['Week','Home ID','home_cover']]).rename(columns={'Home ID':'ID','home_cover':'cover'})
    away_cover_df = (season_cover_df.loc[:,['Week','Away ID','away_cover']]).rename(columns={'Away ID':'ID','away_cover':'cover'})
    season_cover=pd.concat([home_cover_df,away_cover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','ID'],ascending=['True','True'])

spread_1 = season_cover_workings(spread)

def season_cover_2(season_cover_df):    
    # https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
    season_cover_df['cover'] = season_cover_df.groupby (['ID'])['cover'].transform(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers

    return season_cover_df

def season_cover_3(data):
    data['cover_sign'] = np.where((data['cover'] > 0), 1, np.where((data['cover'] < 0),-1,0))
    return data

spread_2=season_cover_2(spread_1)
st.write(spread_2.sort_values(by=['ID','Week'],ascending=['True','True']))
spread_3=season_cover_3(spread_2)
st.write(spread_3.sort_values(by=['ID','Week'],ascending=['True','True']))