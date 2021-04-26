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
    data=data.rename(columns={'Net Turnover':'home_turnover'})
    data['away_turnover'] = -data['home_turnover']
    return data


# st.write(spread)

def season_cover_workings(data,home,away,name,week_start):
    season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_cover_df = (season_cover_df.loc[:,['Week','Home ID',home]]).rename(columns={'Home ID':'ID',home:name})
    away_cover_df = (season_cover_df.loc[:,['Week','Away ID',away]]).rename(columns={'Away ID':'ID',away:name})
    season_cover=pd.concat([home_cover_df,away_cover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','ID'],ascending=['True','True'])



def season_cover_2(season_cover_df,column_name):    
    # https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
    season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].transform(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

def season_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

def turnover_2(season_cover_df):    
    # https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group
    season_cover_df['prev_turnover']=season_cover_df.groupby('ID')['turnover'].shift()
    return season_cover_df.sort_values(by=['ID','Week'],ascending=True)
    # return season_cover_df


spread=spread_workings(data_2020)
st.write(spread)

spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',1)
spread_2=season_cover_2(spread_1,'cover')
spread_3=season_cover_3(spread_2,'cover_sign','cover')
st.write('this is season to date cover')
st.write(spread_3.sort_values(by=['ID','Week'],ascending=['True','True']))


turnover=spread_workings(data_2020)
turnover_1 = season_cover_workings(turnover,'home_turnover','away_turnover','turnover',-1)
turnover_2=turnover_2(turnover_1)
turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')
st.write('this is last game turnover')
st.write(turnover_3.sort_values(by=['ID','Week'],ascending=['True','True']))

st.write('Next trying to do the Power Rankings')
st.write('maybe I should do matrix multiplication on a dummy first to see what shape i Need them in?')
matrix_df=spread_workings(data_2020)
weights = np.array([0.125, 0.25,0.5,1]) # the order mattered!! took me a while to figure this out
sum_weights = np.sum(weights)
# df['Weighted_ma'] = (df['Clean_Pts'].fillna(0).rolling(window=4, center=False)\
#     .apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)) # raw=False
#     # using the fillna ensures no NaN as this function requires min 4 data points in a row - .fillna(method='ffill')
    # so just be careful the result is the last time player had 4 weeks in a row