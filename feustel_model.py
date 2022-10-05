import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
# import os
import base64 
import altair as alt
# import datetime as dt
from datetime import date, timedelta
# from st_aggrid import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

st.set_page_config(layout="wide")

@st.cache
def read_csv_data(file):
    return pd.read_csv(file)

@st.cache
def read_data(file):
    return pd.read_excel(file)

df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_24_09_22.xlsx')
df=df.copy()
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['season_month'] = df['month'].map({9:1,10:2,11:3,12:4,1:5,2:6})
# NL_Raw_Clean['calendar_month']=NL_Raw_Clean['Per.'].map({1:9,2:10,3:11,4:12,5:1,6:2,7:3,8:4,9:5,10:6,11:7,12:8,19:8})
df['season_year'] = np.where((df['season_month'] < 5), df['year'], df['year']-1)
df['Home Team']=df['Home Team'].replace({'Washington Football Team':'Washington Commanders','Washington Redskins':'Washington Commanders','St. Louis Rams':'Los Angeles Rams',
'Oakland Raiders':'Las Vegas Raiders','San Diego Chargers':'Los Angeles Chargers'})
df['Away Team']=df['Away Team'].replace({'Washington Football Team':'Washington Commanders','Washington Redskins':'Washington Commanders','St. Louis Rams':'Los Angeles Rams',
'Oakland Raiders':'Las Vegas Raiders','San Diego Chargers':'Los Angeles Chargers'})
# NL_Raw_Clean['calendar_year']=NL_Raw_Clean['calendar_year']+2000
# NL_Raw_Clean=NL_Raw_Clean.rename(columns={'calendar_year':'year', 'calendar_month':'month'})

# df['']
st.write(df.sort_values(by='Date'))
# for _ in df.groupby('season_year'):
#     pass

df_offensive_home=df.loc[:,['Date','Home Team', 'Home Score', 'season_year']].rename(columns={'Home Team':'team','Home Score':'score'})

df_offensive_home['home_away']=1
df_offensive_away=df.loc[:,['Date','Away Team','Away Score', 'season_year']].rename(columns={'Away Team':'team','Away Score':'score'})
df_offensive_away['home_away']=-1
df_offensive=pd.concat([df_offensive_home,df_offensive_away],axis=0).sort_values(by=['team','Date'],ascending=True).reset_index().drop('index',axis=1)
# st.write('dataframe after concat',df_offensive)
# st.write(df_offensive.shape)
df_groupby_scores=df_offensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum().reset_index().drop('level_2',axis=1)
# st.write('doing a seperate groupby')
# st.dataframe(df_groupby_scores, use_container_width=True)
df_offensive['sum_score']=df_offensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum()\
    .reset_index().drop(['level_2','team','season_year'],axis=1)
df_offensive['mean_score']=df_offensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).mean()\
    .reset_index().drop(['level_2','team','season_year'],axis=1)
df_offensive=df_offensive.rename(columns={'score':'pts_scored','mean_score':'4_game_pts_scored'}).sort_values(by=['team','Date']).drop('sum_score',axis=1)
# st.write(df_offensive)
# df_groupby_scores=df_offensive.groupby(['team','season_year','Date'])['score'].rolling(window=4,min_periods=4, center=False).sum().reset_index()

df_defensive_home=df.loc[:,['Date','Home Team', 'Away Score', 'season_year']].rename(columns={'Home Team':'team','Away Score':'score'})
df_defensive_away=df.loc[:,['Date','Away Team','Home Score', 'season_year']].rename(columns={'Away Team':'team','Home Score':'score'})
df_defensive=pd.concat([df_defensive_home,df_defensive_away],axis=0).sort_values(by=['team','Date'],ascending=True).reset_index().drop('index',axis=1)
# st.write('dataframe after concat',df_defensive)
# st.write(df_defensive.shape)
df_groupby_scores=df_defensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum().reset_index().drop('level_2',axis=1)
# st.write('doing a seperate groupby')
# st.dataframe(df_groupby_scores, use_container_width=True)
df_defensive['sum_score']=df_defensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum()\
    .reset_index().drop(['level_2','team','season_year'],axis=1)
df_defensive['mean_score']=df_defensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).mean()\
    .reset_index().drop(['level_2','team','season_year'],axis=1)
df_defensive=df_defensive.rename(columns={'score':'pts_conceded','mean_score':'4_game_pts_conceded'}).sort_values(by=['team','Date']).drop('sum_score',axis=1)
# st.write(df_defensive)
df_new=pd.merge(df_offensive,df_defensive,how='outer')
df_new['team_cum_sum_pts']=df_new.groupby(['team'])['pts_scored'].cumsum()
df_new['team_cum_sum_games']=df_new.groupby(['team'])['pts_scored'].cumcount()+1
df_new['rolling_avg_team_pts_scored']=df_new['team_cum_sum_pts'] / df_new['team_cum_sum_games']
df_new=df_new.sort_values(by=['Date','team'])
df_new['date_avg_pts_rolling']=df_new['pts_scored'].expanding().mean()
st.write('sorted by date avg score by date',df_new)
df_new=df_new.sort_values(by=['team','Date'],ascending=True)
st.write('after sorting ',df_new)

st.write('sort out team names', df_new['team'].unique())
st.write('calculate rolling avg home pts scored and home pts allowed')