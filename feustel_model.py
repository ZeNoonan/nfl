from operator import is_
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

# df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_24_09_22.xlsx')
df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_14_10_22.xlsx')
df=df.copy()

with st.expander('pro football workings'):
    year=2006
    url=f'https://www.pro-football-reference.com/years/{year}/games.htm'
    # st.write('url', url)

    def fbref_scraper_csv(url):
            test = pd.read_html(url)[0]
            # test.to_excel('C:/Users/Darragh/Documents/Python/NFL/nfl_2022_scores.xlsx')
            test.to_csv(f'C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_{year}.csv')
            # test.to_csv('https://github.com/ZeNoonan/nfl/blob/main/nfl_2021.csv')
            return test

    # def read_csv_data(file):
    # file_2006 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2006.csv')
    # file_2007 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2007.csv')
    # file_2008 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2008.csv')
    # file_2009 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2009.csv')
    # file_2010 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2010.csv')
    # file_2011 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2011.csv')
    # file_2012 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2012.csv')
    # file_2013 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2013.csv')
    # file_2014 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2014.csv')
    # file_2015 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2015.csv')
    # file_2016 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2016.csv')
    # file_2017 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2017.csv')
    # file_2018 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2018.csv')
    # file_2019 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2019.csv')
    # file_2020 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2020.csv')
    # file_2021 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2021.csv')
    # file_2022 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2022.csv')
    # combined_file = pd.concat([file_2006,file_2007,file_2008,file_2009,file_2010,file_2011,file_2012,file_2013,
    # file_2014,file_2015,file_2016,file_2017,file_2018,file_2019,file_2020,file_2021])
    # combined_file.to_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2006_2021.csv')
    combined_file=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2006_2021.csv')
    st.write('comb files', combined_file)

    def clean_pro_football_pickle_2021(nfl_data):
    # sourcery skip: inline-immediately-returned-variable
        nfl_data=nfl_data.rename(columns={'Unnamed: 5':'at_venue'})
        nfl_data['Home Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Loser/tie'],nfl_data['Winner/tie'])
        nfl_data['at_venue']=nfl_data['at_venue'].replace({np.nan:'stay'})
        nfl_data['Away Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Winner/tie'],nfl_data['Loser/tie'])
        nfl_data['Home Points']=np.where(nfl_data['at_venue']=='@',nfl_data['PtsL'],nfl_data['PtsW'])
        nfl_data['Away Points']=np.where(nfl_data['at_venue']=='@',nfl_data['PtsW'],nfl_data['PtsL'])
        nfl_data['home_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOL'],nfl_data['TOW']))
        nfl_data['away_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOW'],nfl_data['TOL']))
        
        nfl_data=nfl_data[nfl_data['Week'].str.contains('Week',na=False)==False].copy()
        nfl_data=nfl_data.dropna(subset=["Week"])
        nfl_data['home_turnover']=pd.to_numeric(nfl_data['home_turnover'])
        nfl_data['away_turnover']=pd.to_numeric(nfl_data['away_turnover'])
        nfl_data['Home Points']=pd.to_numeric(nfl_data['Home Points'])
        nfl_data['Away Points']=pd.to_numeric(nfl_data['Away Points'])
        nfl_data['Date']=pd.to_datetime(nfl_data['Date']).dt.normalize()
        # nfl_data['Date'] = pd.to_datetime([dt.datetime.strftime(d, "%Y-%m-%d %H:%M") for d in nfl_data["Date"]])
        # nfl_data['Week'] = nfl_data['Week'].replace({'WildCard':18,'Division':19,'ConfChamp':20,'SuperBowl':21})
        nfl_data['Week'] = nfl_data['Week'].replace({'WildCard':19,'Division':20,'ConfChamp':21,'SuperBowl':22})
        nfl_data['Week']=pd.to_numeric(nfl_data['Week'])
        nfl_data['year'] = pd.DatetimeIndex(nfl_data['Date']).year
        nfl_data['month'] = pd.DatetimeIndex(nfl_data['Date']).month
        nfl_data['season_month'] = nfl_data['month'].map({9:1,10:2,11:3,12:4,1:5,2:6})
        # NL_Raw_Clean['calendar_month']=NL_Raw_Clean['Per.'].map({1:9,2:10,3:11,4:12,5:1,6:2,7:3,8:4,9:5,10:6,11:7,12:8,19:8})
        nfl_data['season_year'] = np.where((nfl_data['season_month'] < 5), nfl_data['year'], nfl_data['year']-1)
        nfl_data['turnover']=nfl_data['home_turnover']-nfl_data['away_turnover']
        return nfl_data

    cleaned_pro_football_file=clean_pro_football_pickle_2021(combined_file)
    st.write('cleaned file', cleaned_pro_football_file)




df['Home Line Close']=df['Home Line Close'].fillna(df['Home Line Open'])
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['season_month'] = df['month'].map({9:1,10:2,11:3,12:4,1:5,2:6})
# NL_Raw_Clean['calendar_month']=NL_Raw_Clean['Per.'].map({1:9,2:10,3:11,4:12,5:1,6:2,7:3,8:4,9:5,10:6,11:7,12:8,19:8})
df['season_year'] = np.where((df['season_month'] < 5), df['year'], df['year']-1)
df['Home Team']=df['Home Team'].replace({'Washington Football Team':'Washington Commanders','Washington Redskins':'Washington Commanders','St. Louis Rams':'Los Angeles Rams',
'Oakland Raiders':'Las Vegas Raiders','San Diego Chargers':'Los Angeles Chargers'})
df['Away Team']=df['Away Team'].replace({'Washington Football Team':'Washington Commanders','Washington Redskins':'Washington Commanders','St. Louis Rams':'Los Angeles Rams',
'Oakland Raiders':'Las Vegas Raiders','San Diego Chargers':'Los Angeles Chargers'})
df=df.sort_values(by=['Date','Home Team']).reset_index().drop('index',axis=1)
df=df.reset_index().rename(columns={'index':'unique_id'})

# df=df.sort_values(by=['unique_id'],ascending=False)



df['avg_home_score']=df['Home Score'].expanding().mean()
df['avg_away_score']=df['Away Score'].expanding().mean()
cols_to_move=['Date','Home Team','Away Team','unique_id','Home Score','Away Score','avg_home_score','avg_away_score']
cols = cols_to_move + [col for col in df if col not in cols_to_move]
df=df[cols]

with st.expander('raw data'):
    st.write(df)
# NL_Raw_Clean['calendar_year']=NL_Raw_Clean['calendar_year']+2000
# NL_Raw_Clean=NL_Raw_Clean.rename(columns={'calendar_year':'year', 'calendar_month':'month'})

# df['']
# st.write(df.sort_values(by='Date'))
# for _ in df.groupby('season_year'):
#     pass

df_offensive_home=df.loc[:,['Date','Home Team', 'Home Score', 'season_year','unique_id','avg_home_score','avg_away_score','Home Line Close']].rename(columns={'Home Team':'team','Home Score':'score'})
df_offensive_home['home_away']=1
df_offensive_away=df.loc[:,['Date','Away Team','Away Score', 'season_year','unique_id','avg_home_score','avg_away_score','Home Line Close']].rename(columns={'Away Team':'team','Away Score':'score'})
df_offensive_away['home_away']=-1
df_offensive=pd.concat([df_offensive_home,df_offensive_away],axis=0).sort_values(by=['team','Date'],ascending=True).reset_index().drop('index',axis=1)
# df_groupby_scores=df_offensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum().reset_index().drop('level_2',axis=1)
# df_offensive['sum_score']=df_offensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum()\
#     .reset_index().drop(['level_2','team','season_year'],axis=1)
# df_offensive['mean_score']=df_offensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).mean()\
#     .reset_index().drop(['level_2','team','season_year'],axis=1)
df_offensive['avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean()\
    .reset_index().drop(['level_2','team','season_year'],axis=1)

# st.write(df_offensive.groupby(['team','season_year'])['score'].shift().expanding(min_periods=4).mean().shift()\
#     .reset_index())

def col_correction(df_offensive,col='avg_pts_scored_team_season'):
    df_offensive['SHIFT avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean().shift()\
        .reset_index().drop(['level_2','team','season_year'],axis=1)
    df_offensive['test_col']=np.where(df_offensive[col].isna(),np.NaN,np.where(df_offensive['SHIFT avg_pts_scored_team_season'].isna(),np.NaN,1))
    df_offensive[col]=df_offensive['SHIFT avg_pts_scored_team_season']*df_offensive['test_col']
    df_offensive=df_offensive.drop(['test_col','SHIFT avg_pts_scored_team_season'],axis=1)
    return df_offensive

df_offensive=col_correction(df_offensive,col='avg_pts_scored_team_season')
# st.write('df off', df_offensive)
# st.write('check to see if shift worked',df_offensive[(df_offensive['team']=='Arizona Cardinals') | (df_offensive['team']=='Arizona Cardinals')])
df_offensive=df_offensive.rename(columns={'score':'pts_scored','mean_score':'4_game_pts_scored'}).sort_values(by=['team','Date'])

df_defensive_home=df.loc[:,['Date','Home Team', 'Away Score', 'season_year','unique_id','avg_home_score','avg_away_score','Home Line Close']]\
    .rename(columns={'Home Team':'team','Away Score':'score'})
df_defensive_home['home_away']=1
df_defensive_away=df.loc[:,['Date','Away Team','Home Score', 'season_year','unique_id','avg_home_score','avg_away_score','Home Line Close']].rename(columns={'Away Team':'team','Home Score':'score'})
df_defensive_away['home_away']=-1
df_defensive=pd.concat([df_defensive_home,df_defensive_away],axis=0).sort_values(by=['team','Date'],ascending=True).reset_index().drop('index',axis=1)
# df_groupby_scores=df_defensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum().reset_index().drop('level_2',axis=1)
# df_defensive['sum_score']=df_defensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).sum()\
#     .reset_index().drop(['level_2','team','season_year'],axis=1)
# df_defensive['mean_score']=df_defensive.groupby(['team','season_year'])['score'].rolling(window=4,min_periods=4, center=False).mean()\
#     .reset_index().drop(['level_2','team','season_year'],axis=1)
df_defensive['avg_pts_conceded_team_season']=df_defensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean()\
    .reset_index().drop(['level_2','team','season_year'],axis=1)

df_defensive=col_correction(df_defensive,col='avg_pts_conceded_team_season')
# st.write('check to see if shift worked defensive',df_defensive[(df_defensive['team']=='Arizona Cardinals') | (df_defensive['team']=='Arizona Cardinals')])
# df_defensive['SHIFT avg_pts_conceded_team_season']=df_defensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean().shift()\
#     .reset_index().drop(['level_2','team','season_year'],axis=1)
# df_defensive['test_col']=np.where(df_defensive['avg_pts_conceded_team_season'].isna(),np.NaN,np.where(df_defensive['SHIFT avg_pts_conceded_team_season'].isna(),np.NaN,1))
# df_defensive['avg_pts_conceded_team_season']=df_defensive['SHIFT avg_pts_conceded_team_season']*df_defensive['test_col']


df_defensive=df_defensive.rename(columns={'score':'pts_conceded','mean_score':'4_game_pts_conceded'}).sort_values(by=['team','Date'])

# st.write('df offensive 1', df_offensive)
# st.write('df defence 1', df_defensive)
df_new=pd.merge(df_offensive,df_defensive,how='outer')
# st.write('after merge', df_new)
df_new['team_cum_sum_pts']=df_new.groupby(['team'])['pts_scored'].cumsum()
df_new['team_cum_sum_games']=df_new.groupby(['team'])['pts_scored'].cumcount()+1
df_new['rolling_avg_team_pts_scored']=df_new['team_cum_sum_pts'] / df_new['team_cum_sum_games']
df_new=df_new.sort_values(by=['Date','unique_id','team'])
# st.write('before rolling avg date', df_new)
df_new['date_avg_pts_rolling']=df_new['pts_scored'].expanding().mean().shift(32) # 16 teams by 2, want the previous week numbers
# df_new['date_avg_pts_rolling_test']=df_new['pts_scored'].expanding().mean().shift(32)
# st.write('df new after date avg pts rolling', df_new)
# st.write('just checking the average pts scored in every match', df_new)
# st.write('sorted by date avg score by date',df_new)
df_new=df_new.sort_values(by=['team','Date'],ascending=True)

df_new=df_new.sort_values(by=['home_away','Date','unique_id','team'],ascending=True)
# st.write('after sorting CHECK THIS OUT',df_new)
df_home=df_new[df_new['home_away']==1].sort_values(by=['Date','unique_id'],ascending=True)
df_home['home_pts_avg']=df_home['pts_scored'].expanding().mean().shift(16) # want to get previous week points
# st.write('home avg points', df_home)
df_away=df_new[df_new['home_away']==-1].sort_values(by=['Date','unique_id'],ascending=True)
df_away['away_pts_avg']=df_away['pts_scored'].expanding().mean().shift(16)
df_new=pd.concat([df_home,df_away],ignore_index=True)
# df_new['home_pts_avg_']=df_new['pts_scored'].expanding().mean()
# df_new['away_pts_avg_']=df_new['pts_scored'].expanding().mean()



df_new=df_new.sort_values(by=['unique_id','home_away'],ascending=[True,False])
# st.write('df before', df_new)
# df_new['away_pts_avg']=df_new['away_pts_avg'].shift(-1)
df_new['avg_away_score']=df_new['avg_away_score'].fillna(method='ffill')
df_new['avg_home_score']=df_new['avg_home_score'].fillna(method='ffill')
df_new['home_adv']=df_new['avg_home_score']-df_new['avg_away_score']
cols_to_move=['Date','team','season_year','Home Line Close','unique_id','pts_scored','pts_conceded','home_adv','date_avg_pts_rolling','avg_pts_scored_team_season',
'avg_pts_conceded_team_season','home_pts_avg','away_pts_avg','avg_home_score','avg_away_score','home_away']
cols = cols_to_move + [col for col in df_new if col not in cols_to_move]
df_new=df_new[cols]
df_new=df_new.loc[:,['Date','team','season_year','unique_id','Home Line Close','pts_scored','pts_conceded','home_adv','date_avg_pts_rolling','avg_pts_scored_team_season',
'avg_pts_conceded_team_season','home_away']]
# st.write('df update', df_new)

# st.write('df after concat', df_new.sort_values(by=['home_away','Date'],ascending=True))

# st.write('sort out team names', df_new['team'].unique())
# st.write('checking rolling team scores', df_new.sort_values(by=['team','Date']))
# st.write('just checking the home adv calc keep it there to sense check', df_new.sort_values(by=['Date','unique_id','team']))
df_home_1=df_new[df_new['home_away']==1].rename(columns={'pts_scored':'home_pts_scored','pts_conceded':'home_pts_conceded','team':'home_team',
'avg_pts_scored_team_season':'home_avg_pts_scored_team_season','avg_pts_conceded_team_season':'home_avg_pts_conceded_team_season','date_avg_pts_rolling':'home_date_avg_pts_rolling'})\
    .set_index(['unique_id']).drop('home_away',axis=1).copy()
df_away_1=df_new[df_new['home_away']==-1].rename(columns={'pts_scored':'away_pts_scored','pts_conceded':'away_pts_conceded','team':'away_team',
'avg_pts_scored_team_season':'away_avg_pts_scored_team_season','avg_pts_conceded_team_season':'away_avg_pts_conceded_team_season','date_avg_pts_rolling':'away_date_avg_pts_rolling'})\
    .set_index(['unique_id']).drop(['home_adv','home_away'],axis=1).copy()
# st.write('df home', df_home_1, 'away', df_away_1)
# df_combined=pd.concat([df_home_1,df_away_1],axis=0)
# st.write('before home and away are combined', df_home_1, 'away', df_away_1)
df_combined=pd.merge(df_home_1.reset_index(),df_away_1.reset_index(),on=['unique_id','Date','season_year', 'Home Line Close'],how='outer')
# st.write('before calcs are done', df_combined)
df_combined['away_defensive_rating']=df_combined['away_avg_pts_conceded_team_season'] / df_combined['away_date_avg_pts_rolling']
df_combined['projected_team_a_pts']= df_combined['home_avg_pts_scored_team_season'] * df_combined['away_defensive_rating']
df_combined['home_defensive_rating']=df_combined['home_avg_pts_conceded_team_season'] / df_combined['away_date_avg_pts_rolling']
df_combined['projected_team_b_pts']= df_combined['away_avg_pts_scored_team_season'] * df_combined['home_defensive_rating']
df_combined['projected_team_a_pts']= df_combined['projected_team_a_pts'] + (df_combined['home_adv']/2)
df_combined['projected_team_b_pts']= df_combined['projected_team_b_pts'] - (df_combined['home_adv']/2)
df_combined['proj_spread'] = df_combined['projected_team_b_pts'] - df_combined['projected_team_a_pts']
df_combined['bet_sign']=np.where(df_combined['proj_spread']>df_combined['Home Line Close'],-1,np.where(df_combined['proj_spread']<df_combined['Home Line Close'],1,0))
df_combined['home_win']=df_combined['home_pts_scored'] - df_combined['away_pts_scored']
df_combined['home_win'] = np.where((df_combined['home_pts_scored'] > df_combined['away_pts_scored']), 1, np.where((df_combined['home_pts_scored'] < df_combined['away_pts_scored']),-1,0))
df_combined['home_cover']=(np.where(((df_combined['home_pts_scored'] + df_combined['Home Line Close']) > df_combined['away_pts_scored']), 1,
np.where(((df_combined['home_pts_scored']+ df_combined['Home Line Close']) < df_combined['away_pts_scored']),-1,0)))
df_combined['home_cover_result']=df_combined['home_cover'].astype(int)
df_combined['away_cover'] = -df_combined['home_cover']
df_combined['result']=df_combined['home_cover_result'] * df_combined['bet_sign']

cols_to_move=['Date','season_year','unique_id','home_team','away_team','home_pts_scored','away_pts_scored','Home Line Close','proj_spread',
'bet_sign','home_cover_result','result']
cols = cols_to_move + [col for col in df_combined if col not in cols_to_move]
df_combined=df_combined[cols]

# st.write('df_comb', df_combined[df_combined['season_year']==2022].set_index('Date'))
# st.download_button(label="Download data as CSV",data=df_combined[df_combined['season_year']==2021].to_csv().encode('utf-8'),file_name='df_spread.csv',mime='text/csv',key='after_merge_spread')

st.write('The home-away date avg pts rolling is the average points scored in every match so we can see what the avg pts scored and conceded is both will be same')
st.write('Database', df_combined)
result_count=df_combined.groupby(['season_year'])['result'].value_counts()
clean_df_for_pivot=df_combined.loc[:,['season_year','result','bet_sign']]
result_pivot=pd.pivot_table(clean_df_for_pivot, index='result', columns='season_year', aggfunc='count')
result_pivot.index=result_pivot.index.astype('str')
result_pivot.loc['total_games']=result_pivot.sum(axis=0)
result_pivot.loc['winning_%']=result_pivot.loc['1']/(result_pivot.loc['1']+result_pivot.loc['-1'])
result_pivot.loc['cum_winning_%']=result_pivot.loc['1'].cumsum() / (result_pivot.loc['1'].cumsum()+result_pivot.loc['-1'].cumsum())
# st.write(result_count.columns)
# result_count=result_count.reset_index()
# st.dataframe(result_count)
st.dataframe(result_pivot)

with st.expander('Average Error Calcs'):
    avg_error_data=df_combined.copy()
    avg_error_data=avg_error_data[avg_error_data['proj_spread'].notna()]
    # st.write('avg error df', avg_error_data)
    # avg_error_data['test']=avg_error_data['home_team']-avg_error_data['away_team']
    avg_error_data.insert(loc=8,column='betting_odds_error',value=(abs(avg_error_data['home_pts_scored']+avg_error_data['Home Line Close']-avg_error_data['away_pts_scored'])))
    avg_error_data.insert(loc=9,column='proj_odds_error',value=(abs(avg_error_data['home_pts_scored']+avg_error_data['proj_spread']-avg_error_data['away_pts_scored'])))
    # avg_error_data['avg_err_betting']=avg_error_data['betting_odds_error'].expanding().mean()
    avg_error_data_groupby=avg_error_data.groupby(['season_year'])['betting_odds_error','proj_odds_error'].mean()
    st.write('avg error df data', avg_error_data)
    st.write('avg error df groupby comparing the average error on bookies odds versus our internal projections', avg_error_data_groupby)

with st.expander('Splitting the Spread up to analyse'):
    avg_error_data_spread=avg_error_data.copy()
    st.write('d',avg_error_data_spread)
    decile_df=avg_error_data_spread.groupby(pd.qcut(avg_error_data_spread['Home Line Close'], 10))['result'].sum().reset_index()
    st.write(decile_df)
    # avg_error_data=avg_error_data[avg_error_data['proj_spread'].notna()]
