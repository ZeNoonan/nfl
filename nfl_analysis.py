import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
# import os
import base64 
import altair as alt
import datetime as dt
# from st_aggrid import AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

st.set_page_config(layout="wide")

finished_week=18

@st.cache
def read_data(file):
    return pd.read_excel(file)

@st.cache
def read_csv_data(file):
    return pd.read_csv(file)

# Run this once below 
odds_data_excel = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_betting_odds_current.xlsx')
def csv_save(x):
    x.to_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_odds.csv')
    return x
csv_save(odds_data_excel)
# odds_data = read_csv_data('C:/Users/Darragh/Documents/Python/NFL/nfl_odds.csv').copy()
# odds_data = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_betting_odds_current.xlsx').copy()
# st.write(odds_data)
odds_data = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_odds.csv').copy()

# https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/
team_names_id = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_teams.csv').copy()
# st.write(team_names_id)
# team_names_id=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_teams.csv')


url='https://www.pro-football-reference.com/years/2021/games.htm'


def fbref_scraper_csv(url):
        test = pd.read_html(url)[0]
        # test.to_excel('C:/Users/Darragh/Documents/Python/NFL/nfl_2021_scores.xlsx')
        test.to_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_2021.csv')
        # test.to_csv('https://github.com/ZeNoonan/nfl/blob/main/nfl_2021.csv')
        return test

fbref_scraper_csv(url)

prior_nfl_data = pd.read_csv('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_2020.csv')
# prior_nfl_data=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_2020.csv')

data_2021=pd.read_csv('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_2021.csv')
# data_2021=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_2021.csv')
# st.write('check data', data_2021)

def clean_csv(x):
    # x['Date']=pd.to_datetime(x['Date'])
    x['year'] = x['Date'].str[:4]
    x['month'] = x['Date'].str[5:7]
    x['day'] = x['Date'].str[8:10]
    x['Date'] = pd.to_datetime(x[['year','month','day']])
    return x

nfl_data=data_2021.copy()
# st.markdown(get_table_download_link(data_2021), unsafe_allow_html=True)

# with st.beta_expander('Historical odds function'):
odds_data=odds_data.loc[:,['Date','Home Team','Away Team','Home Score','Away Score','Home Line Close']].copy()
# st.write('odds data before datetime', odds_data)
odds_data['Date']=pd.to_datetime(odds_data['Date']).dt.normalize()
odds_data['year']=odds_data['Date'].dt.year
odds_data['month']=odds_data['Date'].dt.month
odds_data['day']=odds_data['Date'].dt.day

# odds_data['Date'] = pd.to_datetime([dt.datetime.strftime(d, "%Y-%m-%d %H:%M") for d in odds_data["Date"]])
# st.write('pre odds', odds_data.dtypes)
team_names_id=team_names_id.rename(columns={'Team':'Home Team'})
odds_data=pd.merge(odds_data,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'}).sort_values(by='Date',ascending=False)
team_names_id=team_names_id.rename(columns={'Home Team':'Away Team'})
odds_data=pd.merge(odds_data,team_names_id,on='Away Team').rename(columns={'ID':'Away ID','Home Score':'Home Points',
'Away Score':'Away Points','Home Line Close':'Spread'}).sort_values(by='Date',ascending=False)
odds_data['Spread']=pd.to_numeric(odds_data['Spread'])
# st.write('odds', odds_data)


# with st.beta_expander('Pro Football Function'):
def clean_pro_football_pickle(nfl_data):
    nfl_data=nfl_data.rename(columns={'Unnamed: 5':'at_venue'})
    nfl_data['Home Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Loser/tie'],nfl_data['Winner/tie'])
    nfl_data['at_venue']=nfl_data['at_venue'].replace({np.nan:'stay'})
    nfl_data['Away Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Winner/tie'],nfl_data['Loser/tie'])
    # nfl_data['Home Points']=np.where(nfl_data['at_venue']=='@',nfl_data['Pts.1'],nfl_data['Pts'])
    nfl_data['Home Points']=np.where(nfl_data['at_venue']=='@',nfl_data['PtsL'],nfl_data['PtsW'])
    nfl_data['Away Points']=np.where(nfl_data['at_venue']=='@',nfl_data['PtsW'],nfl_data['PtsL'])
    nfl_data['home_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOL'],nfl_data['TOW']))
    nfl_data['away_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOW'],nfl_data['TOL']))
    # nfl_data=nfl_data[nfl_data['Week'].str.contains('Week')==False].copy()
    nfl_data=nfl_data[nfl_data['Week'].str.contains('Week',na=False)==False].copy()
    nfl_data=nfl_data.dropna(subset=["Week"])
    nfl_data['home_turnover']=pd.to_numeric(nfl_data['home_turnover'])
    nfl_data['away_turnover']=pd.to_numeric(nfl_data['away_turnover'])
    nfl_data['Home Points']=pd.to_numeric(nfl_data['Home Points'])
    nfl_data['Away Points']=pd.to_numeric(nfl_data['Away Points'])
    nfl_data['Date']=pd.to_datetime(nfl_data['Date'])
    nfl_data['Week'] = nfl_data['Week'].replace({'WildCard':18,'Division':19,'ConfChamp':20,'SuperBowl':21})
    nfl_data['Week']=pd.to_numeric(nfl_data['Week'])
    fb_ref_2020=nfl_data.loc[:,['Week','Day','Date','Time','Home Team', 'Away Team', 'Home Points','Away Points','home_turnover','away_turnover']]
    fb_ref_2020['Turnover'] = fb_ref_2020['home_turnover'] - fb_ref_2020['away_turnover']
    team_names_id_1=team_names_id.rename(columns={'Away Team':'Home Team'})
    fb_ref_2020=pd.merge(fb_ref_2020,team_names_id_1,on='Home Team').rename(columns={'ID':'Home ID'})
    fb_ref_2020=pd.merge(fb_ref_2020,team_names_id,on='Away Team').rename(columns={'ID':'Away ID'})
    season_pro = pd.merge(fb_ref_2020,odds_data,on=['Date','Home Team','Away Team', 'Home Points','Away Points', 'Home ID','Away ID'], how='left')
    return season_pro

def clean_pro_football_pickle_2021(nfl_data):
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
    nfl_data['Week'] = nfl_data['Week'].replace({'WildCard':18,'Division':19,'ConfChamp':20,'SuperBowl':21})
    nfl_data['Week']=pd.to_numeric(nfl_data['Week'])
    fb_ref_2020=nfl_data.loc[:,['Week','Day','Date','Time','Home Team', 'Away Team', 'Home Points','Away Points','home_turnover','away_turnover']]
    fb_ref_2020['Turnover'] = fb_ref_2020['home_turnover'] - fb_ref_2020['away_turnover']
    team_names_id_1=team_names_id.rename(columns={'Away Team':'Home Team'})
    fb_ref_2020=pd.merge(fb_ref_2020,team_names_id_1,on='Home Team').rename(columns={'ID':'Home ID'})
    fb_ref_2020=pd.merge(fb_ref_2020,team_names_id,on='Away Team').rename(columns={'ID':'Away ID'})

    fb_ref_2020['year']=fb_ref_2020['Date'].dt.year
    fb_ref_2020['month']=fb_ref_2020['Date'].dt.month
    fb_ref_2020['day']=fb_ref_2020['Date'].dt.day

    odds_data_updated=odds_data.drop(['Home Points', 'Away Points'], axis=1)
    # st.write('odds data before merge', odds_data_updated)
    # st.write(odds_data_updated.dtypes)
    # st.write('fb before merge', fb_ref_2020)
    # st.write('fbreft', fb_ref_2020.dtypes)
    
    # season_pro = pd.merge(fb_ref_2020,odds_data_updated,on=['Date','Home Team','Away Team', 'Home ID','Away ID'], how='left')
    season_pro = pd.merge(fb_ref_2020,odds_data_updated,on=['Date','year','month','day','Home Team','Away Team', 'Home ID','Away ID'], how='left')
    # st.write('after merge', season_pro)
    # st.write(season_pro.dtypes)
    return season_pro

def clean_prior_year(x):
    # x['Week']=x['Week'].replace({18:0,19:0,20:0,21:0,17:-1,16:-2,15:-3})
    x['Week']=x['Week'].replace({18:0,19:0,20:0,21:0,17:0,16:-1,15:-2,14:-3})
    x=x[x['Week'].between(-3,0)].copy()
    x=x.reset_index().drop('index',axis=1)
    # st.write('Check for errors',x[x['Away ID'].isna()])
    return x

def concat_current_prior(x,y):
    current_plus_prior = pd.concat([x,y],axis=0,ignore_index=True)
    return current_plus_prior
    
def test_clean_pro_football_pickle(nfl_data):
    nfl_data=nfl_data.rename(columns={'Unnamed: 5':'at_venue'})
    nfl_data['Home Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Loser/tie'],nfl_data['Winner/tie'])
    nfl_data['at_venue']=nfl_data['at_venue'].replace({np.nan:'stay'})
    nfl_data['Away Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Winner/tie'],nfl_data['Loser/tie'])
    nfl_data['Home Points']=np.where(nfl_data['at_venue']=='@',nfl_data['Pts.1'],nfl_data['Pts'])
    nfl_data['Away Points']=np.where(nfl_data['at_venue']=='@',nfl_data['Pts'],nfl_data['Pts.1'])
    nfl_data['home_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOL'],nfl_data['TOW']))
    nfl_data['away_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOW'],nfl_data['TOL']))
    nfl_data=nfl_data[~nfl_data['Week'].str.contains('Week',na=False)].copy()
    nfl_data=nfl_data.dropna(subset=["Week"])
    nfl_data['home_turnover']=pd.to_numeric(nfl_data['home_turnover'])
    nfl_data['away_turnover']=pd.to_numeric(nfl_data['away_turnover'])
    nfl_data['Home Points']=pd.to_numeric(nfl_data['Home Points'])
    nfl_data['Away Points']=pd.to_numeric(nfl_data['Away Points'])
    nfl_data['Date']=pd.to_datetime(nfl_data['Date'])
    nfl_data['Week'] = nfl_data['Week'].replace({'WildCard':18,'Division':19,'ConfChamp':20,'SuperBowl':21})
    nfl_data['Week']=pd.to_numeric(nfl_data['Week'])
    fb_ref_2020=nfl_data.loc[:,['Week','Day','Date','Time','Home Team', 'Away Team', 'Home Points','Away Points','home_turnover','away_turnover']]
    fb_ref_2020['Turnover'] = fb_ref_2020['home_turnover'] - fb_ref_2020['away_turnover']
    team_names_id_1=team_names_id.rename(columns={'Away Team':'Home Team'})
    fb_ref_2020=pd.merge(fb_ref_2020,team_names_id_1,on='Home Team').rename(columns={'ID':'Home ID'})
    fb_ref_2020=pd.merge(fb_ref_2020,team_names_id,on='Away Team').rename(columns={'ID':'Away ID'})
    season_pro = pd.merge(fb_ref_2020,odds_data,on=['Date','Home Team','Away Team', 'Home Points','Away Points', 'Home ID','Away ID'], how='left')
    return season_pro


current=clean_pro_football_pickle_2021(nfl_data)
prior_data = clean_prior_year(clean_pro_football_pickle(prior_nfl_data))
# st.write( prior_data[(prior_data['Home Team']=='Miami Dolphins') | (prior_data['Away Team']=='Miami Dolphins')].sort_values(by=['Week','Date','Time']) )

data = concat_current_prior(current,prior_data)

def spread_workings(data):
    data['home_win']=data['Home Points'] - data['Away Points']
    data['home_win'] = np.where((data['Home Points'] > data['Away Points']), 1, np.where((data['Home Points'] < data['Away Points']),-1,0))
    data['home_cover']=(np.where(((data['Home Points'] + data['Spread']) > data['Away Points']), 1,
    np.where(((data['Home Points']+ data['Spread']) < data['Away Points']),-1,0)))
    data['home_cover']=data['home_cover'].astype(int)
    data['away_cover'] = -data['home_cover']
    data=data.rename(columns={'Net Turnover':'home_turnover'})
    data['away_turnover'] = -data['home_turnover']
    return data

def season_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

spread=spread_workings(data)

def season_cover_workings(data,home,away,name,week_start):
    season_cover_df=data[data['Week']>week_start].copy()
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_cover_df = (season_cover_df.loc[:,['Week','Date','Home ID',home]]).rename(columns={'Home ID':'ID',home:name})
    # st.write('checking home turnover section', home_cover_df[home_cover_df['ID']==0])
    away_cover_df = (season_cover_df.loc[:,['Week','Date','Away ID',away]]).rename(columns={'Away ID':'ID',away:name})
    # st.write('checking away turnover section', away_cover_df[away_cover_df['ID']==0])
    season_cover=pd.concat([home_cover_df,away_cover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

def turnover_2(season_cover_df):    
    # https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group
    season_cover_df['prev_turnover']=season_cover_df.groupby('ID')['turned_over_sign'].shift()
    return season_cover_df.sort_values(by=['ID','Week'],ascending=True)

def turnover_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data

def turnover_workings(data,week_start):
    turnover_df=data[data['Week']>week_start].copy()
    turnover_df['home_turned_over_sign'] = np.where((turnover_df['Turnover'] > 0), 1, np.where((turnover_df['Turnover'] < 0),-1,0))
    turnover_df['away_turned_over_sign'] = - turnover_df['home_turned_over_sign']
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_turnover_df = (turnover_df.loc[:,['Week','Date','Home ID','home_turned_over_sign']]).rename(columns={'Home ID':'ID','home_turned_over_sign':'turned_over_sign'})
    # st.write('checking home turnover section', home_turnover_df[home_turnover_df['ID']==0])
    away_turnover_df = (turnover_df.loc[:,['Week','Date','Away ID','away_turned_over_sign']]).rename(columns={'Away ID':'ID','away_turned_over_sign':'turned_over_sign'})
    # st.write('checking away turnover section', away_turnover_df[away_turnover_df['ID']==0])
    season_cover=pd.concat([home_turnover_df,away_turnover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

# with st.beta_expander('Last Game Turnover'):
turnover=spread_workings(data)
turnover_1 = turnover_workings(turnover,-1)
turnover_2=turnover_2(turnover_1)
turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')
    # st.write(turnover_3.sort_values(by=['ID','Week'],ascending=['True','True']))

def season_cover_2(season_cover_df,column_name):    
    # https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
    # season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].transform(lambda x: x.cumsum().shift())
    # THE ABOVE DIDN'T WORK IN 2020 PRO FOOTBALL BUT DID WORK IN 2019 DO NOT DELETE FOR INFO PURPOSES
    season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].apply(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

# with st.beta_expander('Season to date Cover'):
spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',0)
spread_2=season_cover_2(spread_1,'cover')
spread_3=season_cover_3(spread_2,'cover_sign','cover')
    # st.write(spread_3.sort_values(by=['ID','Week'],ascending=['True','True']))

matrix_df=spread_workings(data)
matrix_df=matrix_df.reset_index().rename(columns={'index':'unique_match_id'})
test_df = matrix_df.copy()
# st.write('check for unique match id', test_df)
matrix_df['at_home'] = 1
matrix_df['at_away'] = -1
matrix_df['home_pts_adv'] = -3
matrix_df['away_pts_adv'] = 3
matrix_df['away_spread']=-matrix_df['Spread']
matrix_df=matrix_df.rename(columns={'Spread':'home_spread'})
matrix_df_1=matrix_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv','Date','Time','Home Points','Away Points']].copy()

# with st.beta_expander('Games Played to be used in Matrix Multiplication'):
first_qtr=matrix_df_1.copy()
start=-3
finish=0
first_4=first_qtr[first_qtr['Week'].between(start,finish)].copy()
def games_matrix_workings(first_4):
    group_week = first_4.groupby('Week')
    raw_data_2=[]
    game_weights = iter([-0.125, -0.25,-0.5,-1])
    for name, group in group_week:
        group['game_adj']=next(game_weights)
        # st.write('looking at for loop',group)
        raw_data_2.append(group)

    df3 = pd.concat(raw_data_2, ignore_index=True)
    adj_df3=df3.loc[:,['Home ID', 'Away ID', 'game_adj']].copy()
    test_adj_df3 = adj_df3.rename(columns={'Home ID':'Away ID', 'Away ID':'Home ID'})
    concat_df_test=pd.concat([adj_df3,test_adj_df3]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
    test_concat_df_test=concat_df_test.groupby('Home ID')['game_adj'].sum().abs().reset_index()
    test_concat_df_test['Away ID']=test_concat_df_test['Home ID']
    full=pd.concat([concat_df_test,test_concat_df_test]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
    full_stack=pd.pivot_table(full,index='Away ID', columns='Home ID',aggfunc='sum')
    # st.write('Check sum looks good all zero', full_stack.sum())
    full_stack=full_stack.fillna(0)
    full_stack.columns = full_stack.columns.droplevel(0)
    return full_stack
full_stack=games_matrix_workings(first_4)



# with st.beta_expander('CORRECT Testing reworking the DataFrame'):
test_df['at_home'] = 1
test_df['at_away'] = -1
test_df['home_pts_adv'] = 3
test_df['away_pts_adv'] = -3
test_df['away_spread']=-test_df['Spread']
test_df=test_df.rename(columns={'Spread':'home_spread'})
test_df_1=test_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()
test_df_home=test_df_1.loc[:,['Week','Home ID','at_home','home_spread','home_pts_adv']].rename(columns={'Home ID':'ID','at_home':'home','home_spread':'spread','home_pts_adv':'home_pts_adv'}).copy()
test_df_away=test_df_1.loc[:,['Week','Away ID','at_away','away_spread','away_pts_adv']].rename(columns={'Away ID':'ID','at_away':'home','away_spread':'spread','away_pts_adv':'home_pts_adv'}).copy()
test_df_2=pd.concat([test_df_home,test_df_away],ignore_index=True)
test_df_2=test_df_2.sort_values(by=['ID','Week'],ascending=True)
test_df_2['spread_with_home_adv']=test_df_2['spread']+test_df_2['home_pts_adv']
    # st.write(test_df_2)

def test_4(matrix_df_1):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    matrix_df_1['adj_spread']=matrix_df_1['spread_with_home_adv'].rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    return matrix_df_1


# with st.beta_expander('CORRECT Power Ranking to be used in Matrix Multiplication'):
# # https://stackoverflow.com/questions/9621362/how-do-i-compute-a-weighted-moving-average-using-pandas
grouped = test_df_2.groupby('ID')
# https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence
# https://stackoverflow.com/questions/62471485/is-it-possible-to-insert-missing-sequence-numbers-in-python
ranking_power=[]
for name, group in grouped:
    dfseq = pd.DataFrame.from_dict({'Week': range( -3,21 )}).merge(group, on='Week', how='outer').fillna(np.NaN)
    dfseq['ID']=dfseq['ID'].fillna(method='ffill')
    dfseq['home_pts_adv']=dfseq['home_pts_adv'].fillna(0)
    dfseq['spread']=dfseq['spread'].fillna(0)
    dfseq['spread_with_home_adv']=dfseq['spread_with_home_adv'].fillna(0)
    dfseq['home']=dfseq['home'].fillna(0)
    df_seq_1 = dfseq.groupby(['Week','ID'])['spread_with_home_adv'].sum().reset_index()
    update=test_4(df_seq_1)
    ranking_power.append(update)
df_power = pd.concat(ranking_power, ignore_index=True)
    # st.write('power ranking',df_power.sort_values(by=['ID','Week'],ascending=[True,True]))




# with st.beta_expander('CORRECT Power Ranking Matrix Multiplication'):
# https://stackoverflow.com/questions/62775018/matrix-array-multiplication-whats-excel-doing-mmult-and-how-to-mimic-it-in#62775508
inverse_matrix=[]
power_ranking=[]
list_inverse_matrix=[]
list_power_ranking=[]
power_df=df_power.loc[:,['Week','ID','adj_spread']].copy()

games_df=matrix_df_1.copy()
# st.write('Checking the games df', games_df[((games_df['Home ID']==24)|(games_df['Away ID']==24))])
first=list(range(-3,18))
last=list(range(0,21))
for first,last in zip(first,last):
    first_section=games_df[games_df['Week'].between(first,last)]
    full_game_matrix=games_matrix_workings(first_section)
    adjusted_matrix=full_game_matrix.loc[0:30,0:30]
    df_inv = pd.DataFrame(np.linalg.pinv(adjusted_matrix.values), adjusted_matrix.columns, adjusted_matrix.index)
    power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID')\
    .drop('Week',axis=1).rename(columns={'adj_spread':0}).loc[:30,:]
    result = df_inv.dot(pd.DataFrame(power_df_week))
    result.columns=['power']
    avg=(result['power'].sum())/32
    result['avg_pwr_rank']=(result['power'].sum())/32
    result['final_power']=result['avg_pwr_rank']-result['power']
    df_pwr=pd.DataFrame(columns=['final_power'],data=[avg])
    result=pd.concat([result,df_pwr],ignore_index=True)
    result['week']=last+1
    power_ranking.append(result)
power_ranking_combined = pd.concat(power_ranking).reset_index().rename(columns={'index':'ID'})
# st.write('power ranking combined', power_ranking_combined)
    
# with st.beta_expander('Adding Power Ranking to Matches'):
matches_df = spread.copy()
home_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Home ID'})
away_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Away ID'})
updated_df=pd.merge(matches_df,home_power_rank_merge,on=['Home ID','Week']).rename(columns={'final_power':'home_power'})
updated_df=pd.merge(updated_df,away_power_rank_merge,on=['Away ID','Week']).rename(columns={'final_power':'away_power'})
updated_df['calculated_spread']=updated_df['away_power']-updated_df['home_power']
updated_df['spread_working']=updated_df['home_power']-updated_df['away_power']+updated_df['Spread']
updated_df['power_pick'] = np.where(updated_df['spread_working'] > 0, 1,
np.where(updated_df['spread_working'] < 0,-1,0))
    # st.write(updated_df.sort_values(by='Week'))

with st.expander('Season to Date Cover Graph'):
    st.write('Positive number means the number of games to date that you have covered the spread; in other words teams with a positive number have beaten expectations')
    st.write('Negative number means the number of games to date that you have not covered the spread; in other words teams with a negative number have performed below expectations')
    st.write('blanks in graph are where the team got a bye week')
    # df = pd.DataFrame([['mon',19,'cardinals', 3], ['tue',20,'patriots', 4], ['wed',20,'patriots', 5]], columns=['date','week','team', 'stdc'])
    # st.write('df1',df)
    # df2 = pd.DataFrame([['sun',18,'saints'], ['tue',20,'patriots'], ['wed',20,'patriots']], columns=['date','week','team'])
    # st.write('df2',df2)
    # df3=df2.merge(df,on=['date','week','team'], how='left')
    # st.write('merged on left',df3)  # merges on columns A

    stdc_home=spread_3.rename(columns={'ID':'Home ID'})
    stdc_home['cover_sign']=-stdc_home['cover_sign']
    stdc_away=spread_3.rename(columns={'ID':'Away ID'})
    updated_df=updated_df.drop(['away_cover'],axis=1)
    updated_df=updated_df.rename(columns={'home_cover':'home_cover_result'})
    updated_df=updated_df.merge(stdc_home,on=['Date','Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign':'home_cover_sign'})
    updated_df=pd.merge(updated_df,stdc_away,on=['Date','Week','Away ID'],how='left').rename(columns={'cover':'away_cover','cover_sign':'away_cover_sign'})
    # st.write('check that STDC coming in correctly', updated_df)
    # st.write('Check Total')
    # st.write('home',updated_df['home_cover_sign'].sum())
    # st.write('away',updated_df['away_cover_sign'].sum())
    # st.write('Updated for STDC', updated_df)
    # st.write('Get STDC by Week do something similar for Power Rank')
    # last_occurence = spread_3.groupby(['ID'],as_index=False).last()
    # st.write(last_occurence)
    stdc_df=pd.merge(spread_3,team_names_id,on='ID').rename(columns={'Away Team':'Team'})
    team_names_id_update=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    df_stdc_1=pd.merge(spread_3,team_names_id_update,on='ID').rename(columns={'Away Team':'Team'})
    df_stdc_1=df_stdc_1.loc[:,['Week','ID','Team','cover']].copy()
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['cover'].transform(np.mean)
    
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)

    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

    
with st.expander('Turnover Factor by Match Graph'):
    st.write('-1 means you received more turnovers than other team, 1 means you gave up more turnovers to other team')
    # st.write('this is turnovers', turnover_3)
    turnover_matches = turnover_3.loc[:,['Date','Week','ID','prev_turnover', 'turnover_sign']].copy()
    turnover_home=turnover_matches.rename(columns={'ID':'Home ID'})
    
    turnover_away=turnover_matches.rename(columns={'ID':'Away ID'})
    turnover_away['turnover_sign']=-turnover_away['turnover_sign']
    updated_df=pd.merge(updated_df,turnover_home,on=['Date','Week','Home ID'],how='left').rename(columns={'prev_turnover':'home_prev_turnover','turnover_sign':'home_turnover_sign'})
    updated_df=pd.merge(updated_df,turnover_away,on=['Date','Week','Away ID'],how='left').rename(columns={'prev_turnover':'away_prev_turnover','turnover_sign':'away_turnover_sign'})
    # st.write()
    df_stdc_1=pd.merge(turnover_matches,team_names_id_update,on='ID').rename(columns={'Away Team':'Team'})
    df_stdc_1['average']=df_stdc_1.groupby('Team')['turnover_sign'].transform(np.mean)

    color_scale = alt.Scale(domain=[1,0,-1],range=["red", "lightgrey","LimeGreen"])

    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=alt.Scale(scheme='redyellowgreen')))
    alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=color_scale))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    
    text_cover=chart_cover.mark_text().encode(text=alt.Text('turnover_sign:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

with st.expander('Betting Slip Matches'):
    betting_matches=updated_df.loc[:,['Week','Date','Home ID','Home Team','Away ID', 'Away Team','Spread','Home Points','Away Points',
    'home_power','away_power','home_cover','away_cover','home_turnover_sign','away_turnover_sign','home_cover_sign','away_cover_sign','power_pick','home_cover_result']]
    betting_matches['total_factor']=betting_matches['home_turnover_sign']+betting_matches['away_turnover_sign']+betting_matches['home_cover_sign']+\
    betting_matches['away_cover_sign']+betting_matches['power_pick']
    betting_matches['bet_on'] = np.where(betting_matches['total_factor']>2,betting_matches['Home Team'],np.where(betting_matches['total_factor']<-2,betting_matches['Away Team'],''))
    betting_matches['bet_sign'] = (np.where(betting_matches['total_factor']>2,1,np.where(betting_matches['total_factor']<-2,-1,0)))
    betting_matches['bet_sign'] = betting_matches['bet_sign'].astype(float)
    betting_matches['home_cover'] = betting_matches['home_cover'].astype(float)
    betting_matches['result']=betting_matches['home_cover_result'] * betting_matches['bet_sign']
    st.write('testing sum of betting result',betting_matches['result'].sum())

    # this is for graphing anlaysis on spreadsheet
    betting_matches['bet_sign_all'] = (np.where(betting_matches['total_factor']>0,1,np.where(betting_matches['total_factor']<-0,-1,0)))
    betting_matches['result_all']=betting_matches['home_cover_result'] * betting_matches['bet_sign_all']
    st.write('testing sum of betting all result',betting_matches['result_all'].sum())
    cols_to_move=['Week','Date','Home Team','Away Team','total_factor','bet_on','result','Spread','Home Points','Away Points','home_power','away_power']
    cols = cols_to_move + [col for col in betting_matches if col not in cols_to_move]
    betting_matches=betting_matches[cols]
    betting_matches=betting_matches.sort_values('Date')
    presentation_betting_matches=betting_matches.copy()

    # https://towardsdatascience.com/7-reasons-why-you-should-use-the-streamlit-aggrid-component-2d9a2b6e32f0
    grid_height = st.number_input("Grid height", min_value=400, value=550, step=100)
    gb = GridOptionsBuilder.from_dataframe(presentation_betting_matches)
    gb.configure_column("Spread", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("home_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("away_power", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=1, aggFunc='sum')
    gb.configure_column("Date", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='dd-MM-yyyy', pivot=True)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)

    test_cellsytle_jscode = JsCode("""
    function(params) {
        if (params.value < 0) {
        return {
            'color': 'red',
        }
        } else {
            return {
                'color': 'black',
            }
        }
    };
    """)
    # # https://github.com/PablocFonseca/streamlit-aggrid/blob/main/st_aggrid/grid_options_builder.py
    gb.configure_column(field="Spread", cellStyle=test_cellsytle_jscode)
    gb.configure_column("home_power", cellStyle=test_cellsytle_jscode)
    gb.configure_column("away_power", cellStyle=test_cellsytle_jscode)


    # gb.configure_pagination()
    # gb.configure_side_bar()
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    grid_response = AgGrid(
        presentation_betting_matches, 
        gridOptions=gridOptions,
        height=grid_height, 
        width='100%',
        # data_return_mode=return_mode_value, 
        # update_mode=update_mode_value,
        # fit_columns_on_grid_load=fit_columns_on_grid_load,
        allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        enable_enterprise_modules=True,
    )

    # container.grid_response
    # AgGrid(betting_matches.sort_values('Date').style.format({'home_power':"{:.1f}",'away_power':"{:.1f}"}))
    
    # st.write( betting_matches[(betting_matches['Home Team']=='Arizona Cardinals') | 
    # (betting_matches['Away Team']=='Arizona Cardinals')].set_index('Week').sort_values(by='Date') )

with st.expander('Power Ranking by Week'):
    power_week=power_ranking_combined.copy()
    team_names_id=team_names_id.rename(columns={'Away Team':'Team'})
    id_names=team_names_id.drop_duplicates(subset=['ID'], keep='first')
    pivot_df=pd.merge(power_week,id_names, on='ID')
    # st.write('after merge', pivot_df)
    pivot_df=pivot_df.loc[:,['Team','final_power','week']].copy()
    # st.write('graphing?',pivot_df)
    power_pivot=pd.pivot_table(pivot_df,index='Team', columns='week')
    pivot_df_test = pivot_df.copy()
    pivot_df_test=pivot_df_test[pivot_df_test['week']<19]
    pivot_df_test['average']=pivot_df.groupby('Team')['final_power'].transform(np.mean)
    # st.write('graphing?',pivot_df_test)
    power_pivot.columns = power_pivot.columns.droplevel(0)
    power_pivot['average'] = power_pivot.mean(axis=1)
    # st.write(power_pivot)
    # https://stackoverflow.com/questions/67045668/altair-text-over-a-heatmap-in-a-script
    pivot_df=pivot_df.sort_values(by='final_power',ascending=False)
    chart_power= alt.Chart(pivot_df_test).mark_rect().encode(alt.X('week:O',axis=alt.Axis(title='week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('final_power:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text=chart_power.mark_text().encode(text=alt.Text('final_power:N',format=",.0f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)
    # https://github.com/altair-viz/altair/issues/820#issuecomment-386856394

with st.expander('Power Pick Factor by Team'):
    st.write('Positive number means the market has undervalued the team as compared to the spread')
    st.write('Negative number means the market has overvalued the team as compared to the spread')    
    power_factor=betting_matches.loc[:,['Week','Home Team','Away Team','power_pick']].rename(columns={'power_pick':'home_power_pick'})
    power_factor['away_power_pick']=-power_factor['home_power_pick']
    home_factor=power_factor.loc[:,['Week','Home Team','home_power_pick']].rename(columns={'Home Team':'Team','home_power_pick':'power_pick'})
    away_factor=power_factor.loc[:,['Week','Away Team','away_power_pick']].rename(columns={'Away Team':'Team','away_power_pick':'power_pick'})
    graph_power_pick=pd.concat([home_factor,away_factor],axis=0).sort_values(by=['Week'])
    graph_power_pick['average']=graph_power_pick.groupby('Team')['power_pick'].transform(np.mean)

    color_scale = alt.Scale(domain=[1,0,-1],range=["LimeGreen", "lightgrey","red"])

    chart_cover= alt.Chart(graph_power_pick).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    # alt.Y('Team',sort=alt.SortField(field='average', order='ascending')),color=alt.Color('turnover_sign:Q',scale=alt.Scale(scheme='redyellowgreen')))
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('power_pick:Q',scale=color_scale))
    text_cover=chart_cover.mark_text().encode(text=alt.Text('power_pick:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)
    # st.write('graph',graph_power_pick)
    # st.write('data',power_factor)


with st.expander('Analysis of Betting Results across 1 to 5 factors'):
    matches_in_regular_season= (32 * 16) / 2
    # st.write('In 2020 there were 13 matches in playoffs looks like this was new so 269 total matches in 2020 season compared with 267 in previous seasons')
    matches_in_playoffs = 13
    total_matches =matches_in_regular_season + matches_in_playoffs
    # st.write('total_matches per my calculation',total_matches)
    analysis=betting_matches.copy()
    analysis=analysis[analysis['Week']<finished_week+1]
    # st.write('analysis',analysis)
    totals = analysis.groupby('total_factor').agg(winning=('result_all','count'))
    totals_graph=totals.reset_index().rename(columns={'winning':'number_of_games'})

    # st.write('totals grpah', totals_graph)
    chart_power= alt.Chart(totals_graph).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='total_factor_per_match',labelAngle=0)),
    alt.Y('number_of_games'))
    text=chart_power.mark_text(dy=-7).encode(text=alt.Text('number_of_games:N',format=",.0f"),color=alt.value('black'))
    st.altair_chart(chart_power + text,use_container_width=True)
    
    totals_1=analysis.groupby([analysis['total_factor'].abs(),'result_all']).agg(winning=('result_all','count')).reset_index()
    totals_1['result_all']=totals_1['result_all'].replace({0:'tie',1:'win',-1:'lose'})
    # st.write('checking graph data',totals_1.dtypes)
    # totals_1['total_factor']=totals_1['total_factor'].astype(str)
    # st.write('checking graph data',totals_1.dtypes)
    # st.write('checking graph data',totals_1)
    # https://www.quackit.com/css/css_color_codes.cfm
    color_scale = alt.Scale(
    domain=[
        "lose",
        "tie",
        "win"],
        range=["red", "lightgrey","LimeGreen"])
    chart_power= alt.Chart(totals_1).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning'),color=alt.Color('result_all',scale=color_scale))
    # st.altair_chart(chart_power,use_container_width=True)

    
    normalized_table = (totals_1[totals_1['result_all']!='tie']).copy()
    # st.write('graph date to be cleaned',totals_1)
    chart_power= alt.Chart(normalized_table).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning',stack="normalize"),color=alt.Color('result_all',scale=color_scale))
    overlay = pd.DataFrame({'winning': [0.5]})
    vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2).encode(y='winning:Q')
    text = alt.Chart(normalized_table).mark_text(dx=-1, dy=+60, color='white').encode(
    x=alt.X('total_factor:O'),
    y=alt.Y('winning',stack="normalize"),
    detail='winning',
    text=alt.Text('winning:Q', format='.0f'))
    updated_test_chart=chart_power+vline+text
    
    st.altair_chart(updated_test_chart,use_container_width=True)

    # st.write('shows the number of games at each factor level')
    # st.write(totals.rename(columns={'winning':'number_of_games'}))
    # st.write('sum of each factor level should correspond to table above',totals_1)
    # st.write('sum of winning column should be 267 I think',totals_1['winning'].sum())
    # st.write('count of week column should be 267',analysis['Week'].count())

    reset_data=totals_1.copy()
    reset_data['result_all']=reset_data['result_all'].replace({'tie':0,'win':1,'lose':-1})
    reset_data=reset_data.pivot(index='result_all',columns='total_factor',values='winning').fillna(0)
    reset_data['betting_factor_total']=reset_data[3]+reset_data[4]+reset_data[5]
    reset_data=reset_data.sort_values(by='betting_factor_total',ascending=False)

    reset_data=reset_data.reset_index()
    # st.write('reset data', reset_data)
    reset_data['result_all']=reset_data['result_all'].astype(str)
    reset_data=reset_data.set_index('result_all')

    reset_data.loc['Total']=reset_data.sum()
    reset_data.loc['No. of Bets Made'] = reset_data.loc[['1','-1']].sum() 
    reset_data=reset_data.apply(pd.to_numeric, downcast='integer')
    reset_data.loc['% Winning'] = ((reset_data.loc['1'] / reset_data.loc['No. of Bets Made'])*100)
    st.write('This shows the betting result')
    st.write(reset_data)
    st.write('Broken down by the number of factors indicating the strength of the signal')

with st.expander('Analysis of Factors'):
    analysis_factors = betting_matches.copy()
    analysis_factors=analysis_factors[analysis_factors['Week']<finished_week+1]
    def analysis_factor_function(analysis_factors):
        analysis_factors['home_turnover_success?'] = analysis_factors['home_turnover_sign'] * analysis_factors['home_cover_result']
        analysis_factors['away_turnover_success?'] = analysis_factors['away_turnover_sign'] * analysis_factors['home_cover_result']
        analysis_factors['home_cover_season_success?'] = analysis_factors['home_cover_sign'] * analysis_factors['home_cover_result']  
        analysis_factors['away_cover_season_success?'] = analysis_factors['away_cover_sign'] * analysis_factors['home_cover_result']
        analysis_factors['power_ranking_success?'] = analysis_factors['power_pick'] * analysis_factors['home_cover_result']
        df_table = analysis_factors['home_turnover_success?'].value_counts()
        away_turnover=analysis_factors['away_turnover_success?'].value_counts()
        home_cover=analysis_factors['home_cover_season_success?'].value_counts()
        away_cover=analysis_factors['away_cover_season_success?'].value_counts()
        power=analysis_factors['power_ranking_success?'].value_counts()
        df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1)
        # df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1).reset_index().drop('index',axis=1)
        # st.write('df table', df_table_1)
        # test=df_table_1.reset_index()
        # st.write(test)
        df_table_1['total_turnover'] = df_table_1['home_turnover_success?'].add (df_table_1['away_turnover_success?'])
        # st.write(test)
        df_table_1['total_season_cover'] = df_table_1['home_cover_season_success?'] + df_table_1['away_cover_season_success?']
        # st.write('df table 2', df_table_1)

        df_table_1=df_table_1.reset_index()
        # st.write('reset data', df_table_1)
        df_table_1['index']=df_table_1['index'].astype(str)
        df_table_1=df_table_1.set_index('index')


        df_table_1.loc['Total']=df_table_1.sum()
        # st.write('latest', df_table_1)
        # st.write('latest', df_table_1.shape)
        if df_table_1.shape > (2,7):
            # st.write('Returning df with analysis')
            df_table_1.loc['No. of Bets Made'] = df_table_1.loc[['1','-1']].sum() # No losing bets so far!!!
            df_table_1.loc['% Winning'] = ((df_table_1.loc['1'] / df_table_1.loc['No. of Bets Made'])*100)
        else:
            # st.write('Returning df with no analysis')
            return df_table_1
        return df_table_1
    total_factor_table = analysis_factor_function(analysis_factors)   
    st.write('This is the total number of matches broken down by Factor result')
    cols_to_move=['total_turnover','total_season_cover','power_ranking_success?']
    total_factor_table = total_factor_table[ cols_to_move + [ col for col in total_factor_table if col not in cols_to_move ] ]
    total_factor_table=total_factor_table.loc[:,['total_turnover','total_season_cover','power_ranking_success?']]
    st.write(total_factor_table)
    factor_bets = (analysis_factors[analysis_factors['bet_sign']!=0]).copy()
    bets_made_factor_table = analysis_factor_function(factor_bets)
    # cols_to_move=['total_turnover','total_season_cover','power_ranking_success?']
    bets_made_factor_table = bets_made_factor_table[ cols_to_move + [ col for col in bets_made_factor_table if col not in cols_to_move ] ]
    bets_made_factor_table=bets_made_factor_table.loc[:,['total_turnover','total_season_cover','power_ranking_success?']]
    st.write('This is the matches BET ON broken down by Factor result')
    st.write(bets_made_factor_table)

    # st.write('graph work below')
    graph_factor_table = total_factor_table.copy().loc[['-1','0','1'],:].reset_index().rename(columns={'index':'result_all'})
    graph_factor_table['result_all']=graph_factor_table['result_all'].replace({'0':'tie','1':'win','-1':'lose'})
    graph_factor_table=graph_factor_table.melt(id_vars='result_all',var_name='total_factor',value_name='winning')
    chart_power= alt.Chart(graph_factor_table).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning'),color=alt.Color('result_all',scale=color_scale))
    # alt.Y('winning'),color=alt.Color('result_all'))
    # st.write('do the normalised stacked bar chart which shows percentage')
    # st.altair_chart(chart_power,use_container_width=True)

    normalized_table = graph_factor_table.copy()
    normalized_table=normalized_table[normalized_table['result_all']!='tie']
    normalized_table= normalized_table[(normalized_table['total_factor']=='total_turnover') | (normalized_table['total_factor']=='total_season_cover')
     | (normalized_table['total_factor']=='power_ranking_success?')].copy()
    chart_power= alt.Chart(normalized_table).mark_bar().encode(alt.X('total_factor:O',axis=alt.Axis(title='factor',labelAngle=0)),
    alt.Y('winning',stack="normalize"),color=alt.Color('result_all',scale=color_scale))
    overlay = pd.DataFrame({'winning': [0.5]})
    vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2).encode(y='winning:Q')
    
    
    text = alt.Chart(normalized_table).mark_text(dx=-1, dy=+37, color='white').encode(
    x=alt.X('total_factor:O'),
    y=alt.Y('winning',stack="normalize"),
    detail='winning',
    text=alt.Text('winning:Q', format='.0f'))
    
    # chart_power=chart_power+text

    # updated_test_chart = alt.layer(chart_power,vline)
    updated_test_chart=chart_power+vline+text
    
    st.altair_chart(updated_test_chart,use_container_width=True)


    # st.write(graph_factor_table)

with st.expander('Checking Performance where Total Factor = 2 or 3:  Additional Diagnostic'):
    df_factor = betting_matches.copy()
    two_factor_df = df_factor[df_factor['total_factor'].abs()==2]
    # st.write(two_factor_df)
    factor_2_3_home_turnover_filter = (df_factor['total_factor']==2)&(df_factor['home_turnover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['home_turnover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['home_turnover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['home_turnover_sign']==-1)

    factor_2_3_away_turnover_filter = (df_factor['total_factor']==2)&(df_factor['away_turnover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['away_turnover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['away_turnover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['away_turnover_sign']==-1)

    factor_2_3_home_cover_filter = (df_factor['total_factor']==2)&(df_factor['home_cover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['home_cover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['home_cover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['home_cover_sign']==-1)

    factor_2_3_away_cover_filter = (df_factor['total_factor']==2)&(df_factor['away_cover_sign']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['away_cover_sign']==1) | (df_factor['total_factor']==3)&(df_factor['away_cover_sign']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['away_cover_sign']==-1)

    factor_2_3_power_filter = (df_factor['total_factor']==2)&(df_factor['power_pick']==-1) | \
    (df_factor['total_factor']==-2)&(df_factor['power_pick']==1) | (df_factor['total_factor']==3)&(df_factor['power_pick']==1) | \
    (df_factor['total_factor']==-3)&(df_factor['power_pick']==-1)

    df_factor['home_turnover_diagnostic'] = (df_factor['home_turnover_sign'].where(factor_2_3_home_turnover_filter)) * df_factor['home_cover_result']
    df_factor['away_turnover_diagnostic'] = (df_factor['away_turnover_sign'].where(factor_2_3_away_turnover_filter)) * df_factor['home_cover_result']
    df_factor['home_cover_diagnostic'] = (df_factor['home_cover_sign'].where(factor_2_3_home_cover_filter)) * df_factor['home_cover_result']
    df_factor['away_cover_diagnostic'] = (df_factor['away_cover_sign'].where(factor_2_3_away_cover_filter)) * df_factor['home_cover_result']
    df_factor['power_diagnostic'] = (df_factor['power_pick'].where(factor_2_3_power_filter)) * df_factor['home_cover_result']
    # st.write(df_factor)

    df_factor_table = df_factor['home_turnover_diagnostic'].value_counts()
    away_turnover=df_factor['away_turnover_diagnostic'].value_counts()
    home_cover=df_factor['home_cover_diagnostic'].value_counts()
    away_cover=df_factor['away_cover_diagnostic'].value_counts()
    power=df_factor['power_diagnostic'].value_counts()
    df_factor_table_1=pd.concat([df_factor_table,away_turnover,home_cover,away_cover,power],axis=1)
    df_factor_table_1['total_turnover'] = df_factor_table_1['home_turnover_diagnostic'].add (df_factor_table_1['away_turnover_diagnostic'])
    # st.write(test)
    df_factor_table_1['total_season_cover'] = df_factor_table_1['home_cover_diagnostic'] + df_factor_table_1['away_cover_diagnostic']
    # st.write('df table 2', df_factor_table_1)

    df_factor_table_1=df_factor_table_1.reset_index()
    # st.write('reset data', df_factor_table_1)
    df_factor_table_1['index']=df_factor_table_1['index'].astype(int)
    df_factor_table_1['index']=df_factor_table_1['index'].astype(str)
    df_factor_table_1=df_factor_table_1.set_index('index')


    df_factor_table_1.loc['Total']=df_factor_table_1.sum()
    # st.write('latest', df_factor_table_1)
    # st.write('latest', df_factor_table_1.shape)

    if df_factor_table_1.shape > (2,7):
        df_factor_table_1.loc['No. of Bets Made'] = df_factor_table_1.loc[['1','-1']].sum() 
        df_factor_table_1.loc['% Winning'] = ((df_factor_table_1.loc['1'] / df_factor_table_1.loc['No. of Bets Made'])*100)
    # else:
    #     # st.write('Returning df with no anal')
    #     return df_factor_table_1


    cols_to_move=['total_turnover','total_season_cover','power_diagnostic']
    df_factor_table_1 = df_factor_table_1[ cols_to_move + [ col for col in df_factor_table_1 if col not in cols_to_move ] ]
    df_factor_table_1=df_factor_table_1.loc[:,['total_turnover','total_season_cover','power_diagnostic']]
    st.write(df_factor_table_1.style.format('{:.0f}',subset=['total_turnover','total_season_cover','power_diagnostic']))


with st.expander('Underdog Analyis'):
    underdog_df = betting_matches.copy()
    filter_bets_underdog=(underdog_df['Spread']>0.1) &(underdog_df['bet_sign']!=0)
    filter_bets_favourite=(underdog_df['Spread']<0.1) &(underdog_df['bet_sign']!=0)
    underdog_df['home_underdog_bet_result']=underdog_df['result'].where(filter_bets_underdog)
    underdog_df['home_favourite_bet_result']=underdog_df['result'].where(filter_bets_favourite)
    underdog_df['away_underdog_bet_result']=underdog_df['result'].where(filter_bets_favourite)*-1
    underdog_df['away_favourite_bet_result']=underdog_df['result'].where(filter_bets_underdog)*-1
    underdog_df['home_underdog_all_result']=underdog_df['home_cover_result'].where(underdog_df['Spread']>0.1)
    underdog_df['home_favourite_all_result']=underdog_df['home_cover_result'].where(underdog_df['Spread']<0.1)
    underdog_df['away_favourite_all_result']=(underdog_df['home_cover_result'].where(underdog_df['Spread']>0.1))*-1
    underdog_df['away_underdog_all_result']=(underdog_df['home_cover_result'].where(underdog_df['Spread']<0.1))*-1
    underdog_table = underdog_df['home_underdog_bet_result'].value_counts()
    away_underdog_bet = underdog_df['away_underdog_bet_result'].value_counts()
    home_fav_bet = underdog_df['home_favourite_bet_result'].value_counts()
    away_fav_bet = underdog_df['away_favourite_bet_result'].value_counts()
    underdog_results = pd.concat([underdog_table,away_underdog_bet,home_fav_bet,away_fav_bet],axis=1)
    underdog_results=underdog_results.sort_index(ascending=False)
    underdog_results['underdog']=underdog_results['home_underdog_bet_result']+underdog_results['away_underdog_bet_result']
    underdog_results['favourite']=underdog_results['home_favourite_bet_result']+underdog_results['away_favourite_bet_result']

    underdog_results=underdog_results.reset_index()
    # st.write('reset data', underdog_results)
    underdog_results['index']=underdog_results['index'].astype(int)
    underdog_results['index']=underdog_results['index'].astype(str)
    underdog_results=underdog_results.set_index('index')


    underdog_results.loc['Total']=underdog_results.sum()
    # st.write('underdog', underdog_results)
    # st.write('underdog', underdog_results.shape)
    if underdog_results.shape > (2,6):
        underdog_results.loc['No. of Bets Made'] = underdog_results.loc[['1','-1']].sum() 
        underdog_results.loc['% Winning'] = underdog_results.loc['1'] / underdog_results.loc['No. of Bets Made']
    cols_to_move=['underdog','favourite']
    underdog_results = underdog_results[ cols_to_move + [ col for col in underdog_results if col not in cols_to_move ] ]
    st.write('This shows the total number of BETS made and whether it was an underdog or favourite that covered')
    st.write('not sure if this has value or not')
    st.write(underdog_results)

    home_underdog_all = underdog_df['home_underdog_all_result'].value_counts()
    away_underdog_all = underdog_df['away_underdog_all_result'].value_counts()
    home_fav_all = underdog_df['home_favourite_all_result'].value_counts()
    away_fav_all = underdog_df['away_favourite_all_result'].value_counts()
    all_results = pd.concat([home_underdog_all,away_underdog_all,home_fav_all,away_fav_all],axis=1)
    all_results=all_results.sort_index(ascending=False)
    all_results['underdog']=all_results['home_underdog_all_result']+all_results['away_underdog_all_result']
    all_results['favourite']=all_results['home_favourite_all_result']+all_results['away_favourite_all_result']

    all_results=all_results.reset_index()
    # st.write('reset data', all_results)
    all_results['index']=all_results['index'].astype(int)
    all_results['index']=all_results['index'].astype(str)
    all_results=all_results.set_index('index')


    all_results.loc['Total']=all_results.sum()
    # st.write('checking',all_results)
    # st.write(all_results.shape)
    if all_results.shape > (2,6):
        all_results.loc['No. of Bets Made'] = all_results.loc[['1','-1']].sum() 
        all_results.loc['% Winning'] = all_results.loc['1'] / all_results.loc['No. of Bets Made']
    cols_to_move=['underdog','favourite']
    all_results = all_results[ cols_to_move + [ col for col in all_results if col not in cols_to_move ] ]
    st.write(all_results)

# test
    
    



# with st.beta_expander('Pro Football Ref Scraper'):
    pass
    # def fbref_scraper():
    #     test = pd.read_html('https://www.pro-football-reference.com/years/2021/games.htm')[0]
    #     test.to_pickle('C:/Users/Darragh/Documents/Python/NFL/pro_football_ref/nfl_2021.pkl')
    #     return test  
        
    # test=fbref_scraper()

with st.expander('Tests'):
    st.write('To Check that all ok with odds data')
    st.write(odds_data[odds_data['Away ID'].isna()])

    def test_clean_prior_year(x):
        x['Week']=x['Week'].replace({18:0,19:0,20:0,21:0,17:0,16:-1,15:-2,14:-3})
        x=x[x['Week'].between(-3,0)].copy()
        x=x.reset_index().drop('index',axis=1)
        st.write('Check for errors',x[x['Away ID'].isna()])
        return x


    # test_prior=clean_pro_football_pickle(prior_nfl_data)
    # test_prior_data = test_clean_prior_year(clean_pro_football_pickle(prior_nfl_data))

    def pre_season(data_2021):
        # not sure if this even works, think its for pre-season
        data_2021=data_2021.rename(columns={'VisTm':'Winner/tie','HomeTm':'Loser/tie','Unnamed: 2':'Date'})
        data_2021['month']=data_2021['Date'].str.split(' ').str[0]
        data_2021['date_in_month']=data_2021['Date'].str.split(' ').str[1]
        data_2021['year']=2021
        data_2021['TOW']=0
        data_2021['TOL']=0
        data_2021=data_2021.set_index('Week').drop(['Pre0','Pre1','Pre2','Pre3','Week'],axis=0).reset_index()
        data_2021['Week']=pd.to_numeric(data_2021['Week'])
        data_2021['year']=np.where(data_2021['Week']>16,2022,2021)
        data_2021['Date']=pd.to_datetime(data_2021['year'].astype(str) + data_2021['month']+ data_2021['date_in_month'].astype(str),format='%Y%B%d')
        data_2021.loc['Week','Week']='Week'
        return data_2021
    
    st.write('Check sum if True all good', full_stack.sum().sum()==0)

