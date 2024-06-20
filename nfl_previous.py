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

season_picker = st.selectbox("Select a season to run",('season_2023','season_2022','season_2021'),index=0)
placeholder_1=st.empty()
placeholder_2=st.empty()

finished_week=21
last_week=22 # what is this for?? its for graphing i think NO its for power rank
number_of_teams=32

season_list={'season_2023': {
    "odds_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_odds_2023_2024.csv",
    "scores_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_scores_2023_2024.csv",
    "team_id": "C:/Users/Darragh/Documents/Python/NFL/nfl_teams_2023_2024.csv",
    "year":'2023',
    "prior_year_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_scores_2022_2023.csv"},
'season_2022': {
    "odds_file": "https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_odds_2022_2023.csv",
    "scores_file": "https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_scores_2022_2023.csv",
    "team_id": "https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_teams_2022_2023.csv",
    "year":'2022',
    "prior_year_file": "https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_scores_2021_2022.csv"},
'season_2021' : {
    "odds_file": "https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_odds_2021_2022.csv",
    "scores_file": "https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_scores_2021_2022.csv",
    "team_id": "https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_teams_2021_2022.csv",
    "year":'2021',
    "prior_year_file": 'https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_scores_2019_2020.csv'}}

# season_list={'season_2022': {
#     "odds_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_odds_2022_2023.csv",
#     "scores_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_scores_2022_2023.csv",
#     "team_id": "C:/Users/Darragh/Documents/Python/NFL/nfl_teams_2022_2023.csv",
#     "prior_year_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_scores_2021_2022.csv"},
# 'season_2021' : {
#     "odds_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_odds_2021_2022.csv",
#     "scores_file": "C:/Users/Darragh/Documents/Python/NFL/nfl_scores_2021_2022.csv",
#     "team_id": "C:/Users/Darragh/Documents/Python/NFL/nfl_teams_2021_2022.csv",
#     "prior_year_file": 'C:/Users/Darragh/Documents/Python/NFL/nfl_scores_2019_2020.csv'}}




# @st.cache
def read_data(file):
    return pd.read_excel(file)

# @st.cache
def read_csv_data(file):
    return pd.read_csv(file,parse_dates=['Date'])

# odds_data_excel = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds.xlsx')
odds_data_excel = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_data.xlsx')
def csv_save(x):
    x.to_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_odds_2023_2024.csv')
    return x
csv_save(odds_data_excel)

odds_data = read_csv_data(season_list[season_picker]['odds_file']).copy()
# odds_data = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_betting_odds_2021_2022.xlsx').copy()
# odds_data = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds.xlsx').copy()
# odds_data = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_odds.csv').copy()

# https://www.aussportsbetting.com/data/historical-nfl-results-and-odds-data/
# team_names_id = read_csv_data('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_teams.csv').copy()
# st.write(team_names_id)
# team_names_id=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_teams.csv')
team_names_id=pd.read_csv(season_list[season_picker]["team_id"])

year=season_list[season_picker]['year']
# url=f'https://www.pro-football-reference.com/years/{year}/games.htm'
url='https://www.pro-football-reference.com/years/2023/games.htm'
# st.write('url', pd.read_html(url)[0])

def fbref_scraper_csv(url):
        test = pd.read_html(url)[0]
        # test.to_excel('C:/Users/Darragh/Documents/Python/NFL/nfl_2022_scores.xlsx')
        test.to_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_scores_2023_2024.csv')
        # test.to_csv('https://github.com/ZeNoonan/nfl/blob/main/nfl_2021.csv')
        return test

# fbref_scraper_csv(url)

# prior_nfl_data = pd.read_csv('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_2020.csv')
# prior_nfl_data=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_2020.csv')
prior_nfl_data=pd.read_csv(season_list[season_picker]["prior_year_file"])

# data_2021=pd.read_csv('https://raw.githubusercontent.com/ZeNoonan/nfl/main/nfl_2021.csv')
# data_2021=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_2021.csv')
data_2022=pd.read_csv(season_list[season_picker]["scores_file"])

# st.write('check data', data_2021)

def clean_csv(x):
    # x['Date']=pd.to_datetime(x['Date'])
    x['year'] = x['Date'].str[:4]
    x['month'] = x['Date'].str[5:7]
    x['day'] = x['Date'].str[8:10]
    x['Date'] = pd.to_datetime(x[['year','month','day']])
    return x

# nfl_data=data_2021.copy()

def pre_season(data_2021,start_year=2022, end_year=2023):
    # not sure if this even works, think its for pre-season
    data_2021=data_2021.rename(columns={'VisTm':'Winner/tie','HomeTm':'Loser/tie','Unnamed: 2':'Date'})
    data_2021['month']=data_2021['Date'].str.split(' ').str[0]
    data_2021['date_in_month']=data_2021['Date'].str.split(' ').str[1]
    data_2021['year']=2022
    data_2021['TOW']=0
    data_2021['TOL']=0
    data_2021['PtsW']=0
    data_2021['PtsL']=0
    data_2021=data_2021[~data_2021['Week'].isin(['Pre0','Pre1','Pre2','Pre3','Week'])]
    # data_2021['Week']=pd.to_numeric(data_2021['Week'])
    data_2021['year']=np.where(data_2021['Week']>16,end_year,start_year)
    data_2021['Date']=pd.to_datetime(data_2021['year'].astype(str) + data_2021['month']+ data_2021['date_in_month'].astype(str),format='%Y%B%d')
    data_2021.loc['Week','Week']='Week'
    return data_2021

# st.write('wildcard', data_2022)
# data_2022=pre_season(data_2022)
nfl_data=data_2022.copy()
# st.write('nfl_data', nfl_data)
# st.markdown(get_table_download_link(data_2021), unsafe_allow_html=True)

# with st.beta_expander('Historical odds function'):
odds_data=odds_data.loc[:,['Date','Home Team','Away Team','Home Score','Away Score','Home Line Close']].copy()
# st.write('odds data before datetime', odds_data)
odds_data['Date']=pd.to_datetime(odds_data['Date'],format='mixed').dt.normalize()
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
    # sourcery skip: inline-immediately-returned-variable
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
    season_pro = pd.merge(fb_ref_2020,odds_data_updated,on=['Date','year','month','day','Home Team','Away Team', 'Home ID','Away ID'], how='left',indicator=True)
    # st.write('after merge', season_pro)
    # st.write(season_pro.dtypes)
    return season_pro

def clean_prior_year(x):
    x['Week']=x['Week'].replace({18:0,19:0,20:0,21:0,17:-1,16:-2,15:-3})
    # x['Week']=x['Week'].replace({18:0,19:0,20:0,21:0,17:0,16:-1,15:-2,14:-3})
    x=x[x['Week'].between(-3,0)].copy()
    x=x.reset_index().drop('index',axis=1)
    # st.write('Check for errors',x[x['Away ID'].isna()])
    return x

def concat_current_prior(x,y):
    # sourcery skip: inline-immediately-returned-variable
    current_plus_prior = pd.concat([x,y],axis=0,ignore_index=True)
    return current_plus_prior
    
def test_clean_pro_football_pickle(nfl_data):
    # sourcery skip: inline-immediately-returned-variable
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


def clean_pro_football_pickle_work_on_this_cover_preseason_as_well(nfl_data):
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
    season_pro = pd.merge(fb_ref_2020,odds_data_updated,on=['Date','year','month','day','Home Team','Away Team', 'Home ID','Away ID'], how='left',indicator=True)
    # st.write('after merge', season_pro)
    # st.write(season_pro.dtypes)
    return season_pro

current=clean_pro_football_pickle_2021(nfl_data)
# st.write('current line 315 looks ok',current.sort_values(by='Week'))
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
# st.write('looking ok? yes', spread)

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
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=[True,True,True])

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
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=[True,True,True])

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
    season_cover_df.loc[column_name] = season_cover_df.groupby (['ID'])[column_name].apply(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

def season_cover_2_updated(spead_1):  
    x=spread_1.groupby (['ID'])['cover'].apply(lambda x: x.cumsum().shift()).reset_index().rename(columns={'level_1':'unique_id'})
    # st.write('in function and applying shift',x)
    y=spread_1.drop('cover',axis=1).reset_index().rename(columns={'index':'unique_id'})
    # st.write('dropping cover in function',y.reset_index().rename(columns={'index':'unique_id'}))
    # st.write('dropping cover in function',y)
    season_cover_df=pd.merge(x,y,how='outer',on=['unique_id','ID'])
    # st.write('after merge', season_cover_df)
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop(['index','unique_id'],axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

def season_cover_2_updated_graph(spead_1):  
    x=spread_1.groupby (['ID'])['cover'].apply(lambda x: x.cumsum()).reset_index().rename(columns={'level_1':'unique_id'})
    y=spread_1.drop('cover',axis=1).reset_index().rename(columns={'index':'unique_id'})
    season_cover_df=pd.merge(x,y,how='outer',on=['unique_id','ID'])
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop(['index','unique_id'],axis=1)
    return season_cover_df

# with st.beta_expander('Season to date Cover'):
spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',0)
# st.write('before shift outside function', spread_1.reset_index())
# st.write('outside function apply shift',spread_1.groupby (['ID'])['cover'].apply(lambda x: x.cumsum().shift()).reset_index().rename(columns={'level_1':'index'}) )
spread_2=season_cover_2_updated(spread_1)
# st.write('after shift', spread_2)
spread_3=season_cover_3(spread_2,'cover_sign','cover')

spread_2_graph=season_cover_2_updated_graph(spread_1)
spread_3_graph=season_cover_3(spread_2_graph,'cover_sign','cover')
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
# st.write('spread row 371',test_df_2)

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
    dfseq = pd.DataFrame.from_dict({'Week': range( -3,22 )}).merge(group, on='Week', how='outer').fillna(np.NaN)
    dfseq['ID']=dfseq['ID'].fillna(method='ffill')
    dfseq['home_pts_adv']=dfseq['home_pts_adv'].fillna(0)
    dfseq['spread']=dfseq['spread'].fillna(0)
    dfseq['spread_with_home_adv']=dfseq['spread_with_home_adv'].fillna(0)
    dfseq['home']=dfseq['home'].fillna(0)
    df_seq_1 = dfseq.groupby(['Week','ID'])['spread_with_home_adv'].sum().reset_index()
    update=test_4(df_seq_1)
    ranking_power.append(update)
df_power = pd.concat(ranking_power, ignore_index=True)
# st.write('power ranking',df_power[df_power['ID']==12])
# st.write('power ranking',df_power.sort_values(by=['ID','Week'],ascending=[True,True]))




# with st.beta_expander('CORRECT Power Ranking Matrix Multiplication'):
# https://stackoverflow.com/questions/62775018/matrix-array-multiplication-whats-excel-doing-mmult-and-how-to-mimic-it-in#62775508
inverse_matrix=[]
power_ranking=[]
list_inverse_matrix=[]
list_power_ranking=[]
# st.write('power df', df_power)
power_df=df_power.loc[:,['Week','ID','adj_spread']].copy()

games_df=matrix_df_1.copy()
# st.write('games df', games_df)

# st.write('Checking the games df', games_df[((games_df['Home ID']==24)|(games_df['Away ID']==24))])

first=list(range(-3,last_week-3))
# sourcery skip: remove-zero-from-range
last=list(range(0,last_week))
for first,last in zip(first,last):
    first_section=games_df[games_df['Week'].between(first,last)]
    # st.write('first section', first_section)
    full_game_matrix=games_matrix_workings(first_section)
    # st.write('full game matrix', full_game_matrix)
    adjusted_matrix=full_game_matrix.loc[0:(number_of_teams-2),0:(number_of_teams-2)]
    # st.write('adjusted matrix', adjusted_matrix)
    df_inv = pd.DataFrame(np.linalg.pinv(adjusted_matrix.values), adjusted_matrix.columns, adjusted_matrix.index)
    # st.write('df inv', df_inv)
    power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID')\
    .drop('Week',axis=1).rename(columns={'adj_spread':0}).loc[:(number_of_teams-2),:]
    # st.write('power_df_week', power_df_week,'first', first, 'last', last)
    result = df_inv.dot(pd.DataFrame(power_df_week))
    # st.write('result', result)
    result.columns=['power']
    avg=(result['power'].sum())/number_of_teams
    result['avg_pwr_rank']=(result['power'].sum())/number_of_teams
    result['final_power']=result['avg_pwr_rank']-result['power']
    df_pwr=pd.DataFrame(columns=['final_power'],data=[avg])
    result=pd.concat([result,df_pwr],ignore_index=True)
    result['week']=last+1
    power_ranking.append(result)
power_ranking_combined = pd.concat(power_ranking).reset_index().rename(columns={'index':'ID'})
# st.write('power ranking combined', power_ranking_combined)

# first=list(range(-3,19))
# last=list(range(0,22))
# for first,last in zip(first,last):
#     power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID')
#     st.write('power_df_week', power_df_week)



# with st.beta_expander('Adding Power Ranking to Matches'):
matches_df = spread.copy()
# st.write('matches df OK line 551', matches_df)
home_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Home ID'})
away_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Away ID'})
updated_df=pd.merge(matches_df,home_power_rank_merge,on=['Home ID','Week']).rename(columns={'final_power':'home_power'})
updated_df=pd.merge(updated_df,away_power_rank_merge,on=['Away ID','Week']).rename(columns={'final_power':'away_power'})
updated_df['calculated_spread']=updated_df['away_power']-updated_df['home_power']
updated_df['spread_working']=updated_df['home_power']-updated_df['away_power']+updated_df['Spread']
updated_df['power_pick'] = np.where(updated_df['spread_working'] > 0, 1,
np.where(updated_df['spread_working'] < 0,-1,0))
# st.write('line 559 looking ok? NO',updated_df.sort_values(by='Week'))



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
    # st.write(spread_3)
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    df_stdc_1['average']=df_stdc_1.groupby('Team')['cover'].transform(np.mean)
    
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)

    df_stdc_1_graph=pd.merge(spread_3_graph,team_names_id_update,on='ID').rename(columns={'Away Team':'Team'})
    df_stdc_1_graph=df_stdc_1_graph.loc[:,['Week','ID','Team','cover']].copy()
    df_stdc_1_graph['average']=df_stdc_1_graph.groupby('Team')['cover'].transform(np.mean)

    chart_cover= alt.Chart(df_stdc_1).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

    chart_cover= alt.Chart(df_stdc_1_graph).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
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

with placeholder_2.expander('Betting Slip Matches'):
    # st.write('updated df', updated_df)
    betting_matches=updated_df.loc[:,['Week','Date','Home ID','Home Team','Away ID', 'Away Team','Spread','Home Points','Away Points',
    'home_power','away_power','home_cover','away_cover','home_turnover_sign','away_turnover_sign','home_cover_sign','away_cover_sign','power_pick',
    'home_cover_result']]
    betting_matches['total_factor']=betting_matches['home_turnover_sign']+betting_matches['away_turnover_sign']+betting_matches['home_cover_sign']+\
    betting_matches['away_cover_sign']+betting_matches['power_pick']
    betting_matches['bet_on'] = np.where(betting_matches['total_factor']>2,betting_matches['Home Team'],np.where(betting_matches['total_factor']<-2,betting_matches['Away Team'],''))
    betting_matches['bet_sign'] = (np.where(betting_matches['total_factor']>2,1,np.where(betting_matches['total_factor']<-2,-1,0)))
    betting_matches['bet_sign'] = betting_matches['bet_sign'].astype(float)
    betting_matches['home_cover'] = betting_matches['home_cover'].astype(float)
    betting_matches['result']=betting_matches['home_cover_result'] * betting_matches['bet_sign']
    

    # this is for graphing anlaysis on spreadsheet
    betting_matches['bet_sign_all'] = (np.where(betting_matches['total_factor']>0,1,np.where(betting_matches['total_factor']<-0,-1,0)))
    betting_matches['result_all']=betting_matches['home_cover_result'] * betting_matches['bet_sign_all']
    filt=betting_matches['bet_sign']!=0
    betting_matches['filtered_bets_made']=betting_matches['bet_sign'].where(filt)
    betting_matches['tease_spread']=betting_matches['filtered_bets_made']*6+betting_matches['Spread']
    betting_matches['tease_home_cover']=(np.where(((betting_matches['Home Points'] + betting_matches['tease_spread']) > betting_matches['Away Points']), 1,
    np.where(((betting_matches['Home Points']+ betting_matches['tease_spread']) < betting_matches['Away Points']),-1,0)))
    betting_matches['tease_result']=betting_matches['tease_home_cover'] * betting_matches['bet_sign']

    # st.write('testing sum of betting all result',betting_matches['result_all'].sum())
    cols_to_move=['Week','Date','Home Team','Away Team','total_factor','bet_on','result','Spread','Home Points','Away Points','home_power','away_power']
    cols = cols_to_move + [col for col in betting_matches if col not in cols_to_move]
    betting_matches=betting_matches[cols]
    betting_matches=betting_matches.sort_values('Date')
    presentation_betting_matches=betting_matches.copy()

    st.write(betting_matches[betting_matches['Week']==(finished_week)].set_index('Week').style.format({'home_power':"{:.1f}",'away_power':"{:.1f}",'result':"{:.0f}",
                                                                                       'Home Points':"{:.0f}",'Date':"{:%d.%m.%Y}",
                                           'Away Points':"{:.0f}",'Week':"{:.0f}",'home_cover':"{:.0f}",'away_cover':"{:.0f}",
                                           'calculated_spread':"{:.1f}",'Spread':"{:.1f}"}))
    
    st.write(betting_matches[betting_matches['Week']==(finished_week+1)].set_index('Week').style.format({'home_power':"{:.1f}",'away_power':"{:.1f}",'result':"{:.0f}",
                                                                                       'Home Points':"{:.0f}",'Date':"{:%d.%m.%Y}",
                                           'Away Points':"{:.0f}",'Week':"{:.0f}",'home_cover':"{:.0f}",'away_cover':"{:.0f}",
                                           'calculated_spread':"{:.1f}",'Spread':"{:.1f}"}))

    st.write(betting_matches[betting_matches['Week']==(15)].set_index('Week').style.format({'home_power':"{:.1f}",'away_power':"{:.1f}",'result':"{:.0f}",
                                                                                       'Home Points':"{:.0f}",'Date':"{:%d.%m.%Y}",
                                           'Away Points':"{:.0f}",'Week':"{:.0f}",'home_cover':"{:.0f}",'away_cover':"{:.0f}",
                                           'calculated_spread':"{:.1f}",'Spread':"{:.1f}"}))

        # https://stackoverflow.com/questions/68056855/pandas-style-conditional-formatting-highlight-on-text
    def color_recommend(value):
        if value == -1:
            color = 'red'
        elif value == 1:
            color = 'green'
        else:
            return
        return f'background-color: {color}'

    df_extract=betting_matches.loc[:,['Week','Date','Home Team','Away Team','total_factor','bet_on','Spread','power_pick','home_cover_sign','away_cover_sign',
                      'home_turnover_sign','away_turnover_sign']].set_index(['Date','Home Team','Away Team','total_factor','bet_on','Spread'])
    # df_extract.query('Week==')
    st.write(df_extract.loc[df_extract['Week']==(finished_week+1)].reset_index().set_index(['Week','Date','Home Team','Away Team','total_factor','bet_on','Spread'])\
              .style.format({'home_power':"{:.1f}",'away_power':"{:.1f}",'result':"{:.0f}",'Home Points':"{:.0f}",'Date':"{:%d.%m.%Y}",
                                           'Away Points':"{:.0f}",'Week':"{:.0f}",'home_cover_sign':"{:.0f}",'away_cover_sign':"{:.0f}",
                                           'calculated_spread':"{:.1f}",'Spread':"{:.1f}"}).applymap(color_recommend))
    

    #        
    #   .set_index(['Week']))
    # st.write(df_extract.loc[df_extract['Week']==(finished_week+1)]\
    #          .set_index(['Week']).style.format({'home_power':"{:.1f}",'away_power':"{:.1f}",'result':"{:.0f}",'Home Points':"{:.0f}",'Date':"{:%d.%m.%Y}",
    #                                        'Away Points':"{:.0f}",'Week':"{:.0f}",'home_cover_sign':"{:.0f}",'away_cover_sign':"{:.0f}",
    #                                        'calculated_spread':"{:.1f}",'Spread':"{:.1f}"}).applymap(color_recommend) )

    # https://towardsdatascience.com/7-reasons-why-you-should-use-the-streamlit-aggrid-component-2d9a2b6e32f0
    grid_height = st.number_input("Grid height", min_value=400, value=450, step=100)
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
    # st.write(pivot_df_test)
    pivot_df_test=pivot_df_test[pivot_df_test['week']<finished_week+1]
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
    # st.write('test',reset_data)
    reset_data=reset_data.pivot(index='result_all',columns='total_factor',values='winning').fillna(0)
    # st.write('look',reset_data)
    dfBool=pd.Series(reset_data.columns.isin([3,4,5]) )
    reset_data['betting_factor_total']=reset_data[reset_data.columns[dfBool]].sum(axis=1)
    # reset_data['betting_factor_total']=reset_data[3]+reset_data[4]+reset_data[5]
    # reset_data['betting_factor_total']=reset_data.iloc[:,:-3].sum()
    reset_data=reset_data.sort_values(by='betting_factor_total',ascending=False)

    reset_data=reset_data.reset_index()
    # st.write('reset data', reset_data)
    reset_data['result_all']=reset_data['result_all'].astype(str)
    reset_data=reset_data.set_index('result_all')

    reset_data.loc['Total']=reset_data.sum()
    # reset_data.loc['No. of Bets Made'] = reset_data.loc[['1','-1']].sum()
    reset_data.loc['Winning_Bets']=(reset_data.loc[reset_data.index.isin({'1'})].sum(axis=0))
    reset_data.loc['Losing_Bets']=(reset_data.loc[reset_data.index.isin({'-1'})].sum(axis=0))
    reset_data.loc['No. of Bets Made']=(reset_data.loc[reset_data.index.isin({'1'})].sum(axis=0))+(reset_data.loc[reset_data.index.isin({'-1'})].sum(axis=0)) 
    reset_data.loc['PL_Bets']=reset_data.loc['Winning_Bets'] - reset_data.loc['Losing_Bets']
    reset_data=reset_data.apply(pd.to_numeric, downcast='integer')
    reset_data.loc['% Winning'] = ((reset_data.loc[reset_data.index.isin({'1'})].sum(axis=0) / reset_data.loc['No. of Bets Made'])).replace({'<NA>':np.NaN})
    st.write('This shows the betting result')
    # st.write(reset_data)

    # https://stackoverflow.com/questions/64428836/use-pandas-style-to-format-index-rows-of-dataframe
    reset_data = reset_data.style.format("{:.0f}", na_rep='-')
    reset_data = reset_data.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])

    st.write(reset_data)
    st.write('Broken down by the number of factors indicating the strength of the signal')

with placeholder_1.expander('Weekly Results'):
    # weekly_results=analysis.groupby(['Week','result']).agg(winning=('result','sum'),count=('result','count'))
    # weekly_test=analysis[analysis['total_factor'].abs()>2].loc[:,['Week','result']].copy()

    # df9 = weekly_test.groupby(['result','Week']).size().unstack(fill_value=0)
    # df9=df9.reset_index()
    # df9['result']=df9['result'].astype(int).astype(str)
    # df9=df9.set_index('result').sort_index(ascending=False)
    # # df9.columns = df9.columns.astype(str)
    # # df_slice=df9.loc[:,13:]
    # # df9['subtotal_week_13_on']=df_slice.sum(axis=1)
    # # df_all=df9.iloc[:,:-1]
    # # st.write('this is df all',df_all)
    # # df9['grand_total']=df_all.sum(axis=1)
    # df9['grand_total']=df9.iloc[:,:-1].sum(axis=1)
    # df9.loc['Total']=df9.sum()
    # df9.loc['No. of Bets Made'] = df9.loc[['1','-1']].sum() 
    # df9=df9.apply(pd.to_numeric, downcast='integer')
    # df9.loc['% Winning'] = ((df9.loc['1'] / df9.loc['No. of Bets Made'])).replace({'<NA>':np.NaN})

    # # https://stackoverflow.com/questions/64428836/use-pandas-style-to-format-index-rows-of-dataframe
    # df9 = df9.style.format("{:.0f}", na_rep='-')
    # df9 = df9.format(formatter="{:.0%}", subset=pd.IndexSlice[['% Winning'], :])
    # st.write(df9)

    # st.write('Total betting result',betting_matches['result'].sum())
    # pivot_weekly.columns = pivot_weekly.columns.droplevel(0)
    # weekly_test=analysis.groupby([(analysis['total_factor'].abs()>2),'result_all']).agg(winning=('result_all','count'))
    # st.write(pivot_weekly)
    # st.write(weekly_test)
    # --------------------------------------------------------------------------------------------------------------------------

    weekly_results=analysis.groupby(['Week','result']).agg(winning=('result','sum'),count=('result','count'))
    weekly_test=analysis[analysis['total_factor'].abs()>2].loc[:,['Week','result']].copy()
    df9 = weekly_test.groupby(['result','Week']).size().unstack(fill_value=0)
    df9=df9.reset_index()
    df9['result']=df9['result'].round(1).astype(str)
    df9=df9.set_index('result').sort_index(ascending=False)
    df9['grand_total']=df9.sum(axis=1)
    df9.loc['Winning_Bets']=(df9.loc[df9.index.isin({'1.0','1'})].sum(axis=0))
    df9.loc['Losing_Bets']=(df9.loc[df9.index.isin({'-1','-1.0'})].sum(axis=0))
    df9.loc['No. of Bets Made'] = df9.loc['Winning_Bets']+ df9.loc['Losing_Bets']
    df9.loc['PL_Bets']=df9.loc['Winning_Bets'] - df9.loc['Losing_Bets']
    df9=df9.apply(pd.to_numeric, downcast='float')
    graph_pl_data=df9.loc[['PL_Bets'],:].drop('grand_total',axis=1)
    graph_pl_data=graph_pl_data.stack().reset_index().drop('result',axis=1).rename(columns={0:'week_result'})
    graph_pl_data['Week']=graph_pl_data['Week'].astype(int)
    graph_pl_data['total_result']=graph_pl_data['week_result'].cumsum()
    graph_pl_data=graph_pl_data.melt(id_vars='Week',var_name='category',value_name='result')
    df9.loc['% Winning'] = (df9.loc['Winning_Bets'] / (df9.loc['Winning_Bets']+df9.loc['Losing_Bets'])  ).replace({'<NA>':np.NaN})
    table_test=df9.copy()
    # https://stackoverflow.com/questions/64428836/use-pandas-style-to-format-index-rows-of-dataframe
    df9 = df9.style.format("{:.1f}", na_rep='-')
    # df9 = df9.format(formatter="{:.0%}", subset=pd.IndexSlice[['% Winning'], :]).format(formatter="{:.0f}", subset=pd.IndexSlice[['1.0'], :]) \
    #     .format(formatter="{:.0f}", subset=pd.IndexSlice[['-1.0'], :])
        # .format(formatter="{:.0f}", subset=pd.IndexSlice[['-0.0'], :]) \

    def graph_pl(decile_df_abs_home_1,column):
        line_cover= alt.Chart(decile_df_abs_home_1).mark_line().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
        alt.Y(column),color=alt.Color('category'))
        text_cover=line_cover.mark_text(baseline='middle',dx=0,dy=-15).encode(text=alt.Text(column),color=alt.value('black'))
        overlay = pd.DataFrame({column: [0]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=1).encode(y=column)
        return st.altair_chart(line_cover + text_cover + vline,use_container_width=True)

    graph_pl(graph_pl_data,column='result')

    st.write('Total betting result per Betting Table',betting_matches['result'].sum())
    st.write('Total betting result per Above Table',table_test.loc['PL_Bets','grand_total'])
    st.write(df9)



with st.expander('Analysis of Factors'):
    analysis_factors = betting_matches.copy()
    analysis_factors=analysis_factors[analysis_factors['Week']<finished_week+1]
    st.write('where we have a zero for the power ranking success is where the spread was a push cos it exactly covered')
    # test_factor=analysis_factors.copy()
    # test_factor['power_ranking_success?'] = test_factor['power_pick'] * test_factor['home_cover_result']
    # st.write('analysis factors', test_factor[test_factor['power_ranking_success?']==0])

        # sourcery skip: remove-unnecessary-else, swap-if-else-branchesors['Week']<finished_week+1]
    def analysis_factor_function(analysis_factors):
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches
        analysis_factors['home_turnover_success?'] = analysis_factors['home_turnover_sign'] * analysis_factors['home_cover_result']
        analysis_factors['away_turnover_success?'] = analysis_factors['away_turnover_sign'] * analysis_factors['home_cover_result']
        analysis_factors['home_cover_season_success?'] = analysis_factors['home_cover_sign'] * analysis_factors['home_cover_result']  
        analysis_factors['away_cover_season_success?'] = analysis_factors['away_cover_sign'] * analysis_factors['home_cover_result']
        analysis_factors['power_ranking_success?'] = analysis_factors['power_pick'] * analysis_factors['home_cover_result']
        # st.write(analysis_factors)
        df_table = analysis_factors['home_turnover_success?'].value_counts().reset_index().rename(columns={'home_turnover_success?':'factor','count':'home_turnover_success?'}).set_index('factor')
        away_turnover=analysis_factors['away_turnover_success?'].value_counts().reset_index().rename(columns={'away_turnover_success?':'factor','count':'away_turnover_success?'}).set_index('factor')
        home_cover=analysis_factors['home_cover_season_success?'].value_counts().reset_index().rename(columns={'home_cover_season_success?':'factor','count':'home_cover_season_success?'}).set_index('factor')
        away_cover=analysis_factors['away_cover_season_success?'].value_counts().reset_index().rename(columns={'away_cover_season_success?':'factor','count':'away_cover_season_success?'}).set_index('factor')
        power=analysis_factors['power_ranking_success?'].value_counts().reset_index().rename(columns={'power_ranking_success?':'factor','count':'power_ranking_success?'}).set_index('factor')
        df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1)
        # df_table_1=pd.concat([df_table,away_turnover,home_cover,away_cover,power],axis=1).reset_index().drop('index',axis=1)
        # st.write('df',df_table_1)
        # st.write(away_turnover, home_cover,away_cover)
        # st.write('df table', df_table_1)
        # test=df_table_1.reset_index()
        # st.write(test)
        df_table_1.loc[:,'total_turnover'] = df_table_1['home_turnover_success?'] + (df_table_1['away_turnover_success?'])
        # st.write(test)
        df_table_1['total_season_cover'] = df_table_1['home_cover_season_success?'] + df_table_1['away_cover_season_success?']
        # st.write('df table 2', df_table_1)

        df_table_1=df_table_1.reset_index()
        # st.write('reset data', df_table_1)
        df_table_1['index']=df_table_1['factor'].astype(str)
        df_table_1=df_table_1.drop('factor',axis=1)
        # st.write('df table 2', df_table_1)
        df_table_1=df_table_1.set_index('index')


        df_table_1.loc['Total']=df_table_1.sum()
        # st.write('latest', df_table_1)
        # st.write('latest', df_table_1.shape)
        if df_table_1.shape > (2,7):
            # st.write('Returning df with analysis')
            df_table_1.loc['No. of Bets Made'] = df_table_1.loc[['1','-1']].sum() # No losing bets so far!!!
            df_table_1.loc['% Winning'] = ((df_table_1.loc['1'] / df_table_1.loc['No. of Bets Made']))
        else:
            # st.write('Returning df with no analysis')
            return df_table_1
        return df_table_1
    total_factor_table = analysis_factor_function(analysis_factors)   
    st.write('This is the total number of matches broken down by Factor result')
    cols_to_move=['total_turnover','total_season_cover','power_ranking_success?']
    total_factor_table = total_factor_table[ cols_to_move + [ col for col in total_factor_table if col not in cols_to_move ] ]
    total_factor_table=total_factor_table.loc[:,['total_turnover','total_season_cover','power_ranking_success?']]
    
    total_factor_table_presentation = total_factor_table.style.format("{:.2f}", na_rep='-')
    # total_factor_table_presentation = total_factor_table_presentation.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])
    
    st.write(total_factor_table_presentation)
    factor_bets = (analysis_factors[analysis_factors['bet_sign']!=0]).copy()
    bets_made_factor_table = analysis_factor_function(factor_bets)
    # cols_to_move=['total_turnover','total_season_cover','power_ranking_success?']
    bets_made_factor_table = bets_made_factor_table[ cols_to_move + [ col for col in bets_made_factor_table if col not in cols_to_move ] ]
    bets_made_factor_table=bets_made_factor_table.loc[:,['total_turnover','total_season_cover','power_ranking_success?']]
    st.write('This is the matches BET ON broken down by Factor result')
    bets_made_factor_table_presentation = bets_made_factor_table.style.format("{:.2f}", na_rep='-')
    # bets_made_factor_table_presentation = bets_made_factor_table_presentation.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])
    st.write(bets_made_factor_table_presentation)

    # st.write('graph work below')
    transposed_df=total_factor_table.transpose().reset_index().set_index('index')
    dfBool=pd.Series(transposed_df.columns.isin(['0','0.0','1','-1','1.0','-1.0']) )
    test_bool=transposed_df[transposed_df.columns[dfBool]].fillna(0).transpose()
    graph_factor_table = test_bool.reset_index().rename(columns={'index':'result_all'})
    # graph_factor_table = total_factor_table.copy().loc[['-1','0','1'],:].reset_index().rename(columns={'index':'result_all'})
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
    # st.write('copy the work done above in the other section where it was fixed')
    # st.write('df factor', df_factor)
    # st.write('df factor', df_factor)
    df_factor_table = df_factor['home_turnover_diagnostic'].value_counts().reset_index().rename(columns={'home_turnover_diagnostic':'factor','count':'home_turnover_diagnostic'}).set_index('factor')
    away_turnover=df_factor['away_turnover_diagnostic'].value_counts().reset_index().rename(columns={'away_turnover_diagnostic':'factor','count':'away_turnover_diagnostic'}).set_index('factor')
    home_cover=df_factor['home_cover_diagnostic'].value_counts().reset_index().rename(columns={'home_cover_diagnostic':'factor','count':'home_cover_diagnostic'}).set_index('factor')
    away_cover=df_factor['away_cover_diagnostic'].value_counts().reset_index().rename(columns={'away_cover_diagnostic':'factor','count':'away_cover_diagnostic'}).set_index('factor')
    power=df_factor['power_diagnostic'].value_counts().reset_index().rename(columns={'power_diagnostic':'factor','count':'power_diagnostic'}).set_index('factor')

    # df_table = analysis_factors['home_turnover_success?'].value_counts().reset_index().rename(columns={'home_turnover_success?':'factor','count':'home_turnover_success?'}).set_index('factor')
    # away_turnover=analysis_factors['away_turnover_success?'].value_counts().reset_index().rename(columns={'away_turnover_success?':'factor','count':'away_turnover_success?'}).set_index('factor')
    # home_cover=analysis_factors['home_cover_season_success?'].value_counts().reset_index().rename(columns={'home_cover_season_success?':'factor','count':'home_cover_season_success?'}).set_index('factor')
    # away_cover=analysis_factors['away_cover_season_success?'].value_counts().reset_index().rename(columns={'away_cover_season_success?':'factor','count':'away_cover_season_success?'}).set_index('factor')
    # power=analysis_factors['power_ranking_success?'].value_counts().reset_index().rename(columns={'power_ranking_success?':'factor','count':'power_ranking_success?'}).set_index('factor')

        # df_table_1.loc[:,'total_turnover'] = df_table_1['home_turnover_success?'] + (df_table_1['away_turnover_success?'])
        # # st.write(test)
        # df_table_1['total_season_cover'] = df_table_1['home_cover_season_success?'] + df_table_1['away_cover_season_success?']
        # # st.write('df table 2', df_table_1)

        # df_table_1=df_table_1.reset_index()
        # # st.write('reset data', df_table_1)
        # df_table_1['index']=df_table_1['factor'].astype(str)
        # df_table_1=df_table_1.drop('factor',axis=1)
        # # st.write('df table 2', df_table_1)
        # df_table_1=df_table_1.set_index('index')


        # df_table_1.loc['Total']=df_table_1.sum()
        # # st.write('latest', df_table_1)
        # # st.write('latest', df_table_1.shape)
        # if df_table_1.shape > (2,7):
        #     # st.write('Returning df with analysis')
        #     df_table_1.loc['No. of Bets Made'] = df_table_1.loc[['1','-1']].sum() # No losing bets so far!!!
        #     df_table_1.loc['% Winning'] = ((df_table_1.loc['1'] / df_table_1.loc['No. of Bets Made']))

    df_factor_table_1=pd.concat([df_factor_table,away_turnover,home_cover,away_cover,power],axis=1)
    df_factor_table_1['total_turnover'] = df_factor_table_1['home_turnover_diagnostic'] + (df_factor_table_1['away_turnover_diagnostic'])
    # st.write(test)
    df_factor_table_1['total_season_cover'] = df_factor_table_1['home_cover_diagnostic'] + df_factor_table_1['away_cover_diagnostic']
    # st.write('df table 2', df_factor_table_1)

    df_factor_table_1=df_factor_table_1.reset_index()
    # st.write(df_factor_table_1['factor'].round(1))
    # st.write('reset data', df_table_1)
    df_factor_table_1['index']=df_factor_table_1['factor'].astype(int).astype(str)
    # st.write(df_factor_table_1['index'].round(1))
    df_factor_table_1=df_factor_table_1.drop('factor',axis=1)
    # st.write('df table 2', df_table_1)
    df_factor_table_1=df_factor_table_1.set_index('index')


    df_factor_table_1.loc['Total']=df_factor_table_1.sum()
    # st.write('latest', df_factor_table_1)
    # st.write('latest', df_factor_table_1.shape)

    if df_factor_table_1.shape > (2,7):
        df_factor_table_1.loc['No. of Bets Made'] = df_factor_table_1.loc[['1','-1']].sum() 
        df_factor_table_1.loc['% Winning'] = ((df_factor_table_1.loc['1'] / df_factor_table_1.loc['No. of Bets Made']))
    # else:
    #     # st.write('Returning df with no anal')
    #     return df_factor_table_1


    cols_to_move=['total_turnover','total_season_cover','power_diagnostic']
    df_factor_table_1 = df_factor_table_1[ cols_to_move + [ col for col in df_factor_table_1 if col not in cols_to_move ] ]
    df_factor_table_1=df_factor_table_1.loc[:,['total_turnover','total_season_cover','power_diagnostic']]

    df_factor_table_1_presentation = df_factor_table_1.style.format("{:.2f}", na_rep='-')
    # df_factor_table_1_presentation = df_factor_table_1_presentation.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])

    st.write(df_factor_table_1_presentation)




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
    # st.write('uunder df', underdog_df)
    underdog_table = underdog_df['home_underdog_bet_result'].value_counts().reset_index().rename(columns={'home_underdog_bet_result':'factor','count':'home_underdog_bet_result'}).set_index('factor')
    away_underdog_bet = underdog_df['away_underdog_bet_result'].value_counts().reset_index().rename(columns={'away_underdog_bet_result':'factor','count':'away_underdog_bet_result'}).set_index('factor')
    home_fav_bet = underdog_df['home_favourite_bet_result'].value_counts().reset_index().rename(columns={'home_favourite_bet_result':'factor','count':'home_favourite_bet_result'}).set_index('factor')
    away_fav_bet = underdog_df['away_favourite_bet_result'].value_counts().reset_index().rename(columns={'away_favourite_bet_result':'factor','count':'away_favourite_bet_result'}).set_index('factor')
    # st.write(underdog_table)
    underdog_results = pd.concat([underdog_table,away_underdog_bet,home_fav_bet,away_fav_bet],axis=1)
    underdog_results=underdog_results.sort_index(ascending=False)
    # st.write('problem?', underdog_results)
    underdog_results['underdog']=underdog_results['home_underdog_bet_result']+underdog_results['away_underdog_bet_result']
    underdog_results['favourite']=underdog_results['home_favourite_bet_result']+underdog_results['away_favourite_bet_result']

    underdog_results=underdog_results.reset_index()
    # st.write('reset data', underdog_results)
    underdog_results['index']=underdog_results['factor'].astype(int)
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

    underdog_results_presentation = underdog_results.style.format("{:.2f}", na_rep='-')
    # underdog_results_presentation = underdog_results_presentation.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])

    st.write(underdog_results_presentation)

    home_underdog_all = underdog_df['home_underdog_all_result'].value_counts().reset_index().rename(columns={'home_underdog_all_result':'factor','count':'home_underdog_all_result'}).set_index('factor')
    away_underdog_all = underdog_df['away_underdog_all_result'].value_counts().reset_index().rename(columns={'away_underdog_all_result':'factor','count':'away_underdog_all_result'}).set_index('factor')
    home_fav_all = underdog_df['home_favourite_all_result'].value_counts().reset_index().rename(columns={'home_favourite_all_result':'factor','count':'home_favourite_all_result'}).set_index('factor')
    away_fav_all = underdog_df['away_favourite_all_result'].value_counts().reset_index().rename(columns={'away_favourite_all_result':'factor','count':'away_favourite_all_result'}).set_index('factor')
    all_results = pd.concat([home_underdog_all,away_underdog_all,home_fav_all,away_fav_all],axis=1)
    all_results=all_results.sort_index(ascending=False)
    all_results['underdog']=all_results['home_underdog_all_result']+all_results['away_underdog_all_result']
    all_results['favourite']=all_results['home_favourite_all_result']+all_results['away_favourite_all_result']

    all_results=all_results.reset_index()
    # st.write('reset data', all_results)
    all_results['index']=all_results['factor'].astype(int)
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

    all_results_presentation = all_results.style.format("{:.2f}", na_rep='-')
    # all_results_presentation = all_results_presentation.format(formatter="{:.1%}", subset=pd.IndexSlice[['% Winning'], :])

    st.write(all_results_presentation,width=1000)

# test
    
    



# with st.beta_expander('Pro Football Ref Scraper'):
    # pass
    # def fbref_scraper():
    #     test = pd.read_html('https://www.pro-football-reference.com/years/2021/games.htm')[0]
    #     test.to_pickle('C:/Users/Darragh/Documents/Python/NFL/pro_football_ref/nfl_2021.pkl')
    #     return test  
        
    # test=fbref_scraper()

with st.expander('Deep Dive on Power Factor'):
    power_factor_analysis = analysis_factors.copy()
    power_factor_analysis['power_ranking_success?'] = power_factor_analysis['power_pick'] * power_factor_analysis['home_cover_result']
    power_factor_analysis['home_power_less_away'] = power_factor_analysis['away_power']-power_factor_analysis['home_power']
    power_factor_analysis['power_margin'] = power_factor_analysis['home_power_less_away'] - power_factor_analysis['Spread']
    cols_to_move=['Week','Date','Home Team','Away Team','Spread','home_power_less_away','power_margin','power_ranking_success?','home_power','away_power']
    power_factor_analysis = power_factor_analysis[ cols_to_move + [ col for col in power_factor_analysis if col not in cols_to_move ] ]
    week_power_analysis=power_factor_analysis.groupby(['Week'])['power_ranking_success?'].sum().reset_index()
    week_power_analysis['cum_sum']=week_power_analysis['power_ranking_success?'].cumsum()
    
    # st.write(week_power_analysis)


    decile_df_abs_home=power_factor_analysis.groupby(['power_pick'])['power_ranking_success?'].sum().reset_index()
    st.write('breaks out Home Away')
    # st.write(decile_df_abs_home)
    st.altair_chart(alt.Chart(decile_df_abs_home).mark_bar().encode(x='power_pick:N',y='power_ranking_success?'),use_container_width=True)

    decile_df_abs_home_1=power_factor_analysis.groupby(['Week','power_pick'])['power_ranking_success?'].sum().reset_index()
    decile_df_abs_home_1=power_factor_analysis.groupby(['Week','power_pick']).agg(
        power_ranking_success=('power_ranking_success?','sum'),count=('power_pick','count')).reset_index()
    # st.write('testing', test_replicate)
    decile_df_abs_home_1['test_sum']=decile_df_abs_home_1.groupby(['Week'])['power_ranking_success'].transform('sum')
    decile_df_abs_home_1['cum_sum_home_away']=decile_df_abs_home_1.groupby(['power_pick'])['power_ranking_success'].cumsum()
    # st.write('breaks out Home Away by week')
    # st.write(decile_df_abs_home_1)
    # st.write( power_factor_analysis[(power_factor_analysis['Home Team'].str.contains('Hoffen') | power_factor_analysis['Away Team'].str.contains('Hoffen'))] )
    # st.write(power_factor_analysis)


    scale_3=alt.Scale(domain=['1','-1'],range=['blue','red'])
    def graph(decile_df_abs_home_1,column):
        line_cover= alt.Chart(decile_df_abs_home_1).mark_line().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
        alt.Y(column),color=alt.Color('power_pick:Q',scale=scale_3))
        text_cover=line_cover.mark_text(baseline='middle',dx=20,dy=-5).encode(text=alt.Text(column),color=alt.value('black'))
        overlay = pd.DataFrame({column: [0]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=1).encode(y=column)
        return st.altair_chart(line_cover + text_cover + vline,use_container_width=True)


    st.write('Below shows the weekly net result for Home and Away games backed by the Power Factor')
    graph(decile_df_abs_home_1,column='power_ranking_success')
    st.write('Blue = Home and Red = Away')
    st.write('Below shows the cumulative win/loss by home away games')
    graph(decile_df_abs_home_1,column='cum_sum_home_away')
    st.write('What is the breakdown of power pick by Home / Away')
    
    line_cover= alt.Chart(decile_df_abs_home_1).mark_bar().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('count'),color=alt.Color('power_pick:N'))
    text_cover=line_cover.mark_text(baseline='middle').encode(text=alt.Text('count:N'),color=alt.value('black'))
    overlay = pd.DataFrame({'count': [8]})
    vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=1).encode(y='count:Q')
    st.altair_chart(line_cover + vline,use_container_width=True)



    decile_df_abs_spread=power_factor_analysis.groupby(pd.qcut(power_factor_analysis['Spread'].abs(), q=10,duplicates='drop'))['power_ranking_success?'].sum().reset_index()
    st.write(decile_df_abs_spread)
    decile_df_abs_spread=decile_df_abs_spread.drop('Spread',axis=1).reset_index().rename(columns={'index':'Spread'})
    st.write(decile_df_abs_spread)
    line_cover= alt.Chart(decile_df_abs_spread).mark_bar().encode(alt.X('Spread',axis=alt.Axis(title='Spread',labelAngle=0)),
    alt.Y('power_ranking_success?:Q'))
    text_cover=line_cover.mark_text(baseline='middle').encode(text=alt.Text('power_ranking_success?'),color=alt.value('black'))
    overlay = pd.DataFrame({'power_ranking_success?': [0]})
    vline = alt.Chart(overlay).mark_rule(color='red', strokeWidth=1).encode(y='power_ranking_success?:Q')
    st.write('The Below Chart splits the Spread into even buckets and looks at the win/loss record for each bucket')    
    st.altair_chart(line_cover + vline,use_container_width=True)

    st.write('Below is breaking into even buckets the difference between the Model and the Spread to see if there is any insight')
    decile_df=power_factor_analysis.groupby(pd.qcut(power_factor_analysis['power_margin'], 8))['power_ranking_success?'].sum()
    decile_df_abs=power_factor_analysis.groupby(pd.qcut(power_factor_analysis['power_margin'].abs(), 8))['power_ranking_success?'].sum().reset_index()
    st.write(decile_df_abs)
    decile_df_abs=decile_df_abs.drop('power_margin',axis=1).reset_index().rename(columns={'index':'power_margin'})
    # st.write(decile_df)
    # st.write(decile_df_abs)
    line_cover= alt.Chart(decile_df_abs).mark_bar().encode(alt.X('power_margin',axis=alt.Axis(title='power_margin',labelAngle=0)),
    alt.Y('power_ranking_success?:Q'))
    text_cover=line_cover.mark_text(baseline='middle').encode(text=alt.Text('power_ranking_success?'),color=alt.value('black'))
    overlay = pd.DataFrame({'power_ranking_success?': [0]})
    vline = alt.Chart(overlay).mark_rule(color='red', strokeWidth=1).encode(y='power_ranking_success?:Q')
    st.altair_chart(line_cover + vline,use_container_width=True)

with st.expander('Teaser Analyis'):
    teaser_df = betting_matches.copy()
    # st.write('teaser df', teaser_df)
    cols_to_move=['Week','Home Team','Away Team','total_factor','bet_on','result','Spread','Home Points','Away Points','tease_spread','tease_result','filtered_bets_made','tease_home_cover']
    cols = cols_to_move + [col for col in teaser_df if col not in cols_to_move]
    teaser_df=teaser_df[cols]
    # st.write('teaser df', teaser_df)
    # AgGrid(teaser_df)

    group_teaser=teaser_df.groupby(['Week','tease_result']).agg(count=('tease_spread','count')).reset_index()
    # st.write(group_teaser)
    # group_teaser_1=teaser_df.groupby(['Week']).agg(count=('tease_spread','count'),sum=('tease_result','sum'))
    group_teaser=(pd.pivot_table(group_teaser,index='Week', columns='tease_result').fillna(0))
    group_teaser.columns=group_teaser.columns.get_level_values(1)
    group_teaser.columns=['lose','no_result','win']
    # group_teaser=group_teaser.reset_index()
    # st.write(group_teaser.columns)
    # st.write(group_teaser)
    filt=(group_teaser['lose']==0 ) & (group_teaser['win']>2)
    group_teaser['success_week']=group_teaser['win'].where(filt)
    # st.write(group_teaser)
    odds_df=pd.DataFrame.from_dict({1:2,2:2,3:2.5,4:3.25,5:4.5,6:6,7:8,8:11,9:15},orient='index').reset_index().rename(columns={'index':'success_week',0:'odds'})
    market_odds=pd.DataFrame.from_dict({1:-110,2:-110,3:150.5,4:250,5:400,6:600,7:700,8:800,9:900},orient='index').reset_index().rename(columns={'index':'success_week',0:'market_odds'})
    # https://www.sportsbettingstats.com/sportsbooks/best-teaser-odds
    # st.write('odds', odds_df)
    odds_df['american_odds']=np.where(odds_df['odds']>1.999,(odds_df['odds']-1)*100,-100/(odds_df['odds']-1))
    # st.write('market odds', market_odds)
    st.write('there is no 2 point teaser available on Bet365, also the market odds are the worst that i could find, so it doesnt look great for bet365')
    odds_df=pd.merge(odds_df,market_odds,how='outer')
    st.write('6 Point Teaser odds', odds_df)
    test_merge_odds=pd.merge(group_teaser,odds_df,how='left')
    # test_merge_odds['stake']=-10
    test_merge_odds['stake']=np.where((test_merge_odds['win']+test_merge_odds['lose']+test_merge_odds['no_result'])<3,np.NaN,-10)
    test_merge_odds['payout']=test_merge_odds['odds'] * 10
    test_merge_odds['net_winnings']=test_merge_odds['stake']+test_merge_odds['payout'].fillna(0)
    st.write('6 Point Teaser Return', test_merge_odds)
    st.write('total staked',test_merge_odds['stake'].sum())
    st.write('total returned',test_merge_odds['payout'].sum())


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


    
    st.write('Check sum if True all good', full_stack.sum().sum()==0)