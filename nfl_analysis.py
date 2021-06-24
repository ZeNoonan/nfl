import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import os
import base64 
import altair as alt  

st.set_page_config(layout="wide")
# st.header('Need to bring in previous 4 weeks in prior season')

@st.cache
def read_data(file):
    return pd.read_excel(file) 
data_2019 = read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2019_Data.xlsx').copy()
# data_2020=read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2020_Data_Adj_week_zero.xlsx').copy()
data_2020=read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2020_Data.xlsx').copy()
test_data_2020=read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2020_Data_Test.xlsx').copy()
odds_data = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_betting_odds.xlsx').copy()
team_names_id = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_teams.xlsx').copy()

# url='https://www.pro-football-reference.com/years/2015/games.htm'
# def fbref_scraper(url):
#         test = pd.read_html(url)[0]
#         test.to_pickle('C:/Users/Darragh/Documents/Python/NFL/pro_football_ref/nfl_2015.pkl')
#         return test 

# fbref_scraper(url)
with st.echo():
    nfl_data=pd.read_pickle('C:/Users/Darragh/Documents/Python/NFL/pro_football_ref/nfl_2020.pkl')
    prior_nfl_data = pd.read_pickle('C:/Users/Darragh/Documents/Python/NFL/pro_football_ref/nfl_2019.pkl')
# st.write('this is prior year data',prior_nfl_data)

with st.beta_expander('Historical odds function'):
    # st.write(odds_data)
    odds_data=odds_data.loc[:,['Date','Home Team','Away Team','Home Score','Away Score','Home Line Close']].copy()
    team_names_id=team_names_id.rename(columns={'Team':'Home Team'})
    odds_data=pd.merge(odds_data,team_names_id,on='Home Team').rename(columns={'ID':'Home ID'}).sort_values(by='Date',ascending=False)
    team_names_id=team_names_id.rename(columns={'Home Team':'Away Team'})
    odds_data=pd.merge(odds_data,team_names_id,on='Away Team').rename(columns={'ID':'Away ID','Home Score':'Home Points',
    'Away Score':'Away Points','Home Line Close':'Spread'}).sort_values(by='Date',ascending=False)
    # st.write(odds_data.dtypes)

    st.write(odds_data)
    st.write('To Check that all ok')
    st.write(odds_data[odds_data['Away ID'].isna()])


with st.beta_expander('Pro Football Function'):
    
    # st.write('This is before cleaning',nfl_data)
    def clean_pro_football_pickle(nfl_data):
        nfl_data=nfl_data.rename(columns={'Unnamed: 5':'at_venue'})
        nfl_data['Home Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Loser/tie'],nfl_data['Winner/tie'])
        nfl_data['at_venue']=nfl_data['at_venue'].replace({np.nan:'stay'})
        nfl_data['Away Team']=np.where(nfl_data['at_venue']=='@',nfl_data['Winner/tie'],nfl_data['Loser/tie'])
        nfl_data['Home Points']=np.where(nfl_data['at_venue']=='@',nfl_data['Pts.1'],nfl_data['Pts'])
        nfl_data['Away Points']=np.where(nfl_data['at_venue']=='@',nfl_data['Pts'],nfl_data['Pts.1'])
        nfl_data['home_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOL'],nfl_data['TOW']))
        nfl_data['away_turnover']=(np.where(nfl_data['at_venue']=='@',nfl_data['TOW'],nfl_data['TOL']))
        nfl_data=nfl_data[nfl_data['Week'].str.contains('Week')==False].copy()
        nfl_data['home_turnover']=pd.to_numeric(nfl_data['home_turnover'])
        nfl_data['away_turnover']=pd.to_numeric(nfl_data['away_turnover'])
        nfl_data['Home Points']=pd.to_numeric(nfl_data['Home Points'])
        nfl_data['Away Points']=pd.to_numeric(nfl_data['Away Points'])
        nfl_data['Date']=pd.to_datetime(nfl_data['Date'])
        nfl_data['Week'] = nfl_data['Week'].replace({'WildCard':18,'Division':19,'ConfChamp':20,'SuperBowl':21})
        nfl_data['Week']=pd.to_numeric(nfl_data['Week'])
        fb_ref_2020=nfl_data.loc[:,['Week','Day','Date','Time','Home Team', 'Away Team', 'Home Points','Away Points','home_turnover','away_turnover']]
        fb_ref_2020['Turnover'] = fb_ref_2020['home_turnover'] - fb_ref_2020['away_turnover']
        # st.write(fb_ref_2020.dtypes)
        # st.write('before the merge Pro-Football Ref',fb_ref_2020)
        # st.write('before the merge Odds Data',odds_data)
        # st.write('Check and see if this is working right')
        # season_pro = pd.merge(fb_ref_2020,odds_data,on=['Date','Home Points','Away Points'], how='left')
        # IT'S IMPORTANT THAT THE ODDS MERGES CORRECTLY WITH FBREF Data for the games with neutral venues as need to get spread right!
        season_pro = pd.merge(fb_ref_2020,odds_data,on=['Date','Home Team','Away Team', 'Home Points','Away Points'], how='left')
        # st.write('season pro #1 after merge with updated merged on',season_pro)
        return season_pro
    # st.write(season_pro.head(3))

def clean_prior_year(x):
    x['Week']=x['Week'].replace({18:0,19:0,20:0,21:0,17:0,16:-1,15:-2,14:-3})
    x=x[x['Week'].between(-3,0)].copy()
    x=x.reset_index().drop('index',axis=1)
    st.write('Check for errors',x[x['Away ID'].isna()])
    return x

def concat_current_prior(x,y):
    current_plus_prior = pd.concat([x,y],axis=0,ignore_index=True)
    return current_plus_prior
    

with st.echo():    
    current=clean_pro_football_pickle(nfl_data)
    prior_data = clean_prior_year(clean_pro_football_pickle(prior_nfl_data))

data = concat_current_prior(current,prior_data)
# st.write('Just check the Data', data.sort_values(by=['Week','Date','Time']))
# st.write( data[(data['Home Team']=='Arizona Cardinals') | (data['Away Team']=='Arizona Cardinals')].sort_values(by=['Week','Date','Time']) )
# st.write( data[(data['Home Team']=='Atlanta Falcons') | (data['Away Team']=='Atlanta Falcons')].sort_values(by=['Week','Date','Time']) )
# st.write( data[(data['Home ID']==21) | (data['Away ID']==21)].sort_values(by=['Week','Date','Time']) )

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
# st.write('spread',spread)

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
    # return season_cover_df

def turnover_cover_3(data,column_sign,name):
    data[column_sign] = np.where((data[name] > 0), 1, np.where((data[name] < 0),-1,0))
    return data


def turnover_workings(data,week_start):
    turnover_df=data[data['Week']>week_start].copy()
    turnover_df['home_turned_over_sign'] = np.where((turnover_df['Turnover'] > 0), 1, np.where((turnover_df['Turnover'] < 0),-1,0))
    turnover_df['away_turned_over_sign'] = - turnover_df['home_turned_over_sign']
    # season_cover_df=(data.set_index('Week').loc[week_start:,:]).reset_index()
    home_turnover_df = (turnover_df.loc[:,['Week','Date','Home ID','home_turned_over_sign']]).rename(columns={'Home ID':'ID','home_turned_over_sign':'turned_over_sign'})
    st.write('checking home turnover section', home_turnover_df[home_turnover_df['ID']==0])
    away_turnover_df = (turnover_df.loc[:,['Week','Date','Away ID','away_turned_over_sign']]).rename(columns={'Away ID':'ID','away_turned_over_sign':'turned_over_sign'})
    st.write('checking away turnover section', away_turnover_df[away_turnover_df['ID']==0])
    season_cover=pd.concat([home_turnover_df,away_turnover_df],ignore_index=True)
    # season_cover_df = pd.melt(season_cover_df,id_vars=['Week', 'home_cover'],value_vars=['Home ID', 'Away ID']).set_index('Week').rename(columns={'value':'ID'}).\
    # drop('variable',axis=1).reset_index().sort_values(by=['Week','ID'],ascending=True)
    return season_cover.sort_values(by=['Week','Date','ID'],ascending=['True','True','True'])

with st.beta_expander('Last Game Turnover'):
    turnover=spread_workings(data)
    # st.write('lets have a look at the data',data[(data['Home Team']=='Arizona Cardinals') | (data['Away Team']=='Arizona Cardinals')].sort_values(by=['Week','Date','Time']))
    turnover_1 = turnover_workings(turnover,-1)
    # st.write('turnover 1', turnover_1[turnover_1['ID']==0])
    
    turnover_2=turnover_2(turnover_1)
    # st.write('turnover 2 NEXT CHECK', turnover_2[turnover_2['ID']==0])
    turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')
    # st.write('this is last game turnover')
    st.write(turnover_3.sort_values(by=['ID','Week'],ascending=['True','True']))

def season_cover_2(season_cover_df,column_name):    
    # https://stackoverflow.com/questions/54993050/pandas-groupby-shift-and-cumulative-sum
    # season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].transform(lambda x: x.cumsum().shift())
    # THE ABOVE DIDN'T WORK IN 2020 PRO FOOTBALL BUT DID WORK IN 2019 DO NOT DELETE FOR INFO PURPOSES
    season_cover_df[column_name] = season_cover_df.groupby (['ID'])[column_name].apply(lambda x: x.cumsum().shift())
    season_cover_df=season_cover_df.reset_index().sort_values(by=['Week','Date','ID'],ascending=True).drop('index',axis=1)
    # Be careful with this if you want full season, season to date cover, for week 17, it is season to date up to week 16
    # if you want full season, you have to go up to week 18 to get the full 17 weeks, just if you want to do analysis on season covers
    return season_cover_df

with st.beta_expander('Season to date Cover'):
    # st.write('this is spread #0', spread)
    spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',0)

    # st.write ('this is spread showing the actual cover in the week',spread_1[spread_1['ID']==31])
    # test_1 = spread_1.reset_index().drop('index',axis=1)
    # st.write(spread_1)
    # test_1['season cover test'] = test_1.groupby ('ID')['cover'].apply(lambda x: x.cumsum().shift())
    # st.write( test_1.groupby ('ID')['cover'].transform(lambda x: x.cumsum().shift()) )
    # st.write(test_1[test_1['ID']==0])
    # st.write(test_1[test_1['ID']==31])
    # st.write(test_1[test_1['ID']==17])

    spread_2=season_cover_2(spread_1,'cover')
    # st.write('this cumsum cover to date and shifted')
    # st.write(spread_2[spread_2['ID']==31])
    spread_3=season_cover_3(spread_2,'cover_sign','cover')
    # st.write('this is season to date cover')
    st.write(spread_3.sort_values(by=['ID','Week'],ascending=['True','True']))


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
# st.write('LOOKS OKMatrix Df check for date time', matrix_df.head())
# matrix_df=matrix_df.reset_index().rename(columns={'index':'unique_match_id'})
matrix_df_1=matrix_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv','Date','Time','Home Points','Away Points']].copy()
# st.write('checking #1 matrix_df_1',matrix_df_1.head())
# test_4=matrix_df_1[matrix_df_1['Week'].between(-3,finish)].copy()

with st.beta_expander('Games Played to be used in Matrix Multiplication'):
    first_qtr=matrix_df_1.copy()
    start=-3
    finish=0
    first_4=first_qtr[first_qtr['Week'].between(start,finish)].copy()
    # st.write('checking first 4 #2',first_4)
    def games_matrix_workings(first_4):
        group_week = first_4.groupby('Week')
        raw_data_2=[]
        game_weights = iter([-0.125, -0.25,-0.5,-1])
        for name, group in group_week:
            group['game_adj']=next(game_weights)
            raw_data_2.append(group)

        df3 = pd.concat(raw_data_2, ignore_index=True)
        adj_df3=df3.loc[:,['Home ID', 'Away ID', 'game_adj']].copy()
        test_adj_df3 = adj_df3.rename(columns={'Home ID':'Away ID', 'Away ID':'Home ID'})
        concat_df_test=pd.concat([adj_df3,test_adj_df3]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
        test_concat_df_test=concat_df_test.groupby('Home ID')['game_adj'].sum().abs().reset_index()
        test_concat_df_test['Away ID']=test_concat_df_test['Home ID']
        full=pd.concat([concat_df_test,test_concat_df_test]).sort_values(by=['Home ID', 'game_adj'],ascending=[True,False])
        # st.write('full',full)
        full_stack=pd.pivot_table(full,index='Away ID', columns='Home ID',aggfunc='sum')
        # st.write('full stack pivot THIS IS WHERE ISSUE IS', full_stack)
        # st.write('Check sum looks good all zero', full_stack.sum())
        full_stack=full_stack.fillna(0)
        full_stack.columns = full_stack.columns.droplevel(0)
        return full_stack
    st.write('Check that First_4 is working', first_4)
    full_stack=games_matrix_workings(first_4)
    st.write('Check sum if True all good', full_stack.sum().sum()==0)
    st.write('this is 1st part games played, need to automate this for every week')
    # st.write('this is the GamesMatrixWorkings Function',full_stack)

# st.header('OVER HERE')
# test_first_section=matrix_df_1[matrix_df_1['Week'].between(-3,0)]
# st.write('check this out', test_first_section)
# test_full_game_matrix=games_matrix_workings(test_first_section)
# st.write('error here which is being caused by the games matrix workings function',test_full_game_matrix)


# st.write('checking the test df for unique match id', test_df)
with st.beta_expander('CORRECT Testing reworking the DataFrame'):
    test_df['at_home'] = 1
    test_df['at_away'] = -1
    test_df['home_pts_adv'] = 3
    test_df['away_pts_adv'] = -3
    test_df['away_spread']=-test_df['Spread']
    test_df=test_df.rename(columns={'Spread':'home_spread'})
    # st.write('checking for unique match id',test_df)
    test_df_1=test_df.loc[:,['unique_match_id','Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()
    
    # st.write(test_df_1.sort_values(by=['ID','Week'],ascending=True))
    test_df_home=test_df_1.loc[:,['Week','Home ID','at_home','home_spread','home_pts_adv']].rename(columns={'Home ID':'ID','at_home':'home','home_spread':'spread','home_pts_adv':'home_pts_adv'}).copy()
    test_df_away=test_df_1.loc[:,['Week','Away ID','at_away','away_spread','away_pts_adv']].rename(columns={'Away ID':'ID','at_away':'home','away_spread':'spread','away_pts_adv':'home_pts_adv'}).copy()
    test_df_2=pd.concat([test_df_home,test_df_away],ignore_index=True)
    test_df_2=test_df_2.sort_values(by=['ID','Week'],ascending=True)
    test_df_2['spread_with_home_adv']=test_df_2['spread']+test_df_2['home_pts_adv']
    st.write(test_df_2)

def test_4(matrix_df_1):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    matrix_df_1['adj_spread']=matrix_df_1['spread_with_home_adv'].rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    return matrix_df_1


with st.beta_expander('CORRECT Power Ranking to be used in Matrix Multiplication'):
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
    st.write('power ranking',df_power.sort_values(by=['ID','Week'],ascending=[True,True]))




with st.beta_expander('CORRECT Power Ranking Matrix Multiplication'):
    # https://stackoverflow.com/questions/62775018/matrix-array-multiplication-whats-excel-doing-mmult-and-how-to-mimic-it-in#62775508
    # st.write('check')
    inverse_matrix=[]
    power_ranking=[]
    list_inverse_matrix=[]
    list_power_ranking=[]
    power_df=df_power.loc[:,['Week','ID','adj_spread']].copy()
    games_df=matrix_df_1.copy()
    first=list(range(-3,18))
    last=list(range(0,21))
    for first,last in zip(first,last):
        # st.header('start xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # st.write('week no.', last)
        # st.write('this is first',first)
        # st.write('this is last',last)
        first_section=games_df[games_df['Week'].between(first,last)]
        # st.write(first_section)
        full_game_matrix=games_matrix_workings(first_section)
        # st.write('this if full game matrix',full_game_matrix,'this is current week',last)
        adjusted_matrix=full_game_matrix.loc[0:30,0:30]
        # st.write('this is the last number',last)
        # st.write(adjusted_matrix)
        df_inv = pd.DataFrame(np.linalg.pinv(adjusted_matrix.values), adjusted_matrix.columns, adjusted_matrix.index)
        # st.write('this is the inverse matrix',df_inv, 'last number ie current week', last)
        # st.write('this is shape of inverse matrix', df_inv.shape)

        power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID')\
        .drop('Week',axis=1).rename(columns={'adj_spread':0}).loc[:30,:]
        
        # st.write('this is the power_df_week',power_df_week)
        # st.write('this is the shape', power_df_week.shape)
        # st.write(pd.DataFrame(power_df_week).dtypes)
        # st.write('this is PD Dataframe power df week',pd.DataFrame(power_df_week) )
        result = df_inv.dot(pd.DataFrame(power_df_week))
        # st.header('this is result of matrix multplication')
        # st.write(result)
        result.columns=['power']
        avg=(result['power'].sum())/32
        result['avg_pwr_rank']=(result['power'].sum())/32
        result['final_power']=result['avg_pwr_rank']-result['power']
        df_pwr=pd.DataFrame(columns=['final_power'],data=[avg])
        result=pd.concat([result,df_pwr],ignore_index=True)
        result['week']=last+1
        power_ranking.append(result)
        # st.write('week no.', last)
        # st.header('end xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    power_ranking_combined = pd.concat(power_ranking).reset_index().rename(columns={'index':'ID'})
    st.write('power ranking combined', power_ranking_combined)
    
with st.beta_expander('Adding Power Ranking to Matches'):
    matches_df = spread.copy()
    st.write('This is matches_df', matches_df.head())
    st.write('This is power ranking combined', power_ranking_combined.head())
    home_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Home ID'})
    away_power_rank_merge=power_ranking_combined.loc[:,['ID','week','final_power']].copy().rename(columns={'week':'Week','ID':'Away ID'})
    updated_df=pd.merge(matches_df,home_power_rank_merge,on=['Home ID','Week']).rename(columns={'final_power':'home_power'})
    updated_df=pd.merge(updated_df,away_power_rank_merge,on=['Away ID','Week']).rename(columns={'final_power':'away_power'})
    updated_df['calculated_spread']=updated_df['away_power']-updated_df['home_power']
    updated_df['spread_working']=updated_df['home_power']-updated_df['away_power']+updated_df['Spread']
    updated_df['power_pick'] = np.where(updated_df['spread_working'] > 0, 1,
    np.where(updated_df['spread_working'] < 0,-1,0))
    st.write(updated_df)

with st.beta_expander('Adding Season to Date Cover to Matches'):
    # df = pd.DataFrame([['mon',19,'cardinals', 3], ['tue',20,'patriots', 4], ['wed',20,'patriots', 5]], columns=['date','week','team', 'stdc'])
    # st.write('df1',df)
    # df2 = pd.DataFrame([['sun',18,'saints'], ['tue',20,'patriots'], ['wed',20,'patriots']], columns=['date','week','team'])
    # st.write('df2',df2)
    # df3=df2.merge(df,on=['date','week','team'], how='left')
    # st.write('merged on left',df3)  # merges on columns A

    # st.write('this is season to date cover', spread_3)
    stdc_home=spread_3.rename(columns={'ID':'Home ID'})
    stdc_home['cover_sign']=-stdc_home['cover_sign']
    stdc_away=spread_3.rename(columns={'ID':'Away ID'})
    updated_df=updated_df.drop(['away_cover'],axis=1)
    # st.header('Check')
    # st.write('check updated df #1',updated_df)
    updated_df=updated_df.rename(columns={'home_cover':'home_cover_result'})
    updated_df=updated_df.merge(stdc_home,on=['Date','Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign':'home_cover_sign'})
    # st.write('check updated df #2', updated_df)
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
    stdc_df=stdc_df.loc[:,['Week','Team','cover']].copy()
    # stdc_df['last_week']=
    # stdc_df['Week']=stdc_df['Week'].replace({17:'week_17'})
    
    stdc_df['average']=stdc_df.groupby('Team')['cover'].transform(np.mean)
    # st.write(stdc_df.sort_values(by=['Team','Week']))
    
    stdc_pivot=pd.pivot_table(stdc_df,index='Team', columns='Week')
    stdc_pivot.columns = stdc_pivot.columns.droplevel(0)
    # st.write(stdc_pivot)

    chart_cover= alt.Chart(stdc_df).mark_rect().encode(alt.X('Week:O',axis=alt.Axis(title='Week',labelAngle=0)),
    alt.Y('Team',sort=alt.SortField(field='average', order='descending')),color=alt.Color('cover:Q',scale=alt.Scale(scheme='redyellowgreen')))
    # https://altair-viz.github.io/gallery/layered_heatmap_text.html
    # https://vega.github.io/vega/docs/schemes/
    text_cover=chart_cover.mark_text().encode(text=alt.Text('cover:N'),color=alt.value('black'))
    st.altair_chart(chart_cover + text_cover,use_container_width=True)

    
with st.beta_expander('Adding Turnover to Matches'):
    st.write('this is turnovers', turnover_3)
    turnover_matches = turnover_3.loc[:,['Date','Week','ID','prev_turnover', 'turnover_sign']].copy()
    turnover_home=turnover_matches.rename(columns={'ID':'Home ID'})
    
    turnover_away=turnover_matches.rename(columns={'ID':'Away ID'})
    turnover_away['turnover_sign']=-turnover_away['turnover_sign']
    updated_df=pd.merge(updated_df,turnover_home,on=['Date','Week','Home ID'],how='left').rename(columns={'prev_turnover':'home_prev_turnover','turnover_sign':'home_turnover_sign'})
    updated_df=pd.merge(updated_df,turnover_away,on=['Date','Week','Away ID'],how='left').rename(columns={'prev_turnover':'away_prev_turnover','turnover_sign':'away_turnover_sign'})
    # st.write('check matches week 20', updated_df)
    # TEST Workings
    # st.write('check that Turnover coming in correctly', updated_df[updated_df['Week']==18])
    # st.write('Check Total')
    # st.write('home',updated_df['home_turnover_sign'].sum())
    # st.write('away',updated_df['away_turnover_sign'].sum())
    # turnover_excel=test_data_2020.loc[:,['Week','Home ID','Home Team', 'Away ID', 'Away Team','excel_home_prev_turnover','excel_away_prev_turnover','excel_home_turnover_sign','excel_away_turnover_sign']].copy()
    # test_turnover=pd.merge(updated_df,turnover_excel)
    # test_turnover['test_1']=test_turnover['home_prev_turnover']-test_turnover['excel_home_prev_turnover']
    # test_turnover['test_2']=test_turnover['away_prev_turnover']-test_turnover['excel_away_prev_turnover']
    # st.write(test_turnover[test_turnover['test_1']!=0])
    # st.write(test_turnover[test_turnover['test_2']!=0])
    # st.write(test_turnover)

with st.beta_expander('Betting Slip Matches'):
    betting_matches=updated_df.loc[:,['Week','Date','Home ID','Home Team','Away ID', 'Away Team','Spread','Home Points','Away Points',
    'home_power','away_power','home_cover','away_cover','home_turnover_sign','away_turnover_sign','home_cover_sign','away_cover_sign','power_pick','home_cover_result']]
    # st.write('check for duplicate home cover', betting_matches)
    betting_matches['total_factor']=betting_matches['home_turnover_sign']+betting_matches['away_turnover_sign']+betting_matches['home_cover_sign']+\
    betting_matches['away_cover_sign']+betting_matches['power_pick']
    betting_matches['bet_on'] = np.where(betting_matches['total_factor']>2,betting_matches['Home Team'],np.where(betting_matches['total_factor']<-2,betting_matches['Away Team'],''))
    betting_matches['bet_sign'] = (np.where(betting_matches['total_factor']>2,1,np.where(betting_matches['total_factor']<-2,-1,0)))
    betting_matches['bet_sign'] = betting_matches['bet_sign'].astype(float)
    betting_matches['home_cover'] = betting_matches['home_cover'].astype(float)
    # st.write('this is bet sign',betting_matches['bet_sign'].dtypes)
    # st.write('this is home cover',betting_matches['home_cover'].dtypes)
    betting_matches['result']=betting_matches['home_cover_result'] * betting_matches['bet_sign']
    st.write('testing sum of betting result',betting_matches['result'].sum())

    # this is for graphing anlaysis on spreadsheet
    betting_matches['bet_sign_all'] = (np.where(betting_matches['total_factor']>0,1,np.where(betting_matches['total_factor']<-0,-1,0)))
    betting_matches['result_all']=betting_matches['home_cover_result'] * betting_matches['bet_sign_all']
    st.write('testing sum of betting all result',betting_matches['result_all'].sum())
    # st.write('testing factor')
    # st.write(betting_matches['total_factor'].sum())
    # cols_to_move=[]
    # cols = cols_to_move + [col for col in data_4 if col not in cols_to_move]
    # data_5=data_4[cols]
    st.write(betting_matches)
    st.write( betting_matches[(betting_matches['Home Team']=='Arizona Cardinals') | (betting_matches['Away Team']=='Arizona Cardinals')].set_index('Week') )


with st.beta_expander('Analysis of Betting Results across 1 to 5 factors'):
    matches_in_regular_season= (32 * 16) / 2
    matches_in_playoffs = 11
    total_matches =matches_in_regular_season + matches_in_playoffs
    st.write('total_matches',total_matches)
    analysis=betting_matches.copy()
    totals = analysis.groupby('total_factor').agg(winning=('result_all','count'))
    totals_1=analysis.groupby([analysis['total_factor'].abs(),'result_all']).agg(winning=('result_all','count')).reset_index()
    totals_1['result_all']=totals_1['result_all'].replace({0:'tie',1:'win',-1:'lose'})
    st.write('shows the number of games at each factor level')
    st.write(totals.rename(columns={'winning':'number_of_games'}))
    st.write('sum of each factor level should correspond to table above',totals_1)
    st.write('sum of winning column should be 267 I think',totals_1['winning'].sum())
    st.write('count of week column should be 267',analysis['Week'].count())

with st.beta_expander('Analysis of Factors'):
    analysis_factors = betting_matches.copy()
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
        df_table_1.loc['Total']=df_table_1.sum()
        # st.write('latest', df_table_1)
        df_table_1.loc['No. of Bets Made'] = df_table_1.loc[[1,-1]].sum() 
        df_table_1.loc['% Winning'] = df_table_1.loc[1] / df_table_1.loc['No. of Bets Made']
        return df_table_1
    total_factor_table = analysis_factor_function(analysis_factors)     
    st.write('This is the total number of matches broken down by Factor result')
    cols_to_move=['total_turnover','total_season_cover','power_ranking_success?']
    total_factor_table = total_factor_table[ cols_to_move + [ col for col in total_factor_table if col not in cols_to_move ] ]
    st.write(total_factor_table)
    factor_bets = (analysis_factors[analysis_factors['bet_sign']!=0]).copy()
    bets_made_factor_table = analysis_factor_function(factor_bets)
    # cols_to_move=['total_turnover','total_season_cover','power_ranking_success?']
    bets_made_factor_table = bets_made_factor_table[ cols_to_move + [ col for col in bets_made_factor_table if col not in cols_to_move ] ]
    st.write('This is the matches BET ON broken down by Factor result')
    st.write(bets_made_factor_table)

with st.beta_expander('Checking Performance where Total Factor = 2 or 3'):
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
    df_factor_table_1.loc['Total']=df_factor_table_1.sum()
    # st.write('latest', df_factor_table_1)
    df_factor_table_1.loc['No. of Bets Made'] = df_factor_table_1.loc[[1,-1]].sum() 
    df_factor_table_1.loc['% Winning'] = df_factor_table_1.loc[1] / df_factor_table_1.loc['No. of Bets Made']
    cols_to_move=['total_turnover','total_season_cover','power_diagnostic']
    df_factor_table_1 = df_factor_table_1[ cols_to_move + [ col for col in df_factor_table_1 if col not in cols_to_move ] ]
    st.write(df_factor_table_1)

with st.beta_expander('Power Ranking by Week'):
    power_week=power_ranking_combined.copy()
    # st.write('power', power_week)

    # pivot_df=power_week.loc[:,['ID','final_power','week']].copy()
    team_names_id=team_names_id.rename(columns={'Away Team':'Team'})
    pivot_df=pd.merge(power_week,team_names_id, on='ID')
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

with st.beta_expander('Underdog Analyis'):
    underdog_df = betting_matches.copy()
    # underdog_df['home_underdog_away_favourite']=np.where(underdog_df['Spread']>0,1,np.NaN)
    # underdog_df['away_underdog_home_favourite']=np.where(underdog_df['Spread']<0,1,np.NaN)
    
    filter_bets_underdog=(underdog_df['Spread']>0.1) &(underdog_df['bet_sign']!=0)
    filter_bets_favourite=(underdog_df['Spread']<0.1) &(underdog_df['bet_sign']!=0)
    underdog_df['home_underdog_bet_result']=underdog_df['result'].where(filter_bets_underdog)
    underdog_df['home_favourite_bet_result']=underdog_df['result'].where(filter_bets_favourite)
    underdog_df['home_underdog_all_result']=underdog_df['home_cover_result'].where(underdog_df['Spread']>0.1)
    underdog_df['home_favourite_all_result']=underdog_df['home_cover_result'].where(underdog_df['Spread']<0.1)
    underdog_table = underdog_df['home_underdog_bet_result'].value_counts()
    home_fav_bet = underdog_df['home_favourite_bet_result'].value_counts()
    home_underdog_all = underdog_df['home_underdog_all_result'].value_counts()
    home_fav_all = underdog_df['home_favourite_all_result'].value_counts()
    underdog_results = pd.concat([underdog_table,home_fav_bet,home_underdog_all,home_fav_all],axis=1)
    underdog_results=underdog_results.sort_index(ascending=False)
    underdog_results.loc['Total']=underdog_results.sum()
    st.write(underdog_results)



with st.beta_expander('Pro Football Ref Scraper'):
    pass
    # def fbref_scraper():
    #     test = pd.read_html('https://www.pro-football-reference.com/years/2019/games.htm')[0]
    #     test.to_pickle('C:/Users/Darragh/Documents/Python/NFL/pro_football_ref/nfl_2019.pkl')
    #     return test  
        
    # test=fbref_scraper()
    