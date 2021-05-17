import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import os
import base64

st.set_page_config(layout="wide")


@st.cache
def read_data(file):
    return pd.read_excel(file) 
data_2019 = read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2019_Data.xlsx').copy()
# data_2020=read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2020_Data_Adj_week_zero.xlsx').copy()
data_2020=read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2020_Data.xlsx').copy()
test_data_2020=read_data('C:/Users/Darragh/Documents/Python/NFL/NFL_2020_Data_Test.xlsx').copy()

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
st.write('spread',spread)

with st.beta_expander('Season to date Cover'):
    spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',1)
    spread_2=season_cover_2(spread_1,'cover')
    spread_3=season_cover_3(spread_2,'cover_sign','cover')
    st.write('this is season to date cover')
    st.write(spread_3.sort_values(by=['ID','Week'],ascending=['True','True']))
    # st.write('Test workings')
    # st.write(test_data_2020)


with st.beta_expander('Last Game Turnover'):
    turnover=spread_workings(data_2020)
    turnover_1 = season_cover_workings(turnover,'home_turnover','away_turnover','turnover',-1)
    turnover_2=turnover_2(turnover_1)
    turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')
    st.write('this is last game turnover')
    st.write(turnover_3.sort_values(by=['ID','Week'],ascending=['True','True']))

matrix_df=spread_workings(data_2020)
test_df = matrix_df.copy()
matrix_df['at_home'] = 1
matrix_df['at_away'] = -1
matrix_df['home_pts_adv'] = -3
matrix_df['away_pts_adv'] = 3
matrix_df['away_spread']=-matrix_df['Spread']
matrix_df=matrix_df.rename(columns={'Spread':'home_spread'})
matrix_df_1=matrix_df.loc[:,['Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()

with st.beta_expander('Games Played to be used in Matrix Multiplication'):
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
    st.write('Check sum if True all good', full_stack.sum().sum()==0)
    st.write('this is 1st part games played, need to automate this for every week')
    st.write(full_stack)


with st.beta_expander('CORRECT Testing reworking the DataFrame'):
    test_df['at_home'] = 1
    test_df['at_away'] = -1
    test_df['home_pts_adv'] = 3
    test_df['away_pts_adv'] = -3
    test_df['away_spread']=-test_df['Spread']
    test_df=test_df.rename(columns={'Spread':'home_spread'})
    test_df_1=test_df.loc[:,['Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()
    
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
    st.write('power ranking',df_power)

with st.beta_expander('CORRECT Power Ranking Matrix Multiplication'):
    # https://stackoverflow.com/questions/62775018/matrix-array-multiplication-whats-excel-doing-mmult-and-how-to-mimic-it-in#62775508
    inverse_matrix=[]
    power_ranking=[]
    list_inverse_matrix=[]
    list_power_ranking=[]
    power_df=df_power.loc[:,['Week','ID','adj_spread']].copy()
    games_df=matrix_df_1.copy()
    first=list(range(-3,18))
    last=list(range(0,21))
    for first,last in zip(first,last):
        # st.write('this is first',first)
        # st.write('this is last',last)
        first_section=games_df[games_df['Week'].between(first,last)]
        # st.write(first_section)
        full_game_matrix=games_matrix_workings(first_section)
        # st.write(full_game_matrix)
        adjusted_matrix=full_game_matrix.loc[0:30,0:30]
        # st.write('this is the last number',last)
        # st.write(adjusted_matrix)
        df_inv = pd.DataFrame(np.linalg.pinv(adjusted_matrix.values), adjusted_matrix.columns, adjusted_matrix.index)
        # st.write('this is the inverse matrix',df_inv, 'number', last)
        power_df_week=power_df[power_df['Week']==last].drop_duplicates(subset=['ID'],keep='last').set_index('ID').drop('Week',axis=1).rename(columns={'adj_spread':0}).loc[:30,:]
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
    st.write('power ranking combined', power_ranking_combined)
    
with st.beta_expander('Adding Power Ranking to Matches'):
    matches_df = spread.copy()
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
    st.write('this is season to date cover', spread_3)
    stdc_home=spread_3.rename(columns={'ID':'Home ID'})
    stdc_away=spread_3.rename(columns={'ID':'Away ID'})
    updated_df=pd.merge(updated_df,stdc_home,on=['Week','Home ID'],how='left').rename(columns={'cover':'home_cover','cover_sign':'home_cover_sign'})
    updated_df=pd.merge(updated_df,stdc_away,on=['Week','Away ID'],how='left').rename(columns={'cover':'away_cover','cover_sign':'away_cover_sign'})
    st.write('check that STDC coming in correctly', updated_df)
    st.write('Check that home STDC sign is working correctly')    
    
