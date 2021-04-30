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
st.write('spread',spread)

with st.beta_expander('Season to date Cover'):
    spread_1 = season_cover_workings(spread,'home_cover','away_cover','cover',1)
    spread_2=season_cover_2(spread_1,'cover')
    spread_3=season_cover_3(spread_2,'cover_sign','cover')
    st.write('this is season to date cover')
    st.write(spread_3.sort_values(by=['ID','Week'],ascending=['True','True']))

with st.beta_expander('Last Game Turnover'):
    turnover=spread_workings(data_2020)
    turnover_1 = season_cover_workings(turnover,'home_turnover','away_turnover','turnover',-1)
    turnover_2=turnover_2(turnover_1)
    turnover_3=season_cover_3(turnover_2,'turnover_sign','prev_turnover')
    st.write('this is last game turnover')
    st.write(turnover_3.sort_values(by=['ID','Week'],ascending=['True','True']))

matrix_df=spread_workings(data_2020)

def calc(x):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    return np.sum(weights*x) / sum_weights        

def test(matrix_df_1):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    matrix_df_1['test weight spread']=matrix_df_1['spread'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
    return matrix_df_1

def test_1(matrix_df_1):
    weights = np.array([0.125, 0.25,0.5,1])
    sum_weights = np.sum(weights)
    # matrix_df_1['games_weighted']=matrix_df_1['home'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    matrix_df_1['spread_weighted']=matrix_df_1['spread'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    matrix_df_1['home_adv_weighted']=matrix_df_1['home_pts_adv'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    matrix_df_1['adj_spread'] = matrix_df_1['home_adv_weighted'] + matrix_df_1['spread_weighted']
    return matrix_df_1

def test_2(matrix_df_1):
    weights_1 = np.array([0.125,0,0,0 ])
    weights_2 = np.array([0.125, 0.25,0,0])
    weights_3 = np.array([0.125, 0.25,0.5,0])
    weights_4 = np.array([0.125, 0.25,0.5,1])
    sum_weights_1 = np.sum(weights_1)
    sum_weights_2 = np.sum(weights_2)
    sum_weights_3 = np.sum(weights_3)
    sum_weights_4 = np.sum(weights_4)

    matrix_df_1['games_weighted']=matrix_df_1['spread'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    # matrix_df_1['games_weighted']=matrix_df_1['home'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    matrix_df_1['spread_weighted']=matrix_df_1['spread'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    # matrix_df_1['home_adv_weighted']=matrix_df_1['home_pts_adv'].fillna(0).rolling(window=4, center=False).apply(lambda x: np.sum(weights*x), raw=False)
    # matrix_df_1['adj_spread'] = matrix_df_1['home_adv_weighted'] + matrix_df_1['spread_weighted']
    return matrix_df_1

# Why are both of the above functions working????????????? the sum_weights is taken out in test function

matrix_df['at_home'] = 1
matrix_df['at_away'] = -1
matrix_df['home_pts_adv'] = -3
matrix_df['away_pts_adv'] = 3
matrix_df['away_spread']=-matrix_df['Spread']
matrix_df=matrix_df.rename(columns={'Spread':'home_spread'})
matrix_df_1=matrix_df.loc[:,['Week','Home ID','Away ID','at_home','at_away','home_spread','away_spread','home_pts_adv','away_pts_adv']].copy()
# st.write('matrix df1',matrix_df_1)


with st.beta_expander('Power Ranking to be used in Matrix Multiplication'):
    matrix_df_home=matrix_df_1.loc[:,['Week','Home ID','at_home','home_spread','home_pts_adv']].rename(columns={'Home ID':'ID','at_home':'home','home_spread':'spread','home_pts_adv':'home_pts_adv'}).copy()
    matrix_df_away=matrix_df_1.loc[:,['Week','Away ID','at_away','away_spread','away_pts_adv']].rename(columns={'Away ID':'ID','at_away':'home','away_spread':'spread','away_pts_adv':'home_pts_adv'}).copy()
    matrix_df_2=pd.concat([matrix_df_home,matrix_df_away],ignore_index=True)
    weights = np.array([0.125, 0.25,0.5,1]) # the order mattered!! took me a while to figure this out
    sum_weights = np.sum(weights)
    matrix_df_2=matrix_df_2.sort_values(by=['ID','Week'],ascending=True)
    # https://stackoverflow.com/questions/9621362/how-do-i-compute-a-weighted-moving-average-using-pandas
    # st.write(matrix_df_2)
    grouped = matrix_df_2.groupby('ID')
    # raw_data=[]
    raw_data_1=[]
    for name, group in grouped:
        # st.write('name',name)
        # st.write('group',group)
        # raw_data.append(test(group))
        raw_data_1.append(test_1(group))
        # st.write(test(group))
    # df1 = pd.concat(raw_data, ignore_index=True)
    df2 = pd.concat(raw_data_1, ignore_index=True)
    # st.write('this is df1',df1)
    st.write('This is power ranking to be used in Matrix Multiplaction',df2)
    st.write('i checked that ID no.2 and ID no.5 equal the spreadsheet')
    st.write('do i have a problem if i have a blank gameweek, should i insert NaN just thinking of inverse matrix....it has to add up to 0')
    st.write('not necessarily as long as get games played to add up to zero ...have the power points calculated')

with st.beta_expander('Games Played to be used in Matrix Multiplication'):
    first_qtr=matrix_df_1.copy()
    first_4=first_qtr[first_qtr['Week']<1].copy()
    group_week = first_4.groupby('Week')
    st.write('How do I get this to work just to enter 0.125 0.25 etc')
    raw_data_2=[]
    
    for name, group in group_week:
        st.write('Group!', group)
        for i in np.array([0.125, 0.25,0.5,1]):
            group['game_adj']=i
            st.write('Group after game_Adj', group)
        raw_data_2.append(group)

    



    df3 = pd.concat(raw_data_2, ignore_index=True)
    st.write(df3)
    # st.write(first_qtr[first_qtr['Week']<1])
    # st.write(first_qtr)

