from operator import is_
import pandas as pd
import numpy as np
import streamlit as st
import datetime
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

current_date='2022-12-20'
# run the fbref function and bring in date of last completed week, and then check the test NA 

# df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_24_09_22.xlsx')
# df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_14_10_22.xlsx')
# df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_05_11_22.xlsx')
# df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_26_11_22.xlsx')
df = read_data('C:/Users/Darragh/Documents/Python/NFL/nfl_historical_odds_20_12_22.xlsx')
df=df.copy()

with st.expander('pro football workings'):
    year=2022
    url=f'https://www.pro-football-reference.com/years/{year}/games.htm'
    # st.write('url', url)

    def fbref_scraper_csv(url):
            test = pd.read_html(url)[0]
            # test.to_excel('C:/Users/Darragh/Documents/Python/NFL/nfl_2022_scores.xlsx')
            test.to_csv(f'C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_{year}.csv')
            # test.to_csv('https://github.com/ZeNoonan/nfl/blob/main/nfl_2021.csv')
            return test

    # fbref_scraper_csv(url)
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
    file_2022 = pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2022.csv')
    # combined_file = pd.concat([file_2006,file_2007,file_2008,file_2009,file_2010,file_2011,file_2012,file_2013,
    # file_2014,file_2015,file_2016,file_2017,file_2018,file_2019,file_2020,file_2021])
    # combined_file.to_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2006_2021.csv')
    combined_file=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/nfl_feustel_scores_2006_2021.csv')
    combined_file=pd.concat([combined_file,file_2022])
    # st.write('comb files', combined_file)

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
    cleaned_pro_football_file['Date']=pd.to_datetime(cleaned_pro_football_file['Date'],format='%Y-%m-%d')
    df['Date']=pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df.loc [ (df['Date']=='2022-02-13')&(df['Home Team']=='Cincinnati Bengals'), 'Away Team' ] = 'Cincinnati Bengals'
    df.loc [ (df['Date']=='2022-02-13')&(df['Home Team']=='Cincinnati Bengals'), 'Home Team' ] = 'Los Angeles Rams'
    df.loc [ (df['Date']=='2019-02-03')&(df['Home Team']=='Los Angeles Rams'), 'Home Team' ] = 'New England Patriots'
    df.loc [ (df['Date']=='2019-02-03')&(df['Home Team']=='New England Patriots'), 'Away Team' ] = 'Los Angeles Rams'
    df.loc [ (df['Date']=='2018-02-04')&(df['Home Team']=='New England Patriots'), 'Home Team' ] = 'Philadelphia Eagles'
    df.loc [ (df['Date']=='2018-02-04')&(df['Home Team']=='Philadelphia Eagles'), 'Away Team' ] = 'New England Patriots'
    df.loc [ (df['Date']=='2017-02-05')&(df['Home Team']=='Atlanta Falcons'), 'Home Team' ] = 'New England Patriots'
    df.loc [ (df['Date']=='2017-02-05')&(df['Home Team']=='New England Patriots'), 'Away Team' ] = 'Atlanta Falcons'
    df.loc [ (df['Date']=='2015-02-01')&(df['Home Team']=='Seattle Seahawks'), 'Home Team' ] = 'New England Patriots'
    df.loc [ (df['Date']=='2015-02-01')&(df['Home Team']=='New England Patriots'), 'Away Team' ] = 'Seattle Seahawks'
    df.loc [ (df['Date']=='2015-01-04')&(df['Home Team']=='Carolina Panthers'), 'Date' ] = '2015-01-03'
    df.loc [ (df['Date']=='2015-01-03')&(df['Home Team']=='Dallas Cowboys'), 'Date' ] = '2015-01-04'
    df.loc [ (df['Date']=='2014-02-02')&(df['Home Team']=='Denver Broncos'), 'Home Team' ] = 'Seattle Seahawks'
    df.loc [ (df['Date']=='2014-02-02')&(df['Home Team']=='Seattle Seahawks'), 'Away Team' ] = 'Denver Broncos'
    df.loc [ (df['Date']=='2013-02-03')&(df['Home Team']=='San Francisco 49ers'), 'Home Team' ] = 'Baltimore Ravens'
    df.loc [ (df['Date']=='2013-02-03')&(df['Home Team']=='Baltimore Ravens'), 'Away Team' ] = 'San Francisco 49ers'
    df.loc [ (df['Date']=='2010-02-07')&(df['Home Team']=='Indianapolis Colts'), 'Home Team' ] = 'New Orleans Saints'
    df.loc [ (df['Date']=='2010-02-07')&(df['Home Team']=='New Orleans Saints'), 'Away Team' ] = 'Indianapolis Colts'
    df.loc [ (df['Date']=='2008-02-03')&(df['Home Team']=='New England Patriots'), 'Home Team' ] = 'New York Giants'
    df.loc [ (df['Date']=='2008-02-03')&(df['Home Team']=='New York Giants'), 'Away Team' ] = 'New England Patriots'


    # df.loc [ (df['Date']=='2015-01-04')&(df['Home Team']=='New England Patriots'), 'Away Team' ] = 'Seattle Seahawks'

    # st.write('cleaned file', cleaned_pro_football_file.sort_values(by='Date',ascending=False).set_index(['Week','Date','Winner/tie','Loser/tie']))
    # st.write('merge with this file', df.set_index(['Date','Home Team','Away Team']))
    df = pd.merge(cleaned_pro_football_file, df, how='outer')
    # st.write('check this to see what the problem is', df.set_index(['Week','Date','Winner/tie','Loser/tie']))
    df=df.loc[(df['Date']<current_date)] # WATCH THIS FOR WHEN BRINGING IN NEW WEEK
    # st.write('merged file', df.set_index(['Week','Date','Winner/tie','Loser/tie']))
    # st.write('checking that loc works for date',df.loc[(df['Date']>'2022-11-05')])
    # st.write('want to clean this up a bit get rid of the future gameweeks')
    st.write('then focus on the na games where havent merged properly like the final where on neutral field')
    # st.write(df.loc[df['Winner/tie'].isna()])
    
    st.write('Checking NAs in turnover',df.loc[df['turnover'].isna()])
    st.write('Checking NAs in home score',df.loc[df['Home Score'].isna()])
    # st.write('show individual problem games',df.loc[(df['Date']>'2015-01-02') & (df['Date']<'2015-01-05') ])
    # st.write('show individual problem games',df.loc[(df['Date']=='2008-02-03')  ])


# st.write('df raw data 145', df.head(4))
# st.download_button(label="Download data as CSV",data=df.head().to_csv().encode('utf-8'),
# file_name='df_spread.csv',mime='text/csv',key='after_merge_spread')

df['home_score_margin_of_victory']=df['Home Points']-df['Away Points']
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
# st.write('df raw data', df.head(4))
# dummy_df=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/df_dummy_data_1.csv')
dummy_df=pd.read_csv('C:/Users/Darragh/Documents/Python/NFL/df_dummy_data_2.csv')
dummy_df['Date'] = [datetime.datetime.strptime(x, '%d/%m/%Y') for x in dummy_df['Date']]
dummy_df['Date']=pd.to_datetime(dummy_df['Date']).dt.normalize()

# st.write('dummy df', dummy_df)

def avg_score(df):
    df['avg_home_score']=df['Home Score'].expanding().mean()
    df['avg_away_score']=df['Away Score'].expanding().mean()
    return df
df = avg_score(df)
dummy_df=avg_score(dummy_df)
# st.write('dummy', dummy_df)
cols_to_move=['Date','Home Team','Away Team','unique_id','Home Score','Away Score','avg_home_score','avg_away_score']
cols = cols_to_move + [col for col in df if col not in cols_to_move]
df=df[cols]

with st.expander('raw data'):

    st.write('looking at matches where turnoves was greater than 6',df[df['turnover']>6])
    # grouped = df.groupby('season_year')

with st.expander('Turnover 2 Variable Regression'):

    # st.write('data', df)
    def regression_function(df):
        regression_data=df.loc[:,['turnover','home_score_margin_of_victory']]
        return regression_data
    
    def regression_function_1(regression_data):
        regression_graph_data=regression_data.copy()
        regression_graph_data['home_score_margin_of_victory']=regression_graph_data['home_score_margin_of_victory'].abs()
        # st.write('df turnover', df['turnover'])
        # https://towardsdatascience.com/simple-and-multiple-linear-regression-with-python-c9ab422ec29c
        # turnover_regression = np.polyfit(df['home_score_margin_of_victory'], df['turnover'], 1)
        turnover_regression_1 = np.polyfit(df['turnover'], df['home_score_margin_of_victory'], 1)
        return turnover_regression_1
    # st.write('regression output',turnover_regression)
    st.write('regression output so if you multiply 7 turnovers by slope -4.55 you get -31.5 plus 1.87 = -29.9 ie expect to lose by 29.9 points')
    st.write('also if you multiply -7 turnovers so you got 7 turnovers given to you, multiply by -4.55 equals 31.8 points plus 1.87 equals 32.7 points which you can\
        see on the trend line so you would expect to win by 32.7 points')
    st.write('so actually the minus in front of 4.5 is a red herring, just multiply the turnovers by the 4.5 and then add or subtract the 1.87 depending on whether\
         you are home or away')
    regression_data=regression_function(df)
    turnover_regression_1=regression_function_1(regression_data)
    st.write(turnover_regression_1)
    st.write(turnover_regression_1[0])
    st.write(turnover_regression_1[1])




    # https://altair-viz.github.io/gallery/poly_fit_regression.html
    # rng = np.random.RandomState(1)
    # x = rng.rand(40) ** 2
    # y = 10 - 1.0 / (x + 0.1) + rng.randn(40)
    # source = pd.DataFrame({"x": x, "y": y})
    # st.write('source', source)

    # Define the degree of the polynomial fits
    degree_list = [1, 3, 5]

    base = alt.Chart(regression_data).mark_circle(color="black").encode(alt.X("turnover"), alt.Y("home_score_margin_of_victory"))

    polynomial_fit = [base.transform_regression("turnover", "home_score_margin_of_victory", method="poly", order=order, as_=["turnover", str(order)])
    .mark_line()
    .transform_fold([str(order)], as_=["degree", "home_score_margin_of_victory"])
    .encode(alt.Color("degree:N"))
    for order in degree_list]

    st.altair_chart(alt.layer(base, *polynomial_fit),use_container_width=True)
    st.write('Was just curious if there was some home away effect but looks fairly even in terms of turnovers to points')
    st.altair_chart(alt.Chart(regression_data).mark_circle(color="black").encode(alt.X("turnover"), alt.Y("home_score_margin_of_victory")),use_container_width=True)
    st.write('just interesting to see what the -7 and +7 turnovers were....')
    st.write('looking at matches where turnoves was greater than 6',df[df['turnover']>6])
    st.write('looking at matches where turnoves was greater than 6',df[df['turnover']<-6])

    # st.write('checking out season',df[(df['season_year']<2008)]['turnover'])
    
    # def regression_output(df,year):
    #     return np.polyfit(df[(df['season_year']<2008)]['turnover'], df[(df['season_year'] < year)]['home_score_margin_of_victory'], 1)
    # st.write('max+1', df['season_year'].max()+1)
    # for n in range(2021,df['season_year'].max()+2):
    #     st.write(n)
    def turnover_table(df):
        raw_data=[]
        for n in range(df['season_year'].min(),df['season_year'].max()+1):
            x = np.polyfit(df[(df['season_year']<n+1)]['turnover'], df[(df['season_year'] < n+1)]['home_score_margin_of_victory'], 1)
            raw_data.append(pd.Series(x))
        turnover_cum_year = pd.concat(raw_data,axis=1)
        turnover_cum_year.columns=list(range(df['season_year'].min(),df['season_year'].max()+1))
        turnover_cum_year=turnover_cum_year.rename(index={0:'turnover',1:'home_advantage'})
        turnover_cum_year.loc['total']=turnover_cum_year.loc['home_advantage']-turnover_cum_year.loc['turnover']
        return turnover_cum_year

    def turnover_table_year(df):
        raw_data=[]
        for n in range(df['season_year'].min(),df['season_year'].max()+1):
            x = np.polyfit(df[(df['season_year']==n)]['turnover'], df[(df['season_year'] ==n)]['home_score_margin_of_victory'], 1)
            raw_data.append(pd.Series(x))
        turnover_cum_year = pd.concat(raw_data,axis=1)
        turnover_cum_year.columns=list(range(df['season_year'].min(),df['season_year'].max()+1))
        turnover_cum_year=turnover_cum_year.rename(index={0:'turnover',1:'home_advantage'})
        turnover_cum_year.loc['total']=turnover_cum_year.loc['home_advantage']-turnover_cum_year.loc['turnover']
        return turnover_cum_year

    turnover_cum_year=turnover_table(df)
    st.write('Cumulative Turnover regression by year: and 2022 is current year',turnover_cum_year.style.format("{:.1f}", na_rep='-') )
    turnover_every_year=turnover_table_year(df)
    st.write('Annual Turnover regression by year',turnover_every_year.style.format("{:.1f}", na_rep='-') )
    # st.write('Dummy DF', dummy_df)
    turnover_every_year_dummy=turnover_table_year(dummy_df)
    st.write('Annual Dummy Turnover regression by year',turnover_every_year_dummy.style.format("{:.1f}", na_rep='-') )
    
    
def offensive_calc(df):
    df_offensive_home=df.loc[:,['Date','Week','Home Team', 'Home Score', 'season_year','unique_id','avg_home_score','avg_away_score',
    'Home Line Close','turnover','Away Team']]\
        .rename(columns={'Home Team':'team','Home Score':'score','Away Team':'opponent'})
    df_offensive_home['home_away']=1
    df_offensive_away=df.loc[:,['Date','Week','Away Team','Away Score', 'season_year','unique_id','avg_home_score','avg_away_score',
    'Home Line Close','turnover','Home Team']]\
        .rename(columns={'Away Team':'team','Away Score':'score','Home Team':'opponent'})
    df_offensive_away['turnover']=-df_offensive_away['turnover'] # i think this works converts it to same for everyone
    df_offensive_away['home_away']=-1
    df_offensive=pd.concat([df_offensive_home,df_offensive_away],axis=0).sort_values(by=['team','Date'],ascending=True).reset_index().drop('index',axis=1)
    df_offensive=df_offensive.sort_values(by=['unique_id','Date','Week'])
    df_offensive['season_games_played']=df_offensive.groupby(['team','season_year'])['score'].cumcount() # by not adding +1 it basically means i am shifting
    #  the line, so if we are in week 5, it will only count the games up to and including week 4
    # df_offensive['season_games_played']=df_offensive.groupby(['team','season_year'])['score'].cumcount()+1 # CAREFUL WATCH THIS
    df_offensive=df_offensive.sort_values(by=['Date','Week','unique_id'])
    return df_offensive

def offensive_calc_1(df_offensive):
    df_offensive['avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean()\
        .reset_index().drop(['level_2','team','season_year'],axis=1)
    # df_offensive['test_avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].shift()
    return df_offensive

def offensive_calc_test(df_offensive):
    df_offensive['avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    # df_offensive['test_avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].shift()
    return df_offensive


df_offensive=offensive_calc(df)
df_offensive=offensive_calc_test(df_offensive)

df_offensive_dummy=offensive_calc(dummy_df)
# st.write('df offensive dummy and is it sorted correctly', df_offensive_dummy)
# st.write('take a look at groupby here 2021',df_offensive_dummy.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean())
df_offensive_dummy=offensive_calc_test(df_offensive_dummy)
# st.write('after new function', df_offensive_dummy)

def col_correction(df_offensive,col='avg_pts_scored_team_season'):
    # df_offensive=df_offensive.sort_values(by=['Date','Week','unique_id'])
    df_offensive['SHIFT avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean().shift()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    df_offensive['test_col']=np.where(df_offensive[col].isna(),np.NaN,np.where(df_offensive['SHIFT avg_pts_scored_team_season'].isna(),np.NaN,1))
    df_offensive[col]=df_offensive['SHIFT avg_pts_scored_team_season']*df_offensive['test_col']
    df_offensive=df_offensive.drop(['SHIFT avg_pts_scored_team_season'],axis=1)
    return df_offensive

def col_correction_test(df_offensive,col='avg_pts_scored_team_season'):
    df_offensive['SHIFT avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean().shift()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    df_offensive['test_col']=np.where(df_offensive[col].isna(),np.NaN,np.where(df_offensive['SHIFT avg_pts_scored_team_season'].isna(),np.NaN,1))
    df_offensive[col]=df_offensive['SHIFT avg_pts_scored_team_season']*df_offensive['test_col']
    df_offensive=df_offensive.drop(['SHIFT avg_pts_scored_team_season'],axis=1)
    return df_offensive

def col_correction_turnover(df_offensive,col='turnover'):
    df_offensive=df_offensive.sort_values(by=['Date','Week','unique_id'])
    df_offensive['SHIFT turnover']=df_offensive.groupby(['team','season_year'])['turnover'].expanding(min_periods=4).sum().shift()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    # df_offensive['test_col_turn']=np.where(df_offensive[col].isna(),np.NaN,np.where(df_offensive['SHIFT turnover'].isna(),np.NaN,1))
    df_offensive['cum_turnover']=df_offensive['SHIFT turnover']*df_offensive['test_col']
    df_offensive=df_offensive.drop(['test_col','SHIFT turnover'],axis=1)
    return df_offensive

def col_correction_turnover_dummy(df_offensive,col='turnover'):
    df_offensive=df_offensive.sort_values(by=['Date','Week','unique_id'])
    df_offensive['SHIFT turnover']=df_offensive.groupby(['team','season_year'])['turnover'].expanding(min_periods=4).sum().shift()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    # df_offensive['test_col_turn']=np.where(df_offensive[col].isna(),np.NaN,np.where(df_offensive['SHIFT turnover'].isna(),np.NaN,1))
    df_offensive['cum_turnover']=df_offensive['SHIFT turnover']*df_offensive['test_col']
    # df_offensive=df_offensive.drop(['test_col','SHIFT turnover'],axis=1)
    return df_offensive

def col_correction_dummy(df_offensive,col='avg_pts_scored_team_season'):
    df_offensive=df_offensive.reset_index(drop=True)
    df_offensive=df_offensive.sort_values(by=['Date','Week','unique_id'])
    df_offensive['SHIFT avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean().shift()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    df_offensive['test_col']=np.where(df_offensive[col].isna(),np.NaN,np.where(df_offensive['SHIFT avg_pts_scored_team_season'].isna(),np.NaN,1))
    df_offensive[col]=df_offensive['SHIFT avg_pts_scored_team_season']*df_offensive['test_col']
    df_offensive=df_offensive.drop(['SHIFT avg_pts_scored_team_season'],axis=1)
    return df_offensive.sort_values(by=['Date','Week','unique_id'])

df_offensive=col_correction_dummy(df_offensive,col='avg_pts_scored_team_season')
df_offensive=col_correction_turnover_dummy(df_offensive,col='turnover')
df_offensive=df_offensive.rename(columns={'score':'pts_scored','mean_score':'4_game_pts_scored'}).sort_values(by=['Date','unique_id','team'])\
    .reset_index(drop=True)

df_offensive_dummy=col_correction_dummy(df_offensive_dummy,col='avg_pts_scored_team_season')
# st.write('what is going on here', df_offensive_dummy)
df_offensive_dummy=col_correction_turnover_dummy(df_offensive_dummy,col='turnover')
df_offensive_dummy=df_offensive_dummy.rename(columns={'score':'pts_scored','mean_score':'4_game_pts_scored'}).sort_values(by=['Date','unique_id','team'])\
    .reset_index(drop=True)

# st.write('is turnover ok>>',df_offensive_dummy[df_offensive_dummy['team']=='Ireland'].sort_values('unique_id'))


def defensive_calc(df):
    df_defensive_home=df.loc[:,['Date','Week','Home Team', 'Away Score', 'season_year','unique_id','avg_home_score','avg_away_score',
    'Home Line Close','turnover']]\
        .rename(columns={'Home Team':'team','Away Score':'score'})
    df_defensive_home['home_away']=1
    df_defensive_away=df.loc[:,['Date','Week','Away Team','Home Score', 'season_year','unique_id','avg_home_score','avg_away_score','Home Line Close','turnover']]\
        .rename(columns={'Away Team':'team','Home Score':'score'})
    df_defensive_away['home_away']=-1
    df_defensive_away['turnover']=-df_defensive_away['turnover']
    df_defensive=pd.concat([df_defensive_home,df_defensive_away],axis=0).sort_values(by=['team','Date'],ascending=True).reset_index().drop('index',axis=1)
    df_defensive['avg_pts_conceded_team_season']=df_defensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    return df_defensive

def defensive_calc_1(df):
    df_defensive_home=df.loc[:,['Date','Week','Home Team', 'Away Score', 'season_year','unique_id','avg_home_score','avg_away_score',
    'Home Line Close','turnover']]\
        .rename(columns={'Home Team':'team','Away Score':'score'})
    df_defensive_home['home_away']=1
    df_defensive_away=df.loc[:,['Date','Week','Away Team','Home Score', 'season_year','unique_id','avg_home_score','avg_away_score','Home Line Close','turnover']]\
        .rename(columns={'Away Team':'team','Home Score':'score'})
    df_defensive_away['home_away']=-1
    df_defensive_away['turnover']=-df_defensive_away['turnover']
    df_defensive=pd.concat([df_defensive_home,df_defensive_away],axis=0).sort_values(by=['unique_id','Date'],ascending=True).reset_index().drop('index',axis=1)
    df_defensive=df_defensive.sort_values(by=['Date','Week','unique_id'])
    return df_defensive

def defensive_calc_2(df_defensive):
    df_defensive['avg_pts_conceded_team_season']=df_defensive.groupby(['team','season_year'])['score'].expanding(min_periods=4).mean()\
        .reset_index().rename(columns={'level_2':'index'}).drop(['team','season_year'],axis=1).set_index('index')
    return df_defensive



# df_defensive=defensive_calc(df)
# df_defensive=col_correction(df_defensive,col='avg_pts_conceded_team_season')
# df_defensive=df_defensive.rename(columns={'score':'pts_conceded','mean_score':'4_game_pts_conceded'}).sort_values(by=['team','Date'])

df_defensive=defensive_calc_1(df)
df_defensive=defensive_calc_2(df_defensive)
df_defensive=col_correction(df_defensive,col='avg_pts_conceded_team_season')
df_defensive=df_defensive.rename(columns={'score':'pts_conceded','mean_score':'4_game_pts_conceded'}).sort_values(by=['Date','unique_id','team'])\
    .reset_index(drop=True)

df_defensive_dummy=defensive_calc_1(dummy_df)
df_defensive_dummy=defensive_calc_2(df_defensive_dummy)
df_defensive_dummy=col_correction(df_defensive_dummy,col='avg_pts_conceded_team_season')
df_defensive_dummy=df_defensive_dummy.rename(columns={'score':'pts_conceded','mean_score':'4_game_pts_conceded'}).sort_values(by=['Date','unique_id','team'])\
    .reset_index(drop=True)


# df_new=pd.merge(df_offensive,df_defensive,how='outer')
df_new=pd.merge(df_offensive,df_defensive,how='outer',on=['team','unique_id','season_year','avg_home_score','avg_away_score',
'Date','Week','Home Line Close','turnover','home_away']\
    ,validate='one_to_one',indicator=True).drop('test_col_y',axis=1).rename(columns={'test_col_x':'test_col'})

df_new_dummy=pd.merge(df_offensive_dummy,df_defensive_dummy,how='outer',on=['team','unique_id','season_year','avg_home_score','avg_away_score',
'Date','Week','Home Line Close','turnover','home_away']\
    ,validate='one_to_one',indicator=True).drop('test_col_y',axis=1).rename(columns={'test_col_x':'test_col'})

def calcs_2(df_new):
    df_new['team_cum_sum_pts']=df_new.groupby(['team'])['pts_scored'].cumsum()
    df_new['team_cum_sum_games']=df_new.groupby(['team'])['pts_scored'].cumcount()+1
    df_new['rolling_avg_team_pts_scored']=df_new['team_cum_sum_pts'] / df_new['team_cum_sum_games']
    return df_new

def calcs_3(df_new):
    df_new=df_new.sort_values(by=['Date','unique_id','team'])
    df_new['date_avg_pts_rolling']=df_new['pts_scored'].expanding().mean().shift(32) # 16 teams by 2, want the previous week numbers
    df_new['date_avg_pts_rolling_no_shift']=df_new['pts_scored'].expanding().mean()
    return df_new

def calcs_3_dummy(df_new):
    df_new=df_new.sort_values(by=['Date','unique_id','team'])
    df_new['date_avg_pts_rolling']=df_new['pts_scored'].expanding().mean().shift(6) # smaller dummy size for six nations
    df_new['date_avg_pts_rolling_no_shift']=df_new['pts_scored'].expanding().mean()
    return df_new

def calcs_4(df_new):
    df_new=df_new.sort_values(by=['team','Date'],ascending=True)
    df_new=df_new.sort_values(by=['home_away','Date','unique_id','team'],ascending=True)
    df_home=df_new[df_new['home_away']==1].sort_values(by=['Date','unique_id'],ascending=True)
    df_home['home_pts_avg']=df_home['pts_scored'].expanding().mean().shift(16) # want to get previous week points
    df_away=df_new[df_new['home_away']==-1].sort_values(by=['Date','unique_id'],ascending=True)
    df_away['away_pts_avg']=df_away['pts_scored'].expanding().mean().shift(16)
    df_new=pd.concat([df_home,df_away],ignore_index=True)

    df_new=df_new.sort_values(by=['unique_id','home_away'],ascending=[True,False])
    df_new['avg_away_score']=df_new['avg_away_score'].fillna(method='ffill')
    df_new['avg_home_score']=df_new['avg_home_score'].fillna(method='ffill')
    df_new['home_adv']=df_new['avg_home_score']-df_new['avg_away_score']
    cols_to_move=['Date','team','season_year','Home Line Close','unique_id','pts_scored','pts_conceded','home_adv','date_avg_pts_rolling_no_shift',
    'date_avg_pts_rolling','avg_pts_scored_team_season',
    'avg_pts_conceded_team_season','home_pts_avg','away_pts_avg','avg_home_score','avg_away_score','home_away']
    cols = cols_to_move + [col for col in df_new if col not in cols_to_move]
    df_new=df_new[cols]
    df_new=df_new.sort_values(by=['Date','unique_id','team'])
    return df_new

df_new=calcs_2(df_new)
df_new_dummy=calcs_2(df_new_dummy)
df_new=calcs_3(df_new)
df_new_dummy=calcs_3_dummy(df_new_dummy)
df_new=calcs_4(df_new)
df_new_dummy=calcs_4(df_new_dummy)

strength_schedule_df_2=df_new.copy()
# st.write('strenght schedule original', strength_schedule_df_2)
# st.download_button(label="Download",data=df_new_dummy.to_csv().encode('utf-8'),file_name='df_dummy.csv',mime='text/csv',key='dummy_shift')
# st.write('date avg pts rolling in here???', df_new_dummy.dtypes)


def home_away_combine(df_new):
    df_new=df_new.loc[:,['Date','Week','team','season_year','unique_id','Home Line Close','pts_scored','pts_conceded','home_adv','date_avg_pts_rolling',
    'avg_pts_scored_team_season','avg_pts_conceded_team_season','home_away','turnover','cum_turnover','season_games_played','opponent']]

    df_home_1=df_new[df_new['home_away']==1].rename(columns={'pts_scored':'home_pts_scored','pts_conceded':'home_pts_conceded','team':'home_team',
    'avg_pts_scored_team_season':'home_avg_pts_scored_team_season','avg_pts_conceded_team_season':'home_avg_pts_conceded_team_season',
    'date_avg_pts_rolling':'home_date_avg_pts_rolling'})\
        .set_index(['unique_id']).drop('home_away',axis=1).copy()
    df_away_1=df_new[df_new['home_away']==-1].rename(columns={'pts_scored':'away_pts_scored','pts_conceded':'away_pts_conceded','team':'away_team',
    'avg_pts_scored_team_season':'away_avg_pts_scored_team_season','avg_pts_conceded_team_season':'away_avg_pts_conceded_team_season',
    'date_avg_pts_rolling':'away_date_avg_pts_rolling'})\
        .set_index(['unique_id']).drop(['home_adv','home_away'],axis=1).copy()
    # st.write('df home', df_home_1, 'away', df_away_1)
    # df_combined=pd.concat([df_home_1,df_away_1],axis=0)
    # st.write('before home and away are combined', df_home_1, 'away', df_away_1)
    df_combined=pd.merge(df_home_1.reset_index(),df_away_1.reset_index(),on=['unique_id','Date','season_year', 'Home Line Close'],how='outer')
    return df_combined

def home_away_combine_dummy(df_new): # no 'home adv' or 'date avg pts rolling'
    df_new=df_new.loc[:,['Date','Week','team','season_year','unique_id','Home Line Close','pts_scored','pts_conceded','avg_pts_scored_team_season',
    'date_avg_pts_rolling','home_adv',
    'avg_pts_conceded_team_season','home_away','turnover','cum_turnover','season_games_played','opponent']]
    df_home_1=df_new[df_new['home_away']==1].rename(columns={'pts_scored':'home_pts_scored','pts_conceded':'home_pts_conceded','team':'home_team',
    'avg_pts_scored_team_season':'home_avg_pts_scored_team_season','avg_pts_conceded_team_season':'home_avg_pts_conceded_team_season',
    'date_avg_pts_rolling':'home_date_avg_pts_rolling'})\
        .set_index(['unique_id']).copy()
    df_away_1=df_new[df_new['home_away']==-1].rename(columns={'pts_scored':'away_pts_scored','pts_conceded':'away_pts_conceded','team':'away_team',
    'avg_pts_scored_team_season':'away_avg_pts_scored_team_season','avg_pts_conceded_team_season':'away_avg_pts_conceded_team_season',
    'date_avg_pts_rolling':'away_date_avg_pts_rolling'})\
        .set_index(['unique_id']).drop(['home_adv','home_away'],axis=1).copy()
    # st.write('df home', df_home_1, 'away', df_away_1)
    # df_combined=pd.concat([df_home_1,df_away_1],axis=0)
    # st.write('before home and away are combined', df_home_1, 'away', df_away_1)
    df_combined=pd.merge(df_home_1.reset_index(),df_away_1.reset_index(),on=['unique_id','Date','season_year', 'Home Line Close'],how='outer')
    return df_combined

# df_combined=home_away_combine(df_new)
# st.write('where is home_adv', df_new)
df_combined=home_away_combine_dummy(df_new)
df_dummy_turnover=home_away_combine_dummy(df_new_dummy)
# st.write('IS home_adv in here?', 'home_adv' in df_combined.columns)
# st.write('IS home_adv in here?', 'home_adv' in df_dummy_turnover.columns)
strength_schedule_df_1=df_combined.copy()

with st.expander('first simple model'):
    df_turnover_rating=df_combined.copy()
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
    # st.write('data before results are tallied up')
    # AgGrid(df_combined,enable_enterprise_modules=True)

    # st.write('df_comb', df_combined[df_combined['season_year']==2022].set_index('Date'))
    # st.download_button(label="Download data as CSV",data=df_combined[df_combined['season_year']==2021].to_csv().encode('utf-8'),file_name='df_spread.csv',mime='text/csv',key='after_merge_spread')

    st.write('The home-away date avg pts rolling is the average points scored in every match so we can see what the avg pts scored and conceded is both will be same')
    # st.write('Database', df_combined)
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

with st.expander('Average Error Calcs on Simple Model'):
    avg_error_data=df_combined.copy()
    avg_error_data=avg_error_data[avg_error_data['proj_spread'].notna()]
    # st.write('avg error df', avg_error_data)
    # avg_error_data['test']=avg_error_data['home_team']-avg_error_data['away_team']
    avg_error_data.insert(loc=8,column='betting_odds_error',value=(abs(avg_error_data['home_pts_scored']+avg_error_data['Home Line Close']-avg_error_data['away_pts_scored'])))
    avg_error_data.insert(loc=9,column='proj_odds_error',value=(abs(avg_error_data['home_pts_scored']+avg_error_data['proj_spread']-avg_error_data['away_pts_scored'])))
    avg_error_data.insert(loc=11,column='diff_proj_spread_actual',value=(abs(avg_error_data['proj_spread']-avg_error_data['Home Line Close'])))
    # avg_error_data['avg_err_betting']=avg_error_data['betting_odds_error'].expanding().mean()
    avg_error_data_groupby=avg_error_data.groupby(['season_year'])['betting_odds_error','proj_odds_error'].mean().transpose()
    avg_error_data_groupby.loc['Diff']=avg_error_data_groupby.loc['proj_odds_error']-avg_error_data_groupby.loc['betting_odds_error']
    # st.write('avg error df data', avg_error_data)
    st.write('avg error df groupby comparing the average error on bookies odds versus our internal projections', 
    avg_error_data_groupby.style.format("{:.1f}", na_rep='-'))
    st.write('betting odds error lifetime to date',avg_error_data['betting_odds_error'].mean())
    st.write('proj odds error lifetime to date',avg_error_data['proj_odds_error'].mean())

    avg_error_data['decile_spread_diff'] = pd.qcut(avg_error_data['diff_proj_spread_actual'], 10, labels=range(10))
    avg_error_data['decile_spread'] = pd.qcut(avg_error_data['Home Line Close'], 10, labels=range(10))
    st.write('Index below is split into 10 deciles of the difference between the Closing Spread and our Estimated Spread')
    st.write('So want to see the result for where the diff to the spread is greater than 7 per Elihu')
    st.write(   avg_error_data.groupby(['decile_spread_diff']).agg(winning=('result','sum'),count=('result','count'),
    avg_proj_odds_error=('proj_odds_error','mean'),med_proj_odds_error=('proj_odds_error','median'),min_diff_to_spread=('diff_proj_spread_actual','min'),
    max_diff_to_spread=('diff_proj_spread_actual','max') )   )


with st.expander('Splitting the Spread up to analyse'):
    avg_error_data_spread=avg_error_data.copy()
    st.write('d',avg_error_data_spread)
    decile_df=avg_error_data_spread.groupby(pd.qcut(avg_error_data_spread['Home Line Close'], 10))['result'].sum().reset_index()
    st.write(decile_df)
    # avg_error_data=avg_error_data[avg_error_data['proj_spread'].notna()]

with st.expander('Turnover Model'):
    # df_dummy_turnover
    def clean_turnover_df(df_turnover_rating):
        df_turnover_rating=df_turnover_rating.copy().rename(columns={'turnover_x':'turnover_home','turnover_y':'turnover_away','cum_turnover_x':'turnover_cum_home',
        'cum_turnover_y':'turnover_cum_away','season_games_played_x':'season_games_played_home','season_games_played_y':'season_games_played_away'})
        # st.write('think i need a home and away turnover for each team')
        # AgGrid( df_turnover_rating,enable_enterprise_modules=True)
        cols_to_move=['Date','home_team','away_team','turnover_home','turnover_away','turnover_cum_home','turnover_cum_away']
        cols = cols_to_move + [col for col in df_turnover_rating if col not in cols_to_move]
        df_turnover_rating=df_turnover_rating[cols]
        return df_turnover_rating
    df_turnover_rating=clean_turnover_df(df_turnover_rating)
    df_turnover_rating_dummy=clean_turnover_df(df_dummy_turnover)
    # st.write('Take it from here', df_turnover_rating_dummy)
    # AgGrid( df_turnover_rating,enable_enterprise_modules=True)
    st.write('turnover_home: means the turnover gained by the home team in the game and it will be a negative number if the home team won the turnover battle')
    st.write('turnover_cum_home: means the cumulative turnovers gained, same as above, negative number if the home team won the turnover battle')
    st.write('The turnover numbers are shifted, I checked, so in week 5, you are looking at turnovers up to and including week 4')
    st.write('what about season games played???')
    st.write('Dataframe', 
    df_turnover_rating[(df_turnover_rating['home_team']=='Philadelphia Eagles') | (df_turnover_rating['away_team']=='Philadelphia Eagles') ].head(2)  )
    
    def turnover_calcs(df_turnover_rating):
        df_turnover_rating['home_turnover_per_game']=df_turnover_rating['turnover_cum_home'] / df_turnover_rating['season_games_played_home']
        df_turnover_rating['away_turnover_per_game']=df_turnover_rating['turnover_cum_away'] / df_turnover_rating['season_games_played_away']
        # strength_schedule_df=df_turnover_rating.copy()
        df_turnover_rating['home_offensive_rating'] = df_turnover_rating['home_avg_pts_scored_team_season'] + (df_turnover_rating['home_turnover_per_game']*((4*0.8) /2))
        df_turnover_rating['home_defensive_rating']=df_turnover_rating['home_avg_pts_conceded_team_season'] - (df_turnover_rating['home_turnover_per_game']*((4*0.8) /2))

        df_turnover_rating['away_offensive_rating'] = df_turnover_rating['away_avg_pts_scored_team_season'] + (df_turnover_rating['away_turnover_per_game']*((4*0.8) /2))
        df_turnover_rating['away_defensive_rating']=df_turnover_rating['away_avg_pts_conceded_team_season'] - (df_turnover_rating['away_turnover_per_game']*((4*0.8) /2))
        # df_turnover_rating['away_defensive_rating']=df_turnover_rating['away_avg_pts_conceded_team_season'] / df_turnover_rating['away_date_avg_pts_rolling']
        
        df_turnover_rating['projected_team_a_pts']= df_turnover_rating['home_offensive_rating'] * \
            (df_turnover_rating['away_defensive_rating'] / df_turnover_rating['away_date_avg_pts_rolling'])
        
        df_turnover_rating['projected_team_b_pts']= df_turnover_rating['away_offensive_rating'] * \
            (df_turnover_rating['home_defensive_rating'] / df_turnover_rating['away_date_avg_pts_rolling'])

        df_turnover_rating['proj_spread'] = df_turnover_rating['projected_team_b_pts'] - df_turnover_rating['projected_team_a_pts']
        df_turnover_rating['bet_sign']=np.where(df_turnover_rating['proj_spread']>df_turnover_rating['Home Line Close'],-1,np.where(df_turnover_rating['proj_spread']<df_turnover_rating['Home Line Close'],1,0))
        df_turnover_rating['home_win']=df_turnover_rating['home_pts_scored'] - df_turnover_rating['away_pts_scored']
        df_turnover_rating['home_win'] = np.where((df_turnover_rating['home_pts_scored'] > df_turnover_rating['away_pts_scored']), 1, np.where((df_turnover_rating['home_pts_scored'] < df_turnover_rating['away_pts_scored']),-1,0))
        df_turnover_rating['home_cover']=(np.where(((df_turnover_rating['home_pts_scored'] + df_turnover_rating['Home Line Close']) > df_turnover_rating['away_pts_scored']), 1,
        np.where(((df_turnover_rating['home_pts_scored']+ df_turnover_rating['Home Line Close']) < df_turnover_rating['away_pts_scored']),-1,0)))
        df_turnover_rating['home_cover_result']=df_turnover_rating['home_cover'].astype(int)
        df_turnover_rating['away_cover'] = -df_turnover_rating['home_cover']
        df_turnover_rating['result']=df_turnover_rating['home_cover_result'] * df_turnover_rating['bet_sign']

        cols_to_move=['Date','season_year','unique_id','home_team','away_team','home_pts_scored','away_pts_scored','Home Line Close','proj_spread',
        'bet_sign','home_cover_result','result']
        cols = cols_to_move + [col for col in df_turnover_rating if col not in cols_to_move]
        df_turnover_rating=df_turnover_rating[cols]
        return df_turnover_rating


    df_turnover_rating=turnover_calcs(df_turnover_rating)
    df_turnover_rating_dummy=turnover_calcs(df_turnover_rating_dummy)
    # st.download_button(label="Download",data=df_turnover_rating_dummy.to_csv().encode('utf-8'),file_name='df_dummy.csv',mime='text/csv',key='dummy')
    st.write('check that theres no rating for 2021 proj spread strange what is causing it', df_turnover_rating_dummy)

    def pivot_turnover(df_turnover_rating):
        result_count=df_turnover_rating.groupby(['season_year'])['result'].value_counts()
        clean_df_for_pivot=df_turnover_rating.loc[:,['season_year','result','bet_sign']]
        result_pivot=pd.pivot_table(clean_df_for_pivot, index='result', columns='season_year', aggfunc='count')
        result_pivot.index=result_pivot.index.astype('str')
        result_pivot.loc['total_games']=result_pivot.sum(axis=0)
        return result_pivot

    def pivot_turnover_1(result_pivot):
        result_pivot.loc['winning_%']=result_pivot.loc['1']/(result_pivot.loc['1']+result_pivot.loc['-1'])
        # st.write('issue', result_pivot)
        # st.write('issue 2', result_pivot.loc['1'].cumsum())
        # st.write('issue 3', result_pivot.loc['-1'].cumsum())
        # st.write('issue 4', (result_pivot.loc['1'].cumsum()+result_pivot.loc['-1'].cumsum()))
        # st.write('issue 5', (result_pivot.loc['1'].cumsum() / (result_pivot.loc['1'].cumsum()+result_pivot.loc['-1'].cumsum())))
        result_pivot.loc['cum_winning_%']=result_pivot.loc['1'].cumsum() / (result_pivot.loc['1'].cumsum()+result_pivot.loc['-1'].cumsum())
        return result_pivot

    # def pivot_turnover_2(result_pivot):
    #     result_pivot.loc['winning_%']=result_pivot.loc['1']/(result_pivot.loc['1']+result_pivot.loc['-1'])
    #     # st.write('result pivot', result_pivot)
    #     # result_pivot.loc['cum_winning_%']=result_pivot.loc['1'].cumsum() / (result_pivot.loc['1'].cumsum()+result_pivot.loc['-1'].cumsum())
    #     return result_pivot

    # st.write(result_count.columns)
    # result_count=result_count.reset_index()
    # st.dataframe(result_count)
    result_pivot=pivot_turnover(df_turnover_rating)
    # st.write('should replciate this', result_pivot)
    result_pivot=pivot_turnover_1(result_pivot)
    st.dataframe(result_pivot)

    result_pivot_dummy=pivot_turnover(df_turnover_rating_dummy)
    # st.write('what is up with dummy' ,result_pivot_dummy)
    result_pivot_dummy=pivot_turnover_1(result_pivot_dummy)
    st.dataframe(result_pivot_dummy)

with st.expander('Average Error Calcs on Turnover Model'):
    avg_error_data=df_turnover_rating.copy()
    avg_error_data=avg_error_data[avg_error_data['proj_spread'].notna()]
    # st.write('avg error df', avg_error_data)
    # avg_error_data['test']=avg_error_data['home_team']-avg_error_data['away_team']
    avg_error_data.insert(loc=8,column='betting_odds_error',value=(abs(avg_error_data['home_pts_scored']+avg_error_data['Home Line Close']-avg_error_data['away_pts_scored'])))
    avg_error_data.insert(loc=9,column='proj_odds_error',value=(abs(avg_error_data['home_pts_scored']+avg_error_data['proj_spread']-avg_error_data['away_pts_scored'])))
    avg_error_data.insert(loc=11,column='diff_proj_spread_actual',value=(abs(avg_error_data['proj_spread']-avg_error_data['Home Line Close'])))
    # avg_error_data['avg_err_betting']=avg_error_data['betting_odds_error'].expanding().mean()
    avg_error_data_groupby=avg_error_data.groupby(['season_year'])['betting_odds_error','proj_odds_error'].mean().transpose()
    # st.write('avg error df data', avg_error_data)
    avg_error_data_groupby.loc['Diff']=avg_error_data_groupby.loc['proj_odds_error']-avg_error_data_groupby.loc['betting_odds_error']
    st.write('avg error df groupby comparing the average error on bookies odds versus our internal projections',
    avg_error_data_groupby.style.format("{:.1f}", na_rep='-'))
    
    st.write('betting odds error lifetime to date',avg_error_data['betting_odds_error'].mean())
    st.write('proj odds error lifetime to date',avg_error_data['proj_odds_error'].mean())

    st.write('How does the model do when the proj spread is at least 7 points different from the market?')
    
    # df = pd.DataFrame({'A':'foo foo foo bar bar bar'.split(),
    #                'B':[0.1, 0.5, 1.0]*2})
    # # st.write(df)
    # df['C'] = df.groupby(['A'])['B'].transform(
    #                  lambda x: pd.qcut(x, 3, labels=False))
    # # st.write(df)

    # df = pd.DataFrame(dict(revenue=np.random.randint(1000000, 99999999, 100)))
    # df['decile'] = pd.qcut(df.revenue, 10, labels=range(10))
    # df = df.join(df.groupby('decile').revenue.agg(Mean='mean', Std='std'), on='decile')
    # df['revenue_zscore_by_decile'] = df.revenue.sub(df.Mean).div(df.Std)
    # st.write('try this',df.head())
    # https://stackoverflow.com/questions/41354611/select-filter-bins-after-qcut-decile
    avg_error_data['decile_spread_diff'] = pd.qcut(avg_error_data['diff_proj_spread_actual'], 10, labels=range(10))
    avg_error_data['decile_spread'] = pd.qcut(avg_error_data['Home Line Close'], 10, labels=range(10))
    
    # st.write('data',avg_error_data)
    cols_to_move=['Date','home_team','away_team','home_pts_scored','away_pts_scored','Home Line Close','proj_spread','diff_proj_spread_actual','decile_spread_diff']
    cols = cols_to_move + [col for col in avg_error_data if col not in cols_to_move]
    avg_error_data=avg_error_data[cols]
    # st.write('data',avg_error_data)
    
    # avg_error_data = avg_error_data.join(avg_error_data.groupby('decile_spread_diff')['diff_proj_spread_actual'].agg(Mean='mean', Std='std',Max='max',Min='min',
    # Count='count',Median='median'), on='decile_spread_diff')
    
    # st.write('Groupby Stats on Spread',avg_error_data.groupby('decile_spread')['Home Line Close'].agg(Mean='mean', Std='std',Max='max',Min='min',
    # Count='count',Median='median'), on='decile_spread')
    
    # st.write('not sure on what the below stats is telling me')
    # st.write('Groupby Stats on Proj Spread Diff v. Actual Spread',avg_error_data.groupby('decile_spread_diff')['diff_proj_spread_actual'].agg(Mean='mean', Std='std',Max='max',
    # Min='min',Count='count',Median='median'), on='decile_spread_diff')

    # st.write('Groupby Stats on Proj Spread Diff v. Actual Spread V1',avg_error_data.groupby('decile_spread_diff').agg(Mean='mean',
    # Count='count',Median='median'),on=['proj_odds_error'])
    # st.write(  avg_error_data.groupby(['decile_spread_diff'])[['proj_odds_error','result']].sum()    )
    # st.write(  avg_error_data.groupby(['decile_spread_diff'])[['proj_odds_error','result']].agg('sum','count')    )
    st.write('Index below is split into 10 deciles of the difference between the Closing Spread and our Estimated Spread')
    st.write('So want to see the result for where the diff to the spread is greater than 7 per Elihu')
    # st.write(pd.qcut(pd.Series(avg_error_data['diff_proj_spread_actual']), 10,labels=range(10))             )
    # st.write(pd.qcut(pd.Series(avg_error_data['diff_proj_spread_actual']), 10).drop_duplicates().sort_values().reset_index().drop('index',axis=1) )
    st.write(   avg_error_data.groupby(['decile_spread_diff']).agg(winning=('result','sum'),count=('result','count'),
    avg_proj_odds_error=('proj_odds_error','mean'),med_proj_odds_error=('proj_odds_error','median'),min_diff_to_spread=('diff_proj_spread_actual','min'),
    max_diff_to_spread=('diff_proj_spread_actual','max') )   )

    # st.write(avg_error_data.groupby)
    # st.write('x',avg_error_data.groupby(['decile_spread_diff','diff_proj_spread_actual']).agg({'proj_odds_error':['mean','count']}).reset_index())
    
    # avg_error_data = avg_error_data.join(avg_error_data.groupby('decile_spread_diff')['proj_odds_error'].agg(Mean='mean', Std='std',Max='max',Min='min'), on='decile_spread_diff')
    
with st.expander("Strength of Schedule Workings"):
    # test_2022=strength_schedule_df_2[strength_schedule_df_2['season_year']==2022]
    test_2022=strength_schedule_df_2.copy()
    # dummy_2022=df_new_dummy[df_new_dummy['season_year']==2022]
    dummy_2022=df_new_dummy.copy()
    # st.write('dummy 2022 there are some NAs in here need to fix look at duplicates', dummy_2022)
    # def col_correction(df_offensive,team,col='avg_pts_scored_team_season'):
    #     df_offensive=df_offensive[df_offensive['team']!=team]
    #     df_offensive['SHIFT avg_pts_scored_team_season']=df_offensive.groupby(['team','season_year'])['pts_scored'].expanding(min_periods=4).mean().shift()\
    #         .reset_index().drop(['level_2','team','season_year'],axis=1)
    #     df_offensive['test_col']=np.where(df_offensive[col].isna(),np.NaN,np.where(df_offensive['SHIFT avg_pts_scored_team_season'].isna(),np.NaN,1))
    #     df_offensive[team]=df_offensive['SHIFT avg_pts_scored_team_season']*df_offensive['test_col']
    #     df_offensive=df_offensive.drop(['SHIFT avg_pts_scored_team_season'],axis=1)
    #     return df_offensive

    team_list = test_2022['team'].unique()
    test_2022=test_2022.reset_index(drop=True)

    team_list_dummy = dummy_2022['team'].unique()
    dummy_2022=dummy_2022.reset_index(drop=True)

    def home_pts_adj(dummy_2022):
        # want to adjust points scored for home advantage 
        dummy_2022['home_adv_adj']=np.where(dummy_2022['home_away']==1,-2.5,2.5)
        dummy_2022['pts_scored_adj']=dummy_2022['pts_scored']+dummy_2022['home_adv_adj']
        return dummy_2022

    test_2022=home_pts_adj(test_2022)
    dummy_2022=home_pts_adj(dummy_2022)
    # st.write('check the avg pts scored team season as test col runs off it', dummy_2022)

    # st.download_button(label="Download",data=dummy_2022.to_csv().encode('utf-8'),file_name='df_dummy.csv',mime='text/csv',key='dummy_shift_1')
    # raw_data_offence=[]
    # raw_data_defence=[]
    
    def offence_sos(test_2022,team_list,pts_scored='pts_scored'):
        raw_data_offence=[]
        for x in team_list:
            df_1=test_2022[(test_2022['team']!=x) & (test_2022['opponent']!=x)].sort_values(['Week','Date','unique_id'],ascending=[True,True,True])
            df_1[x]=df_1.groupby(['team','season_year'])[pts_scored].cumsum()
            df_1[x]=df_1.groupby(['team','season_year'])[x].shift(1)
            df_1['test_col']=np.where(df_1['avg_pts_scored_team_season'].isna(),np.NaN,np.where(df_1[x].isna(),np.NaN,1))
            df_1[x]=df_1[x]*df_1['test_col']
            extract=df_1.loc[:,['unique_id',x]]
            raw_data_offence.append(df_1.loc[:,x])
            # df_1=df_1.drop(x,axis=1)

        cleaned_container=pd.DataFrame(raw_data_offence).transpose()
        cleaned_container.columns=cleaned_container.columns + '_offence'
        df_1=pd.merge(test_2022,cleaned_container,left_index=True,right_index=True,how='outer')
        return df_1

    # st.write('checking function')
    # ireland_test=dummy_2022[(dummy_2022['team']!='Ireland') & (dummy_2022['opponent']!='Ireland') & (dummy_2022['season_year']==2022) ]\
        # .sort_values(['Week','Date','unique_id'])
    # st.write('is test col working for test pts')
    # ireland_test['test_pts'] = ireland_test.groupby(['team','season_year'])['pts_scored'].cumsum()
    # ireland_test['shift_test_pts'] = ireland_test.groupby(['team','season_year'])['test_pts'].shift(1)
    # st.write( ireland_test [ireland_test['team']=='England'].set_index(['team','opponent','test_pts','shift_test_pts']))
    # st.write( ireland_test.groupby(['team','season_year'])['pts_scored'].cumsum() )

    # st.write(dummy_2022[(dummy_2022['team']!='Ireland') & (dummy_2022['opponent']!='Ireland')].sort_values(['Week','Date','unique_id']\
    #     .groupby(['team','season_year'])['pts_scored'].cumsum()))

    def offence_sos_dummy(test_2022,team_list,pts_scored='pts_scored'):
        raw_data_offence=[]
        for x in team_list:
            df_1=test_2022[(test_2022['team']!=x) & (test_2022['opponent']!=x)].sort_values(['Week','Date','unique_id'],ascending=[True,True,True])
            df_1[x]=df_1.groupby(['team','season_year'])[pts_scored].cumsum()
            df_1[x]=df_1.groupby(['team','season_year'])[x].shift(1)
            df_1['test_col']=np.where(df_1['avg_pts_scored_team_season'].isna(),np.NaN,np.where(df_1[x].isna(),np.NaN,1))
            df_1[x]=df_1[x]*df_1['test_col']
            # extract=df_1.loc[:,['unique_id',x]]
            raw_data_offence.append(df_1.loc[:,x])
            # st.write('team', x, 'data', df_1.sort_values(by=['Date','unique_id'],ascending=[True,True]))
            # st.write('test 2022 first function team', x, 'data', df_1)
            # df_1=df_1.drop(x,axis=1)

        cleaned_container=pd.DataFrame(raw_data_offence).transpose()
        cleaned_container.columns=cleaned_container.columns + '_offence'
        # st.write('cleaned container', cleaned_container)
        # st.write('df 1 to be merged', df_1)
        df_1=pd.merge(test_2022,cleaned_container,left_index=True,right_index=True,how='outer')
        return df_1    

    # df_1=offence_sos(test_2022,team_list,pts_scored='pts_scored_adj')
    df_1=offence_sos_dummy(test_2022,team_list,pts_scored='pts_scored_adj')
    df_1_dummy=offence_sos_dummy(dummy_2022,team_list_dummy,pts_scored='pts_scored_adj')
    
    def defence_sos(df_1,team_list):
        raw_data_defence=[]
        for x in team_list:
            # st.write('x', x)
            df_2=df_1[(df_1['team']!=x) & (df_1['opponent']!=x)].sort_values(['Week','Date','unique_id'],ascending=[True,True,True])
            df_2[x]=df_2.groupby(['team','season_year'])['pts_conceded'].cumsum()
            # st.write('inside function checking for defence', df_2.set_index(['team','opponent']).sort_values(by='Date'))
            df_2[x]=df_2.groupby(['team','season_year'])[x].shift(1)
            # st.write('inside function checking for defence after shift', df_2.set_index(['team','opponent']).sort_values(by='Date'))
            df_2['test_col']=np.where(df_2['avg_pts_scored_team_season'].isna(),np.NaN,np.where(df_2[x].isna(),np.NaN,1))
            # st.write('checking the test col BEFORE calc', df_2.set_index(['team','opponent']).sort_values(by='Date'))
            df_2[x]=df_2[x]*df_2['test_col']
            # st.write('checking AFTER test col calc', df_2.set_index(['team','opponent']).sort_values(by='Date'))
            extract=df_2.loc[:,['unique_id',x]]
            raw_data_defence.append(df_2.loc[:,x])
            # df_2=df_2.drop(x,axis=1)

        cleaned_container_defence=pd.DataFrame(raw_data_defence).transpose()
        cleaned_container_defence.columns=cleaned_container_defence.columns + '_defence'
        # st.write('cleaned container defence before merge', cleaned_container_defence)
        # st.write('dataframe before merge', df_1)
        df_4=pd.merge(df_1,cleaned_container_defence,left_index=True,right_index=True,how='outer')
        return df_4

    df_4=defence_sos(df_1,team_list) # just while i check inside the function
    df_4_dummy=defence_sos(df_1_dummy,team_list_dummy)
    # st.write('dummy dataframe after merge', df_4_dummy)

    def raw_data_sos(df_4,team_list):
        raw_data_diff=[]
        for x in team_list:
            # st.write('team x', x)
            df_5=df_4[(df_4['team']!=x) & (df_4['opponent']!=x)].sort_values(['Week','Date','unique_id'],ascending=[True,True,True])
            df_5[x]=df_5.groupby(['team','season_year'])['pts_scored'].cumcount()+1 # careful with cumcount it starts with zero!!
            df_5[x]=df_5.groupby(['team','season_year'])[x].shift(1)
            df_5['test_col']=np.where(df_5['avg_pts_scored_team_season'].isna(),np.NaN,np.where(df_5[x].isna(),np.NaN,1))
            df_5[x]=df_5[x]*df_5['test_col']
            extract=df_5.loc[:,['unique_id',x]]
            raw_data_diff.append(df_5.loc[:,x])
            # df_5=df_5.drop(x,axis=1)

        cleaned_container_diff=pd.DataFrame(raw_data_diff).transpose()
        cleaned_container_diff.columns=cleaned_container_diff.columns + '_games_played'
        df_5=pd.merge(df_4,cleaned_container_diff,left_index=True,right_index=True,how='outer')
        return df_5

    df_5=raw_data_sos(df_4,team_list)
    df_5_dummy=raw_data_sos(df_4_dummy,team_list_dummy)

    cols_to_move=['Date','team','unique_id','opponent','season_year','Week','pts_scored','pts_conceded',
    'Los Angeles Rams_offence', 'Los Angeles Rams_defence','Los Angeles Rams_games_played',
    'season_games_played','away_pts_avg','avg_home_score','avg_away_score','home_away']
    cols = cols_to_move + [col for col in df_5 if col not in cols_to_move]
    df_5=df_5[cols]      

    df_4=df_5.copy()
    # st.write('put in a column for have you played them, then have a cum sum of htat played multiplied by the other col')
    def sos_workings(df_4,team_list):
        for x in team_list:
            df_4[x+' played']=np.where(df_4['opponent']==x,1,0)
            df_4[x+' sum']=df_4.groupby(['team','season_year'])[x+' played'].cumsum()
            # https://stackoverflow.com/questions/53335567/use-pandas-shift-within-a-group
            df_4[x+' sum']=df_4.groupby(['team','season_year'])[x+' sum'].shift(1) # careful to refer back to above column name, got caught with it
            df_4[x+' pts_diff']=df_4[x+' sum'] * (df_4[x+'_offence']-df_4[x+'_defence'])
            df_4[x+' games_use']=df_4[x+' sum'] * (df_4[x+'_games_played'])
        return df_4

    df_4=sos_workings(df_5,team_list)
    df_4_dummy=sos_workings(df_5_dummy,team_list_dummy)

    cols_to_move=['Date','team','unique_id','opponent','season_year','Week','Ireland_offence','Ireland_defence','Ireland pts_diff',
    'Ireland games_use']
    cols = cols_to_move + [col for col in df_4_dummy if col not in cols_to_move]
    df_4_dummy=df_4_dummy[cols]
    # st.write('df4 dummy so for England for 2022 the Ireland pts diff column matches the spreadsheet CHECK LATER why 2021 defence not flowing through', df_4_dummy)


    def sos_workings_2(df_4,team_list):
        sos_container=[]
        grouped_container=[]
        for week,group_df in df_4.groupby(['season_year','Week']):
            for x in team_list:
                group_df.loc [ (group_df['team']==x), x+'_opp_games' ] = group_df[x+' sum'].sum()
                group_df.loc [ (group_df['team']==x), x+'_diff_total' ] = group_df[x+' pts_diff'].sum()
                group_df.loc [ (group_df['team']==x), x+'_total_opp_games' ] = group_df[x+' games_use'].sum()
                group_df.loc [ (group_df['team']==x), x+'_diff_per_game' ] = group_df[x+'_diff_total']/group_df[x+'_total_opp_games']
                # cols_to_move=['Date','team','unique_id','opponent','season_year','Week','pts_scored','pts_conceded',
                # 'Los Angeles Rams_opp_games',
                # 'Los Angeles Rams_diff_per_game','Los Angeles Rams_total_opp_games',
                # 'Los Angeles Rams sum','Los Angeles Rams_diff_total','Los Angeles Rams pts_diff','Los Angeles Rams games_use',
                # 'Los Angeles Rams_offence','Los Angeles Rams_defence','Arizona Cardinals sum','Los Angeles Rams_games_played',
                # 'season_games_played','avg_home_score','avg_away_score','home_away']

                # cols = cols_to_move + [col for col in group_df if col not in cols_to_move]
                # group_df=group_df[cols]
            grouped_container.append(group_df)
        
        adj_team_list=team_list.copy()
        adj_team_list=[x + '_diff_per_game' for x in adj_team_list]
        df_power = pd.concat(grouped_container)

        # cols_to_move=['Date','team','unique_id','opponent','season_year','Week','Ireland_offence','Ireland_defence','Ireland pts_diff',
        # 'Ireland games_use','Ireland_opp_games','Ireland_diff_total','Ireland_total_opp_games','Ireland_diff_per_game']
        # cols = cols_to_move + [col for col in df_power if col not in cols_to_move]
        # df_power=df_power[cols]
        # st.write('df power in the FUNCTION something wrong with the Defence I think in 2021')
        # AgGrid(df_power)
        df_power['sos']=df_power[adj_team_list].bfill(axis=1).iloc[:,0]
        cols_to_move=['Date','team','unique_id','opponent','season_year','Week','pts_scored','pts_conceded','sos']
        cols = cols_to_move + [col for col in df_power if col not in cols_to_move]
        df_power=df_power[cols]
        return df_power

    df_power=sos_workings_2(df_4,team_list)
    df_power_dummy=sos_workings_2(df_4_dummy,team_list_dummy)
    # st.write('check', dd
    # f_power_dummy.columns)


    cols_to_move=['Date','team','unique_id','opponent','season_year','Week','pts_scored','pts_conceded','sos','Los Angeles Rams_opp_games',
    'Los Angeles Rams_diff_per_game','Los Angeles Rams_total_opp_games',
    'Los Angeles Rams sum','Los Angeles Rams_diff_total','Los Angeles Rams pts_diff','Los Angeles Rams games_use',
    'Los Angeles Rams_offence','Los Angeles Rams_defence','Arizona Cardinals sum','Los Angeles Rams_games_played',
    'season_games_played','away_pts_avg','avg_home_score','avg_away_score','home_away']

    cols = cols_to_move + [col for col in df_power if col not in cols_to_move]
    df_power=df_power[cols]

    cols_to_move=['Date','team','unique_id','opponent','season_year','Week','sos','Ireland_offence','Ireland_defence','Ireland pts_diff',
    'Ireland games_use','Ireland_diff_per_game',
    'Ireland_diff_total','Ireland_total_opp_games']
    cols = cols_to_move + [col for col in df_power_dummy if col not in cols_to_move]
    df_power_dummy=df_power_dummy[cols]


    # sos_container=pd.DataFrame(sos_container).transpose()
    # sos_container.columns=sos_container.columns + '_latest'
    # st.write('cleaned container games played', cleaned_container_diff)

    # df_power=pd.merge(df_power,sos_container,left_index=True,right_index=True,how='outer')
    # st.write('check ireland england 2', df_power_dummy[ (df_power_dummy['season_year']==2022) ])
    # st.write('Ireland recalc', df_power_dummy[(df_power_dummy['season_year']==2022) & (df_power_dummy['team']=='Ireland')].sort_values(by='unique_id'))
    st.write('This is the Six Nations SOS, need to fix 2021', df_power_dummy.sort_values(by=['Date','Week','unique_id']))
    
    st.write('df power Week 15', df_power[(df_power['Week']==15) & (df_power['season_year']==2022) ].set_index('team'))
    # st.write('Latest Week is:', df_power['season_year']==2022) [ df_power['Week'].max()]
    st.write('Latest Week is:',df_power[df_power['season_year']==2022]['Week'].max() )
    # st.write('Baltimore Hardest Schedule', df_power[(df_power['team']=='Baltimore Ravens')].set_index('team'))
    # st.write('Patriots Easiest Schedule', df_power[df_power['team']=='New England Patriots'].set_index('team'))


    # https://stackoverflow.com/questions/71255870/calculate-sum-based-on-multiple-rows-from-list-column-for-each-row-in-pandas-dat
    # interesting link above
    # df = pd.DataFrame({'id': [0, 1, 3, 2, 4], 'col_to_sum': [1, 2, 3, 4, 5], 'list_col': [[], [1], [1, 2, 3], [2], [3, 1]]})
    # st.write(df)
    # df = df.set_index('id')
    # df['sum'] = df['list_col'].apply(lambda x: df.loc[x, 'col_to_sum'].sum())
    # df = df.reset_index()
    # st.write(df)
    # st.write('so the key to this is to match up the look up value into the df')
    # st.write('also the key is apply where it goes through it one cell at a time')

    # https://stackoverflow.com/questions/34989341/how-to-remove-nan-value-while-combining-two-column-in-panda-data-frame
    # data = {'Name':['Arizona', 'NY Giants', 'Kansas', 'Buffalo','Denver', 'Raiders'], 'LA':[20, 21, np.NaN, np.NaN,np.NaN,np.NaN],
    #  'Tampa':[np.NaN, np.NaN, 9, 8,np.NaN,np.NaN], 'Houston':[np.NaN, np.NaN, np.NaN, np.NaN,14,17]}
    # df = pd.DataFrame(data)
    # # st.write(df)
    # df['update']=df[['Name','LA','Tampa','Houston']].bfill(axis=1).iloc[:,1]
    # st.write(df[['LA','Tampa','Houston']].bfill(axis=1))
    # st.write(df)