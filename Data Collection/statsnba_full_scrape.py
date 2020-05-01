from bs4 import BeautifulSoup
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import Select

import pickle
import time

START_YEAR, END_YEAR = 1996, 2020

# list of DataFrames for historical data, one for each year
df_train_master = []

# DataFrame containing features for the current year which we will later use 
# to estimate ASG selection probabilities
df_to_predict = pd.DataFrame()

# let's us look up team record and rank by (year, team)
# we need this to augment our player dataset but we need to construct it first
team_rank_historical_lookup = {}

# load the all-star history dictionary that we generated in bballref_ASG_scrape.py
all_star_appearances = pickle.load(open('all_star_appearances.pickle', 'rb'))

# we need a map from the team's full name to their short form (prefix)
team_prefix = {
	'Atlanta Hawks' : 'ATL',
	'Boston Celtics' : 'BOS',
	'Charlotte Hornets Old' : 'CHH', # deprecated
	'Chicago Bulls' : 'CHI',
	'Cleveland Cavaliers' : 'CLE',
	'Dallas Mavericks' : 'DAL',
	'Denver Nuggets' : 'DEN',
	'Detroit Pistons' : 'DET',
	'Golden State Warriors' : 'GSW',
	'Houston Rockets' : 'HOU',
	'Indiana Pacers' : 'IND',
	'Los Angeles Clippers' : 'LAC', # deprecated
	'LA Clippers' : 'LAC',
	'Los Angeles Lakers' : 'LAL',
	'Miami Heat' : 'MIA',
	'Milwaukee Bucks' : 'MIL',
	'Minnesota Timberwolves' : 'MIN',
	'New Jersey Nets' : 'NJN', # deprecated
	'New York Knicks' : 'NYK',
	'Orlando Magic' : 'ORL',
	'Philadelphia 76ers' : 'PHI',
	'Phoenix Suns' : 'PHX',
	'Portland Trail Blazers' : 'POR',
	'Sacramento Kings' : 'SAC',
	'San Antonio Spurs' : 'SAS',
	'Seattle SuperSonics' : 'SEA', # deprecated
	'Toronto Raptors' : 'TOR',
	'Utah Jazz' : 'UTA',
	'Vancouver Grizzlies' : 'VAN', # deprecated
	'Washington Bullets' : 'WAS', # deprecated
	'Washington Wizards' : 'WAS',
	'Memphis Grizzlies' : 'MEM',
	'New Orleans Hornets' : 'NOH', # deprecated
	'Charlotte Bobcats' : 'CHA', # deprecated
	'New Orleans/Oklahoma City Hornets' : 'NOK', # deprecated
	'Oklahoma City Thunder' : 'OKC',
	'Brooklyn Nets' : 'BKN',
	'Charlotte Hornets New' : 'CHA',
	'New Orleans Pelicans' : 'NOP'
}

# Charlotte's short form in the pre-Bobcats era was CHH but now it's CHA, so we adjust accordingly
def adjust_hornets(row):
	if row['TEAM'] == 'Charlotte Hornets':
		return 'Charlotte Hornets Old' if row['Year'] <= 2001 else 'Charlotte Hornets New'
	return row['TEAM']

# function will construct our team rank lookup by year (nested dictionary data structure)
def fill_team_rank_historical_lookup(row):
	year = row['Year']
	team = row['TEAM']
	rank = row['Conference Rank']
	gp = row['GP']
	prefix = team_prefix[team]
	if year not in team_rank_historical_lookup:
		team_rank_historical_lookup[year] = {}
	team_rank_historical_lookup[year][prefix] = (rank, gp)

# we also need average league pace by year, so we can normalize all statistics to be pace-adjusted
html = requests.get('https://www.basketball-reference.com/leagues/NBA_stats_per_game.html').content
s_pace = BeautifulSoup(html, 'html.parser')

pace_table = s_pace.find('table')
df_pace = pd.read_html(str(pace_table))[0]
df_pace.columns = df_pace.columns.droplevel()

# maps year to average league pace
pace_lookup = {}

for i, row in df_pace.iterrows():
	if pd.isnull(row['Season']) or row['Season'] == 'Season':
		continue
	year = int(row['Season'][:4])
	pace_lookup[year] = row['Pace']
	if year == START_YEAR:
		break

# this function looks up if a player was selected for the ASG in the prior year
# this could have been done succintly in a lambda function, but the 1999 lockout added an annoying wrinkle
def was_AS_last_year(row):
	if row['Year'] == 1999:
		return 1 if 1998 in all_star_appearances[row['PLAYER']] else 0
	return 1 if row['Year'] in all_star_appearances[row['PLAYER']] else 0

# initialize the chromedriver
d = webdriver.Chrome('./chromedriver')

# crude time delay to wait before attempting to scrape tabular data after XML document has loaded
TIME_DELAY_TEAMS = 3
TIME_DELAY_PLAYERS = 10

for year in range(START_YEAR, END_YEAR):

	if year == 1998: # lockout
		continue

	start_date = (10, 1, year) # month, day, year (not padded)
	end_date = (1, 21, year+1) # month, day, year (not padded)

	season_label = str(year) + '-' + str(year+1)[2:]
	print('Scraping stats.nba.com for {} season...'.format(season_label))

	# contains the majortiy of our desired statistics (PTS, REB, AST, etc.)
	url_players_traditional = '''https://stats.nba.com/players/traditional/?Season={}&SeasonType=Regular%20Season&sort=PTS
	&dir=-1&DateFrom={}%2F{}%2F{}&DateTo={}%2F{}%2F{}'''.format(season_label, *start_date, *end_date)
	
	# contains advanced statistics (TS%, USG%, PIE)
	url_players_advanced = '''https://stats.nba.com/players/advanced/?Season={}&SeasonType=Regular%20Season&sort=PTS
	&dir=-1&DateFrom={}%2F{}%2F{}&DateTo={}%2F{}%2F{}'''.format(season_label, *start_date, *end_date)
	
	# contains DEFWS (defensive win-shares)
	url_players_defense = '''https://stats.nba.com/players/defense/?Season={}&SeasonType=Regular%20Season&sort=DEF_WS
	&dir=-1&DateFrom={}%2F{}%2F{}&DateTo={}%2F{}%2F{}'''.format(season_label, *start_date, *end_date)
	
	# contains team rankings by conference at any instance of time
	url_teams = '''https://stats.nba.com/teams/traditional/?sort=W_PCT
	&dir=-1&Season={}&SeasonType=Regular%20Season&Conference={}&DateFrom={}%2F{}%2F{}&DateTo={}%2F{}%2F{}'''

	for conf in ['East', 'West']:
		d.get(url_teams.format(season_label, conf, *start_date, *end_date))

		# crude time delay to ensure element is loaded, definitely a more elegant way to do this
		time.sleep(TIME_DELAY_TEAMS)

		s_teams = BeautifulSoup(d.page_source, 'html.parser').find('table') 
		df = pd.read_html(str(s_teams))[0]
		df['Year'] = year
		df['Conference'] = conf
		df.rename(columns={'Unnamed: 0' : 'Conference Rank'}, inplace=True)

		df['TEAM'] = df[['TEAM','Year']].apply(adjust_hornets, axis=1)
		df[['TEAM', 'Year', 'Conference Rank', 'GP']].apply(fill_team_rank_historical_lookup, axis=1)

	d.get(url_players_traditional)

	time.sleep(TIME_DELAY_PLAYERS)

	# by default, only 50 players are displayed per page, but we can change this using the dropdown select element
	select = Select(d.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select'))
	select.select_by_visible_text('All')

	s_traditional = BeautifulSoup(d.page_source, 'html.parser').find('table')

	d.get(url_players_advanced)

	time.sleep(TIME_DELAY_PLAYERS)

	select = Select(d.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select'))
	select.select_by_visible_text('All')

	s_advanced = BeautifulSoup(d.page_source, 'html.parser').find('table')

	d.get(url_players_defense)

	time.sleep(TIME_DELAY_PLAYERS)

	select = Select(d.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[1]/div/div/select'))
	select.select_by_visible_text('All')

	s_defense = BeautifulSoup(d.page_source, 'html.parser').find('table')

	df_traditional = pd.read_html(str(s_traditional))[0].dropna(subset=['PLAYER'])

	df_advanced = pd.read_html(str(s_advanced))[0].dropna(subset=['PLAYER'])

	df_defense = pd.read_html(str(s_defense))[0].rename(columns={'Player' : 'PLAYER'}).dropna(subset=['PLAYER'])
	
	df = df_traditional.merge(df_advanced[['PLAYER','TS%', 'USG%', 'PIE']], on='PLAYER')
	df = df.merge(df_defense[['PLAYER', 'DEFWS']], on='PLAYER')

	# stitching it all together
	df['Year'] = year
	df['Avg. Pace'] = df['Year'].map(lambda x : pace_lookup[x])
	df['Team Conference Rank'] = df[['TEAM', 'Year']].apply(lambda row : team_rank_historical_lookup[row['Year']][row['TEAM']][0], axis=1)
	df['Team GP'] = df[['TEAM', 'Year']].apply(lambda row : team_rank_historical_lookup[row['Year']][row['TEAM']][1], axis=1)
	df['PLAYER'] = df['PLAYER'].map(lambda x : 'Ron Artest' if x == 'Metta World Peace' else x)
	df['Prior ASG Appearances'] = df[['PLAYER', 'Year']].apply(lambda row : sum(y<=row['Year'] for y in all_star_appearances[row['PLAYER']]), axis=1)
	df['AS Last Year?'] = df[['PLAYER', 'Year']].apply(was_AS_last_year, axis=1)
	df['Selected?'] = df[['PLAYER', 'Year']].apply(lambda row : 1 if row['Year']+1 in all_star_appearances[row['PLAYER']] else 0, axis=1)

	# desired raw features, before any feature engineering/transformation
	df = df[['Year', 'Avg. Pace', 'PLAYER', 'TEAM', 'Team Conference Rank', 'GP', 'Team GP', 'W', 
			'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'TS%', '3PM', 'DEFWS', 'USG%', 'PIE', 'Prior ASG Appearances', 'AS Last Year?', 'Selected?']]

	if year < END_YEAR-1:
		df_train_master.append(df)
	else:
		df_to_predict = df.drop('Selected?', axis=1)

d.quit()

pd.concat(df_train_master).to_csv('ASG_train.csv', index=False)
df_to_predict.to_csv('ASG_to_predict.csv', index=False)
