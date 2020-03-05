from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
import pickle
import requests
from unidecode import unidecode

# this dictionary will map players to a set containing all the years in which they were selected for an all-star game, either initially or as a replacement
all_star_appearances = defaultdict(set)

# rows to ignore when iterating the roster tables
ignore_fields = set(['Team Totals', 'Reserves'])

START_YEAR, END_YEAR = 1970, 2020

 # unidecode doesn't catch the accented c in Peja's last name (Stojakovic), fix it
 # also overwrite any instance of Metta World Peace to Ron Artest
def fix_name(full_name):
	first_name = full_name.split(' ')[0]
	if first_name == 'Peja':
		return 'Peja Stojakovic'
	elif first_name == 'Metta':
		return 'Ron Artest'
	else:
		return unidecode(full_name)

for year in range(START_YEAR, END_YEAR):

	# no ASG played in 1999 because of the lockout
	if year == 1999:
		continue

	print('Scraping ASG {} data...'.format(year))

	# will store all the all-stars for this year
	all_stars = set([])

	html = requests.get('https://www.basketball-reference.com/allstar/NBA_{}.html'.format(year)).content
	soup = BeautifulSoup(html, 'html.parser')

	# this part was annoying - back when ASG was always East vs. West, the tables were encoded with id="East"/id="West" so they could be extracted more easily/reliably
	# but now, you have games like Giannis vs. LeBron and the table id's are different, so I had to extract them by index, which is unreliable in the event that the 
	# site's design changes in the future

	# gets rosters for team 1 and team 2
	s1, s2 = soup.findAll('table')[1:3]

	df1 = pd.read_html(str(s1))[0]
	df2 = pd.read_html(str(s2))[0]

	# get the all-stars from teams 1 and 2
	for df in [df1, df2]:
		for i, row in df.iterrows():
			if pd.notnull(row[0]) and row[0] not in ignore_fields:
				player = row[0]
				all_stars.add(fix_name(player))

	# gets all li elements in the page
	s3 = soup.findAll('li') 

	# finds the li element that contains the data pertaining to injury related selections - players who were selected but couldn't participate due to injury,
	# and their respective replacements
	#
	# since all_stars is a hashset, we don't need to worry about accidentally double counting an all-star
	for s in s3:
		if 'Did not play' in str(s):
			for player in [name.get_text() for name in s.findAll('a')]: # all the injured players and their replacements
				all_stars.add(fix_name(player))
			break

	# update the appearances dictionary
	for player in all_stars:
		all_star_appearances[player].add(year)

sorted_all_star_appearances = sorted([(player, sorted(list(appearances))) for player, appearances in all_star_appearances.items()], key = lambda x : -len(x[1]))

print('\nAll all-star appearances since 1970 (sorted by number of appearances):\n')

for player, appearances in sorted_all_star_appearances:
	print('{}: {}'.format(player, appearances))

# export the dictionary to local disk for future recall in statsnba_fullscrape.py
out = open('all_star_appearances.pickle', 'wb')
pickle.dump(all_star_appearances, out)
out.close