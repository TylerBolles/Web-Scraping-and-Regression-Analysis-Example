""" makeData.py scapes movie data from boxofficeguru.com, creating .csv files as a result. As written,
	movies with titles beginning with a, b or c are treated as training data while those beginning with 
	d are treated as testing data and are saved in a seperate file. """
import pandas as pd
from urllib.error import HTTPError

def scrape_data(url):
	""" scrape_data obtains data from the only table in each webpage of the form 
	    'http://www.boxofficeguru.com/[Letter][Number].htm where 
	    Letter and Number are variables. The table (pandas dataframe) containing
	    data from all urls with a valid Number is returned. """
	table = pd.read_html(url, header = 0)[0]
	i = 1
	valid_url = True
	# loop through all numbers such that the resulting url is still valid
	while valid_url:
		i += 1
		try: 
			table = table.append(pd.read_html(url.replace('.h', str(i)+'.h'), header=0)[0])
		except HTTPError:
			valid_url = False
			
	table.rename(columns = 
				{'Gross': 'Total', 
				 'Gross.1': 'OpeningGross',
				 'Theaters': 'OpeningThtrs', 
				 'Theaters.1': 'MaxThtrs',
				 'Unnamed: 3': 'WkndLength'},
				inplace = True)
				
	table.drop('WkndLength', axis = 1, inplace = True) # unnecessary data
	
	# drop any row which lacks a title or a total revenue; some fake rows were
	# added the table due to the formatting of the webpage
	return table[table['Title'].notna() & table['Total'].notna()]
	
url = 'http://www.boxofficeguru.com/.htm'

# build table for training data
train_data = pd.DataFrame()
for letter in ['a','b','c']:
	train_data = train_data.append(scrape_data(url.replace('.h', letter + '.h')))
 
# reformat and save
train_data.reset_index(drop = True, inplace = True)
train_data.to_csv('TrainingData.csv', sep = '\t')

# build, format and save testing data
test_data = scrape_data(url.replace('.h', 'd.h')).reset_index(drop = True)
test_data.to_csv('TestingData.csv', sep = '\t')
