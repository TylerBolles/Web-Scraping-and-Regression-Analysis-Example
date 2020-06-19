""" moviePredict.py estimates the total return of a movie using linear regression 
	and plots the results. The .csv files produced by makeData.py must be in the
	same directoy in order for moviePredict to run. """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices, dmatrix
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

def print_results(y, yH):
	""" for printing some summary results """
	error = abs(1-np.exp(yH-y)) # y and yH are logs of revenue.
	print('Mean Error: ', "{:.2f}".format(np.mean(error)))
	print('Median Error: ', "{:.2f}".format(np.median(error)))
	print('R^2: ', "{:.2f}".format(np.var(yH)/np.var(y)), '\n')
	
def make_plots(yH, df, num_covars):
	""" make_plots plots 'LogTotal' and its fitted valus, yH, against the first
		num_covars (num_covars must be less than 6) in the dataframe, df, while 
		color coding by 'Distributor'. """
	distributors = np.sort(np.unique(df['Distributor']))
	num_groups = len(distributors)
	# color code by Distributor. Make colors 'increase' with increaing
	# average return for the Distributor.
	color_map = np.exp(df.groupby('Distributor').median()['LogTotal'])
	color_map -= -min(color_map)
	color_map /= max(color_map) # now color_map lies in the interval [0,1]
	# compact graphs sharying the same LogTotal axis
	fig, ax = plt.subplots(1, num_covars, sharey = True)
	fig.tight_layout(pad=1.08, h_pad = None, w_pad = None, rect = None)
	plt.subplots_adjust(wspace = 0, hspace = 0)
	for i in range(num_covars):
		ax[i].set_xlabel(df.columns[i+1])
		# enumerate though distributors to color code observed values
		for j in range(num_groups):
			subdf = df.loc[df['Distributor'] == distributors[j]] 
			ax[i].scatter(subdf[subdf.columns[i+1]], subdf['LogTotal'], 
						  color = cm.seismic(color_map[j]), alpha=0.5)
		# add fitted values all in black				  
		ax[i].scatter(df.iloc[:, i+1], yH, color = 'k', alpha = .8, 
					  marker = 'x')
		# eliminate outermost x-tick lables			  
		ax[i].xaxis.set_major_locator(MaxNLocator(prune='both'))
		ax[i].xaxis.label.set_size(20)
	ax[0].set_ylabel('LogTotal', fontsize = 20)
	plt.show()

def transform_data(file_name):
	""" transform_data opens the input dataframe df and returns a transformation of it"""
	
	df = pd.read_csv(file_name, index_col = 0, sep = '\t')
	
	# take logs of positive prices
	df[['LogTotal', 'OpeningGross']] = np.log(df[['Total', 'OpeningGross']])

	# close all movies by at most the date the website stopped updating 
	df['Closing'].fillna('2010-09-20', inplace = True) 

	# convert dates to datetime type for further manipulation
	df[['Opening', 'Closing']] = df[['Opening', 'Closing']].apply(pd.to_datetime)

	# add a day of the year covariate
	df['Day'] = df['Opening'].dt.dayofyear

	# add a number of days played covariate
	df['DaysPlayed'] = (df['Closing']-df['Opening']).dt.days

	# convert opening date to days since jan 1 1989, the earliest possible
	# movie premier on the website
	df['Opening'] = (df['Opening']-pd.to_datetime('1989-01-01')).dt.days

	# the columns to keep for regression analysis
	columns = ['LogTotal', 'OpeningGross', 'MaxThtrs', 'Day', 
			   'DaysPlayed', 'Opening', 'Distributor']
	# drop an rows missing at least one point for the relavent columns
	return df[columns].dropna()

# the regression model
model = "Opening + np.log(DaysPlayed) + OpeningGross \
		 + MaxThtrs + bs(Day, df=7)+ Distributor"
					
# training stage
train_df = transform_data('TrainingData.csv') # data frame
train_dm = dmatrix(model, train_df, return_type = 'dataframe') # design matrix
y = train_df['LogTotal']

fit_results = sm.OLS(endog = y, exog = train_dm).fit() #
yH = fit_results.fittedvalues # fitted values

print('Training Phase')
print_results(y, yH)
make_plots(yH, train_df, 4)

# testing stage
test_df = transform_data('TestingData.csv') # data frame
test_dm = dmatrix(model, test_df, return_type = 'dataframe') # design matrix
y = test_df['LogTotal']
# we must enforce that the design matrix for the testing data has the same
# number of columns as that of the training data appearing in the same order. 
# these can be different due to the encoding of the Distributor variable.
# an example name of these columns is 'Distributor[T.After Dark]'
# the following assumes that no new distributors appear in the testing data.
for col in list(test_dm):
	if 'Distributor' in col:
		test_dm.drop(col, axis = 1, inplace = True)

for i, col in enumerate(list(train_dm)):
	if 'Distributor' in col:
		test_dm.insert(i, col, train_dm[col])

yH = np.dot(test_dm, fit_results.params)

print('Testing Phase')
print_results(y, yH)
make_plots(yH, test_df, 4)


