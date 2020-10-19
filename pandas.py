# TURNING LISTS INTO A DATAFRAME ##############################################

# list_keys is a list and will be cols, list_values is a list of lists
zipped = list(zip(list_keys, list_values))
# Build a dictionary with the zipped list: data
data = dict(zipped)
# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)

# PLOTTING STRAIGHT FROM THE DATAFRAME ########################################

# plotting all columns in the dataframe
df.plot()
plt.show()
# Plot all columns as subplots
df.plot(subplots=True)
plt.show()
# Plot just cetain collumns in teh dataframe
df[['col1', 'col2', 'col3']].plot()
plt.show()
# Generate a line plot with jsut a couple columns
df.plot(x='Month', y=['AAPL', 'IBM'])
plt.show()

# PLOTTING USING SUBPLOTS
fig, axes = plt.subplots(nrows=2, ncols=1)
# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))
plt.show()
# Plot the CDF
df.fraction.plot(ax=axes[1], kind='hist', bins=30, normed=True, cumulative=True, range=(0,.3))
plt.show()

# GETTING THE QUANTILES #######################################################

df.quantile([0.1, 0.5, 0.9])
# gives count, unique, most frequent category, occurences of top category
df['col'].describe()


# PLOTTING THE MEAN OF EACH ROW ###############################################

mean = df.mean(axis=1)
mean.plot()
plt.show()

# PREPARING A DATETIME WITH A PARTICULAR FORMAT ###############################

# Prepare a format string: time_format
time_format='%Y-%m-%d %H:%M'
# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format=time_format)
# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)

# EXTRACTING PORTIONS OF A DATETIME INDEX #####################################

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']
# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']
# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc[:'2010-12-31']

# REINDEXING A TIMESERIES #####################################################

# method describes how missing data is filled (default is NaN)
ts4 = ts2.reindex(ts1.index, method='ffill')

# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')

# RESAMPLING TIMESERIES DATA ##################################################

# Downsample to 6 hour data and aggregate by mean: df1
df1 = df.loc[:,'Temperature'].resample('6H').mean()
# Downsample to daily data and count the number of data points: df2
df2 = df.loc[:,'Temperature'].resample('D').count()

# MAKING A ROLLING AVERAGE ####################################################

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling('D').mean() # rolling(window=24).mean()

# Resample to daily data, aggregating by max: daily_highs
daily_highs = august.resample('D').max()
# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
daily_highs_smoothed = daily_highs.rolling(window=7).mean()

# SEARCHING FOR STRINGS WITH SUBSTRING IN #####################################

df['col'].str.contains('string')
# Becasue True = 1, gives the number of entries with 'string' in them
df['col'].str.contains('string').sum()

# PAD LEADING 0s TO A COL #####################################################

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# FORCE A COLUMN TO BE NUMERICAL ##############################################

# turns non-numerical columsn to NaNs
df_clean['dry_bulb_faren'] = \
    pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# INDEXING ####################################################################

df['row']['col']
df.loc['col', 'row']
df.iloc[rownum, colnum]
df_new = df[['col1', 'col2']]

# FILTERING ###################################################################

df[(df.col1 == 'yes') | (df.col2 > 50)] # | is an or statement

df.loc[:, df.all()] # selects columns with no 0s
df.loc[:, df.any()] # selects columns with any non-zero entries (not all 0)
df.loc[:, df.isnull().any()] # selects coumns that an have NaNs in
df.loc[:, df.notnull().all()] # selects columns with no nan values

df.dropna(how='any') # drops rows with any NaNs
df.dropna(how='all', axis = 1) # drops columsn with all NaNs

df.col1[df.col2 > 50] +=5 # adds 5 to col 1 when col2 is over 50

# drops columns with less than 1000 non nan
titanic.dropna(thresh=1000, axis='columns')

# DATAFRAME TRANSFORMATIONS ###################################################

def to_celsius(F):
    return 5/9*(F - 32)
# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)

# the dictionary changing from keys to values
red_vs_blue = {'Obama':'blue', 'Romney':'red'}
# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)

# SETTING A MULTI INDEX #######################################################

# Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state', 'month'])
# Sort the MultiIndex: sales
sales = sales.sort_index()

# REFERRENCING A MULTI INDEX ##################################################

# Look up data for NY in month 1 in sales: NY_month1
NY_month1 = sales.loc[('NY', 1), :]
# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA', 'TX'], 2), :]
# Access the inner month index and look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None), 2), :]

# MAKING A PIVOT TABLE ########################################################

# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(index = 'weekday',
                            columns = 'city',
                            values = 'signups')
# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index = 'weekday',
                    columns = 'city')

# WHEN THERE ARE MULTIPLE VALUES FOR INDEX COLUMN PAIRS #######################

# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index = 'weekday',
                                      columns = 'city',
                                      aggfunc = 'count')

# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index = 'weekday',
                                        aggfunc = sum)

# Add in the margins: signups_and_visitors_total
signups_and_visitors_total = users.pivot_table(index = 'weekday',
                                              aggfunc = sum,
                                              margins = True)

# STACKING AND UNSTACKING #####################################################

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')
# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0, 1)

# GROUP BY, AGGREGATING AND APPLYING ##########################################

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked', 'pclass'])
# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()

# GROUPING BY ANOTHER DF. HAS TO BE SAME INDEX ################################

# Read life_fname into a DataFrame: life
life = pd.read_csv(life_fname, index_col='Country')
# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col='Country')
# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])
# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())

# CAN ACCEPT MULIPLE STAISTICAL METHODS #######################################

# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')
# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]
# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max', 'median'])
# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])
# Print the median fare in each class
print(aggregated.loc[:, ('fare', 'median')])

# PERSONAL AGGREGATORS ########################################################

# aggregation becuase it take a series and returns a single number
# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(['Year', 'region'])
# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()
# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}
# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# TRANSFORMATIONS INSTEAD OF AGGREGATORS ######################################

# goes from a series to a series
from scipy.stats import zscore
# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby('region')['life', 'fertility'].transform(zscore)

# IDENTIFYING OUTLIERS ########################################################

standardized = gapminder_2010.groupby('region')['life', 'fertility'].transform(zscore)
# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)
# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# FILLNAa WITH MEDIANS FROM CERTIAN SUBSETS OF THE DATA #######################

# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex', 'pclass'])
# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())
# Impute age and assign to titanic['age']
titanic.age = by_sex_class['age'].transform(impute_median)

# WHEN THE TRANSFORMATION IS TOO COMPLICATED FOR TRANSORM. APPLY ##############

def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})
regional = gapminder_2010.groupby('region')
# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# GROUP BY WITH FILTERING #####################################################

def c_deck_survival(gr):
    c_passengers = gr['cabin'].str.startswith('C').fillna(False)
    return gr.loc[c_passengers, 'survived'].mean()
by_sex = titanic.groupby('sex')
# Call by_sex.apply with the function c_deck_survival
c_surv_by_sex = by_sex.apply(c_deck_survival)

# or

by_company = sales.groupby('Company')
# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)

# or (by using another df.index as group)

under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})
# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10, 'pclass'])['survived'].mean()

# FOR WORKING OUT THE TOP OR BOTTOM VALUES OF A COLUMN ########################

counted = counted.sort_values('totals', ascending=False)

# findng the coutry witht eh most unique sports to win medals in
country_grouped = medals.groupby('NOC')
# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()
# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)
