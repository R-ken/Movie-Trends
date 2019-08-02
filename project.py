# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 00:47:45 2019

@author: Rocky
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats


def main():
    wikidata = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True, encoding="utf8")
    rotten_tomato = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
    
    # Do the various criteria for success (critic reviews, audience reviews, profit/loss) correlate with each other?
    
    # 1. see if audience average and critic average correlate with one another. Look for the correlation coefficient and see what it gives
    
    # clean up data a bit, remove NULL rows and NaN values
    rotten_noNull = rotten_tomato[rotten_tomato['audience_average'].notnull()]
    rotten_noNull = rotten_noNull.dropna()
    
    random = rotten_noNull.sample(n=1000)
    
    x1 = np.array(random['critic_average'])
    y1 = np.array(random['audience_average'])
    
    reg1 = stats.linregress(x1,y1)
    
    # print p value which will tell us if the regression has a slope
    # print r value which tells us if the two data sets are correlated
    
    print(reg1.pvalue)  # 0
    print(reg1.rvalue)  # 0.6991
    
    # plot
    
    plt.figure(figsize=(10, 5)) # change the size to something sensible
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, '.')
    plt.title('Audience and Critic Average')
    plt.xlabel('Critic Average')
    plt.ylabel('Audience Average')
    
    # From the plot, we can see where most of the values are located and the relation between critic and audience average
    
    # 2. See if the country of origin matters in the percentage of people who rate the movie (chi squared)
    
    merge = pd.merge(wikidata, rotten_tomato, on=['imdb_id'])
    
    # Select data we want and remove any null or NaN values
    data3 = merge[['country_of_origin', 'audience_percent', 'critic_percent']]
    data3 = data3[data3['audience_percent'].notnull()]
    data3 = data3.dropna()
    
    data3 = data3.groupby('country_of_origin').mean()
    
    # begin chi squared test
    
    contingency = data3.iloc[:,0:2].values
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(p) #0.1711 so p > 0.05, so categories do not matter or equivalently: the country of origin does not affect the percentage of ratings
    
    
    # 3. See which country of origin contains the most movies
    # take a snip of the top few rows showing the results
    
    result3 = wikidata.groupby('country_of_origin').count().sort_values('imdb_id', ascending=False)
    
    # 4. Do a T-Test on the total audience average of who liked it and total critic average of positive reviews to see if in general, 
    # does the audience like movies easier than critics giving positive reviews ( need to make sure they are normal and equal variance )
    
    # we have lots of data values, CLT would assume normality
    
    # test equal variance 
    
    # 5. Split movies so one set is 2000 and before and the other is 2000 and after. Take the audience average of them both and do a mann whitney test to see if people like the older movies more or the newer ones
    # Can do the same for critic average and see what critics think too
    
    # create a year dataframe which contains the publication date in int years and the imdb_id
    year = wikidata
    year = year[['imdb_id', 'publication_date']]
    year['publication_date'] = year['publication_date'].str[0:4]
    
    year = year.dropna()
    
    year['publication_date'] = year['publication_date'].astype(int)
    
    # Create year_less_2000 dataframe which contains all movies published before 2000
    
    year_less_2000 = year[year['publication_date'] < 2000]
    
    # Create year_2000 dataframe which contains all movies published on 2000 or later
    
    year_2000 = year[year['publication_date'] >= 2000]
    
    merge5_1 = pd.merge(rotten_tomato, year_less_2000, on=['imdb_id'])
    merge5_2 = pd.merge(rotten_tomato, year_2000, on=['imdb_id'])
    
    year_less_2000_avg = merge5_1[['audience_average']]
    year_2000_avg = merge5_2[['audience_average']]
    
    year_less_2000_avg = year_less_2000_avg.dropna()
    year_2000_avg = year_2000_avg.dropna()
    
    # MAnn Whitney test requires both data sets to be the same size so I took the first 1000 values
    year_less_2000_avg2 = year_less_2000_avg.sample(n=1000)
    year_2000_avg2 = year_2000_avg.sample(n=1000)
    
    year_less_2000_avg2 = year_less_2000_avg2.values
    year_2000_avg2 = year_2000_avg2.values
    
    year_less_2000_avg2 = year_less_2000_avg2.ravel()
    year_2000_avg2 = year_2000_avg2.ravel()
    
    print(stats.mannwhitneyu(year_less_2000_avg2, year_2000_avg2).pvalue) # we get 0.0096 which is < 0.05 so we do not reject null hypothesis and conclude that both data sets have similar averages
    
    # Can we predict ratings from some given audience percent, critic average and critic percent?
    # rotten_noNull contains no null roows
    
    MLdata = rotten_noNull[['audience_average','audience_percent', 'critic_average','critic_percent']]
    
    X = MLdata.iloc[:,1:4].values
    y = MLdata.iloc[:,0].values.astype('int')
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y)
    
    model = RandomForestClassifier(n_estimators = 100,
                                   max_depth=3, min_samples_leaf=10)
    
    model.fit(X_train, y_train)
    
    print(model.score(X_valid, y_valid))   #0.84 score


if __name__ == "__main__":
    main()
































