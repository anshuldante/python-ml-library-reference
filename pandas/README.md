# Pandas Reference

| Method                                                                                  | Description                                                                 |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| pd.read_csv                                                                             | Read a CSV file                                                             |
| df.info()                                                                               | Column names, Non-null count and data-types                                 |
| df.columns                                                                              | Column names                                                                |
| df.dtypes                                                                               | Columns datatypes                                                           |
| df['col'] *OR* df.col                                                                   | column name, length, datatype and some data                                 |
| df['col'].unique()                                                                      | unique values in the column                                                 |
| df['col'].value_counts(normalize=True)                                                  | counts by unique values, normalize=True returns ratios False returns counts |
| df[df['col'] == 'val'] *OR* df.query('col' == 'val')                                    | rows filtered by column = value                                             |
| df[df.col.str.contains('val')]                                                          | rows with col having the str 'val'                                          |
| df = pd.DataFrame(np.load('x.npy'))                                                     | data frame from npy file or from a numpy array                              |
| df.set_index('col)                                                                      | use selected columns as index instead of the default row numbers            |
| df.loc['val], df.loc[[1,2,3]], df.loc[1:50, 'col']                                      | rows based on a single, list or slice or labels for rows and/or columns     |
| df.sort_index()                                                                         | sort the index in default ascending order                                   |
| df.index.nlevels                                                                        | number of levels in the index                                               |
| df.index.get_level_values(1)                                                            | all values in the passed level number                                       |
| df.groupby('col')['col2].mean(), titanic.groupby(titanic['Age'].isnull())               | Group by and aggregate                                                      |
| gapminder.pivot_table('babies', 'year', 'region')                                       | segment babies data by both year and region, then take mean                 |
| titanic.pivot_table('Survived', index='Cabin_ind', columns='Embarked', aggfunc='count') | Survival count against cabin indicator and embarked                         |
| pd.read_csv(x, na_values=['?'])                                                         | defines that '?' is the nan symbol in the dataset                           |
| df.astype({"normalized-losses": "int64")                                                | to assign types to columns                                                  |
| df.columns = ['symboling']                                                              | to assign names to columns                                                  |
| df.horsepower=df.horsepower.fillna(df.horsepower.mean())                                | fill nan values                                                             |
| df['cylinders'] = df['num-of-cylinders'].map(num_map)                                   | map values in a column using dictionary                                     |
| df = df.dropna(subset=['price'])                                                        | drop nans for the specified columns                                         |
| df = df.reset_index(drop=True)                                                          | reset index                                                                 |
| df.horsepower=df.horsepower.fillna(df.horsepower.mean())                                | fill nans with the mean                                                     |
| pd.get_dummies(iris, columns=["species"])                                               | get dummy variables with values for the specified column                    |
| df.corr()                                                                               | Correlation matrix for all columns                                          |
| df.drop(['B', 'C'], axis=1)                                                             | drop a column                                                               |
| df.drop([0, 1])                                                                         | drop by index                                                               |
| pd.concat([cars, dummies], axis=1)                                                      | concatenate 2 or more dataframes                                            |
| pd.sort_values('col',ascending=False)                                                   | sort dataframe by column                                                    |
| train['Fare'].nlargest(10)                                                              | 10 largest values of fare                                                   |
| train['Age'].nsmallest(10)                                                              | 10 smallest values of age.                                                  |
| train.isnull().sum()                                                                    | number of null values in every column                                       |
| d = train.groupby('familySize')['Survived'].value_counts(normalize = True).unstack()    | Reindex and flatten the dataframe                                           |
| pd.qcut(heart.age, 10)                                                                  | Create 10 quintiles (deciles)                                               |
| y_train.to_csv('file.csv', index=False)                                                 | save as a csv                                                               |

```python
# Different ways to create dataframes from raw data.
pd.DataFrame([{'title': 'David Bowie', 'year': 1969},
              {'title': 'The Man Who Sold the World', 'year': 1970},
              {'title': 'Hunky Dory', 'year': 1971}])
pd.DataFrame([('Ziggy Stardust', 1), ('Aladdin Sane', 1), ('Pin Ups', 1)], columns=['title','toprank'])
pd.DataFrame({'title': ['David Bowie', 'The Man Who Sold the World', 'Hunky Dory',
                        'Ziggy Stardust', 'Aladdin Sane', 'Pin Ups', 'Diamond Dogs',
                        'Young Americans', 'Station To Station', 'Low', 'Heroes', 'Lodger'],
              'release': ['1969-11-14', '1970-11-04', '1971-12-17', '1972-06-16',
                          '1973-04-13', '1973-10-19', '1974-05-24', '1975-03-07',
                          '1976-01-23', '1977-01-14', '1977-10-14', '1979-05-18']})
```

```python
# Different ways to select rows and columns in case of multi-index
nobels_multi.loc[(slice(1901,1910), 'Chemistry'),:]
nobels_multi.loc[(range(2000), slice(None)), :]
nobels_multi.loc[(slice(None), 'Chemistry'),:]
```

```python
# fancy index with multiple conditions
nobels[(nobels.year >= 1901) & (nobels.year <= 1910) & (nobels.discipline == 'Chemistry')]
nobels.query('year >= 1901 and year <= 1910 and discipline == "Chemistry"')
```

```python
# Create a scatter plot with multiple subplots
axes = gapminder[gapminder['country'] == 'China'].sort_values('year').plot('year', 'life_expectancy', label='China')
gapminder[gapminder['country'] == 'Italy'].sort_values('year').plot('year', 'life_expectancy', label='Italy', ax=axes)
gapminder[gapminder['country'] == 'United States'].sort_values('year').plot('year', 'life_expectancy', label='USA', ax=axes)
gapminder[gapminder['country'] == 'India'].sort_values('year').plot('year', 'life_expectancy', label='India', ax=axes)
pp.ylabel('life expectancy')
```

```python
# "pivot" the third level of the multiindex (years) to create a row of columns;
# result is names (rows) x years (columns)
allyears_indexed.loc[('F',claires),:].unstack(level=2)

# fix stacked plot by filling NaNs with zeros, adding labels, setting axis range
pp.figure(figsize=(12,2.5))
pp.stackplot(range(1880,2019),
             allyears_indexed.loc[('F',claires),:].unstack(level=2).fillna(0),
             labels=claires);

pp.legend(loc='upper left')
pp.axis(xmin=1880, xmax=2018);
```

```python
# get the top ten names for sex and year

def getyear(sex, year):
    return (allyears_byyear.loc[sex, year]             # select M/F, year
               .sort_values('number', ascending=False) # sort by most common
               .head(10)                               # only ten
               .reset_index()                          # lose the index
               .name)                                  # return a name-only Series
```

```python
# get all time favorites: select F, group by name, sum over years, sort, cap 
alltime_f = allyears_byyear.loc['F'].groupby('name').sum().sort_values('number', ascending=False).head(10)
```

```python
# Fill in the missing values of age while grouping over Pclass due to the big correlation value between the 2
train.loc[train.Age.isnull(), 'Age'] = train.groupby("Pclass").Age.transform('median')
```

```python
# Fare and Pclass also had a hig correlation, so grouping by Pclass and filling null values
train['Fare']  = train.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
```

```python
# Take the first characters of cabins and then map into numbers
train['Cabin'] = train['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
train['Cabin'] = train['Cabin'].map(cabin_category)
```

```python
# Save the data as an HTML and open it in a web-browser
html = house[0:100].to_html()
with open("data.html", "w") as f:
    f.write(html)

full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))
html = house[0:100].to_html()
with open("data.html", "w") as f:
    f.write(html)

full_filename = os.path.abspath("data.html")
webbrowser.open("file://{}".format(full_filename))
```
