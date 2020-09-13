# Seaborn Reference

| Method                                                                               | Description                                                                                                       |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| sns.boxplot(x='target',y='thalach', data=df)                                         | box plot                                                                                                          |
| sns.pairplot(iris, hue='species')                                                    | plot all variables against each other                                                                             |
| sns.regplot(x="petal_length", y="petal_width", data=iris, order=2)                   | plot x vs y and add a regression line (order decides the power of input)                                          |
| sns.lmplot(x="petal_length", y="petal_width", data=iris, hue='species')              | plot x vs y and add a regression line with conditionals                                                           |
| sns.residplot(iris['petal_length'], y, lowess=True, color="g")                       | Residual plot (between input and predicted variable)                                                              |
| sns.scatterplot(x=iris['petal_length'], y=iris['petal_width'],hue=iris['species'])   | scatterplot between x and y colored by z, in pyplot, the variable values have to be mapped to numbers first       |
| sns.distplot(cars['price_usd'])                                                      | Density distribution plot                                                                                         |
| g = sns.PairGrid(df) g.map(plt.scatter);                                             | Same as pair plot but with more options                                                                           |
| g = sns.FacetGrid(df, col="fbs", margin_titles=True) g.map(sns.distplot, "age");     | grid of plots based on the values of col                                                                          |
| sns.countplot(x="target", data=df, hue="col")                                        | counts by value plot, grouped by col (great for categorical and ordinal variable variables with a few values)     |
| sns.heatmap(titanic_orig.isnull(), yticklabels = False, cmap='plasma')               | heatmap of correlations without y axis labels but with the colormap plasma                                        |
| sns.stripplot(y = 'Survived', x = 'Age', data = train)                               | scatterplot with 1 categorical variable.                                                                          |
| sns.violinplot(x='Embarked',y="Fare", data=train,inner=None,order = ['C', 'Q', 'S']) | Draw a combination of boxplot and kernel density estimate.                                                        |
| sns.catplot(x="Embarked", y="Fare", kind="violin", inner=None,data=train)            | Figure-level interface for drawing categorical plots onto a FacetGrid (1+ categorical)                            |
| sns.catplot(x="Pclass", y="Fare", kind="swarm", data=train, height = 6)              | Swarm plot with facetgrid                                                                                         |
| sns.swarmplot(x="Pclass", y="Fare", data=train,hue = "Survived")                     | Draw a categorical scatterplot with non-overlapping points and hue to use different colors based on on 'survived' |
| train['Fare'].nlargest(10).plot(kind='bar', title = '10 largest Fare')               | 10 largest values of fare                                                                                         |
| train['Age'].nsmallest(10).plot(kind='bar', color = ['#A83731','#AF6E4D'])           | 10 smallest values of age.                                                                                        |
| sns.catplot(x=col, y='Survived', data=conti, kind='point', aspect=3,)                | Point estimates and confidence intervals using scatter plot glyphs                                                |

```python
# Draw a box plot to show Age distributions with respect to survival status.
sns.boxplot(y = 'Survived', x = 'Age', data = train,
     palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')

# Add a scatterplot for each category.
sns.stripplot(y = 'Survived', x = 'Age', data = train,
     linewidth = 0.6, palette=["#3f3e6fd1", "#85c6a9"], orient = 'h')
```

```python
corr = train.corr()
sns.heatmap(corr[((corr >= 0.3) | (corr <= -0.3)) & (corr != 1)], annot=True, linewidths=.5, fmt= '.2f')
```

```python
# Ratios by deciles for continuous feature
def plotClassificationByQuartiles(data, feature, target, cuts=10):

    df = data.copy()
    df['rank'] = pd.qcut(data[feature], cuts,duplicates='drop')

    fig = plt.figure(figsize = (30,4))

    ax1 = fig.add_subplot(121)
    ax = sns.countplot(df['rank'], ax = ax1)

    # calculate passengers for each category
    labels = (df['rank'].value_counts())
    # add result numbers on barchart
    for i, v in enumerate(labels):
        ax.text(i, v+6, str(v), horizontalalignment = 'center', size = 10, color = 'black')

    plt.title('Distribution by oldpeak deciles')
    plt.ylabel('Number of People')
    plt.ylabel('oldpeak deciles')

    ax2 = fig.add_subplot(122)
    d = df.groupby('rank')[target].value_counts(normalize = True).unstack()
    d.plot(kind='bar', color=["#3f3e6fd1", "#85c6a9"], stacked='True', ax = ax2)
    plt.title('Proportion of target 0/1')
    plt.legend(( 'Target = 0', 'Target = 1'), loc=(1.04,0))
    plt.xticks(rotation = False)

    plt.tight_layout()
    plt.plot()
```
