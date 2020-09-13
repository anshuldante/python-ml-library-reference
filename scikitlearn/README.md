# Sklearn Reference

| Method                                                                                          | Description                                                         |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| model = linear_model.LinearRegression()                                                         | linear regression                                                   |
| results = model.fit(X, y)                                                                       | fit a model                                                         |
| results.intercept_, results.coef_                                                               | y intercept and coefficients for Linear Regression model            |
| tree.plot_tree(clf)                                                                             | plots the tree                                                      |
| graphviz.Source(tree.export_graphviz(clf, out_file=None)).render("iris")                        | Render and save the tree as PDF using graphviz                      |
| X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)        | split the dataset into train-test                                   |
| metrics.mean_absolute_error(y_test, y_pred)                                                     | Mean absolute error                                                 |
| metrics.mean_squared_error(y_test, y_pred)                                                      | Mean squared error                                                  |
| np.sqrt(metrics.mean_squared_error(y_test, y_pred))                                             | Root mean squared error                                             |
| explained_variance_score(y_test, y_pred, multioutput='uniform_average')                         | explained variance score                                            |
| KNeighborsClassifier(k, weights=weights)                                                        | K-nearest neighbors                                                 |
| r2_score(df_japan['mpg'], f(df_japan['weight']))                                                | R-squared (regression score function)                               |
| model = linear_model.LinearRegression()                                                         | Basic Linear Regression model                                       |
| PCA(), pca.explained_variance_ratio_, components_                                               | Principal Component Analysis, explained variance ratios, components |
| DecisionTreeRegressor                                                                           | Decision Tree Regressor                                             |
| DecisionTreeClassifier                                                                          | Decision Tree Classifier                                            |
| tree_clf.score(X_test, y_test)                                                                  | Score for a classifier                                              |
| sklearn.neural_network.MLPClassifier                                                            | Multi Level Perceptron (neural net)                                 |
| sklearn.neighbors.KNeighborsClassifier                                                          | K-nearest neighbors classifier (supervised)                         |
| plt.plot(np.cumsum(pca.explained_variance_ratio_))                                              | Plot the number of component vs cumulative variance explained       |
| scores = cross_val_score(RandomForestClassifier(), tr_features, tr_labels.values.ravel(), cv=5) | k-fold cross validation                                             |
| accuracy = round(accuracy_score(val_labels, y_pred), 3)                                         | accuracy score                                                      |
| precision = round(precision_score(val_labels, y_pred), 3)                                       | precision score                                                     |
| recall = round(recall_score(val_labels, y_pred), 3)                                             | recall score                                                        |

```python
# Outlier removal techniques
ort = IsolationForest(contamination=0.1)
ort = EllipticEnvelope(contamination=0.01)
ort = LocalOutlierFactor()
ort = OneClassSVM()
yhat = ort.fit_predict(X_train)
X_train = X_train[yhat!=-1].reset_index(drop=True)
y_train = y_train[yhat!=-1].reset_index(drop=True)
```

```python
encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(train[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
train = train.join(temp)
train.drop(columns='Embarked', inplace=True)
```

```python
# GridSearchCV sample
clf2 = GridSearchCV(RandomForestClassifier(),params)
clf2.fit(X_train,y_train)
print(f'{clf2.best_estimator_}')
```

```python
# print results of a CV
def print_results(results):
    print(f'BEST PARAMS: {results.best_params_}\n')

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 100],
    'max_depth': [2, 10, 20, None]
}
cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)
```
