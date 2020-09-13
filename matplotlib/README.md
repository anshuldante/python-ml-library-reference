# Matplotlib References

| Method                                                 | Description                                    |
| ------------------------------------------------------ | ---------------------------------------------- |
| plt.plot(df.x, df.y, color="grey")                     | define basic plot params                       |
| plt.xlabel('x')                                        | label for x-axis                               |
| plt.ylabel('y')                                        | label for y-axis                               |
| plt.grid()                                             | display a grid in the plot                     |
| plt.axhline(y, xmin, xmax)                             | add horizontal line                            |
| plt.axvline(x, ymin, ymax)                             | add vertical line                              |
| plt.show()                                             | display all plots                              |
| plt.annotate('text',(1.333, 0))                        | annotate the point x,y in the plot             |
| plt.xticks(range(0,11, 1))/plt.yticks(range(0, 22, 1)) | add labels on the x/y axis.                    |
| plt.quiver(x, y, xg, yg, width = l, units = 'dots')    | A quiver plot                                  |
| plt.fill_between(section,f(section), color='orange')   | fill the selected area with the selected color |
| df_counts.father.hist(title='Father Heights', bins=30) | histogram with 30 bins                         |
| df['col'].corr(df['col2'])                             | correlation between col and col2               |
| tight_layout                                           | Clean the layout                               |
| plt.figure(figsize=(15,5))                             | Set the figure size                            |

```python
# Subplots
fig, ax = plt.subplots()
plt.plot(x,y, color='purple')
ix = np.linspace(0, 3)
verts = [(0, 0)] + list(zip(ix, g(ix))) + [(3, 0)]
poly = Polygon(verts, facecolor='orange')
ax.add_patch(poly)
```

```python
df = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic'],
                   'Salary':[50000,54000,50000,189000,55000,40000,59000]})

salary = df['Salary']
density = stats.gaussian_kde(salary)
n, x, _ = plt.hist(salary, histtype='step', density=True, bins=25)  
plt.plot(x, density(x)*5)
plt.axvline(salary.mean(), color='magenta', linestyle='dashed', linewidth=2)
plt.axvline(salary.median(), color='green', linestyle='dashed', linewidth=2)
plt.show()
```

```python
# Create a 3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(df['horsepower'], df['engine-size'], df['price'],c=colors, marker=m)
```
