# Notes on coding in Python for Data Science
## Multiple Linear Regression
```py
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split # sklearn.cross_validation is depreciated
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
```

## SVR
```py
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)"""

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Plot Title (Regression Model)')
plt.xlabel('X level')
plt.ylabel('Label')
plt.show()
```

## Decision Trees
```py
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result
y_pred = regressor.predict(6.5)
```

## Random Forest
```py
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

```

### Visualising Random Forest Feature Importances

```py
feat_imp = pd.concat([pd.Series(rfr.feature_importances_), pd.Series(X.columns)], axis=1)
plt.rcdefaults()
fig, ax = plt.subplots()
y_labs =feat_imp[1]
y_pos = np.arange(len(y_labs))
ax.barh(y_pos, feat_imp[0], color='green')
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labs)
ax.set_yticklabels(['LABELS'])
ax.invert_yaxis()  # labels read top-to-bottom\n",
ax.set_xlabel('Relative Importance')
ax.set_title('Feature Importance Plot')
plt.savefig('pltName.jpg')
plt.show()
```


# SQL Server Connection

```py
from subprocess import call

# BCP Tableau Inputs into SQL Server
server = 'ServerURL'
user = 'username'
password = 'password'
database = 'database'

table_name = "table"
call('bcp {t} in {f} -c -S {s} -U {u} -P {p} -d {db} -t "{sep}" -r "{nl}" -e {e} -F 2'
         .format(t=table_name, f='{T}'.format(T=table), s=server, u=user, p=password,
                 db=database, sep='\t', nl='\n', e='Docs/bcp_fails/fails_{Y}.txt'.format(Y=table.split('.')[0])))

```

# Database Statements

```py
import sqlalchemy

engine = sqlalchemy.create_engine(
    'mssql+pyodbc://username:password@ServerURL/database?driver=SQL Server')

connection = engine.raw_connection()
cursor = connection.cursor()

sql = '''if (object_id('%s') is not null)
            ALTER TABLE %s
            ALTER COLUMN %s varchar(500)''' % (table, table, col)
            cursor.execute(sql)
cursor.commit()
connection.close()
```
