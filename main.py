from preprocess import *
from evaluation import *

if __name__ == '__main__':
    breastCancer = pd.read_csv("Data/breast-cancer-wisconsin.csv", header=None)
    glass = pd.read_csv('Data/glass.csv', header=None)
    houseVotes = pd.read_csv('Data/house-votes-84.csv', header=None)
    iris = pd.read_csv('Data/iris.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)

    #TESTING ONLY
    X = iris.iloc[:, 0:3].values
    Y = iris.iloc[:, 4]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    classifer1 = GaussianNB()
    classifer1.fit(X_train, y_train)
    y_pred1 = classifer1.predict(X_test)

    e = Evaluation(iris, y_pred1, y_test)
    e.precision()