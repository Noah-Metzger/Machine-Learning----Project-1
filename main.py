from preprocess import *

if __name__ == '__main__':
    breastCancer = pd.read_csv("Data/breast-cancer-wisconsin.csv", header=None)
    glass = pd.read_csv('Data/glass.csv', header=None)
    houseVotes = pd.read_csv('Data/house-votes-84.csv', header=None)
    iris = pd.read_csv('Data/iris.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)