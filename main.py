from preprocess import *
from evaluation import *
from execute import *
from NaiveBayesProfessional import *
import copy
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore")

def tune(tuning):
    """
    Prints the results of average precision and recall values for each bin. Prints out sorted list of values and the number of bins for each corresponding value.

    :param tuning: Python list of average of precison and recall values added together for each cross validated for each set with a different amount of bins.
    """
    index = list(range(0,len(tuning)))

    for i in range(len(tuning)):
        for j in range(0, len(tuning) - i - 1):
            if tuning[i] > tuning[j]:
                temp = tuning[i]
                tuning[i] = tuning[j]
                tuning[j] = temp

                tempIndex = index[i]
                index[i] = index[j]
                index[j] = tempIndex
    print(tuning)
    print(index)

def plot(result1, result2, name):
    """
    Outputs a scatter plot of results from two loss-functions

    :param result1: Python list of values to be plotted on the x-axis.
    :param result2: Python list of values to be plotted on the y-axis.
    :param name: Name of the dataset to be used as title for the plot
    """
    plt.scatter(result1, result2)
    plt.title(name + " Dataset")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

def experiment(preproArr):
    """
    The main driver for the experimentation of the classifier.  Runs classifier, k-fold, cross validation, evaluations, printing and plotting of results

    :param preproArr: A Python list of Preprocessor objects.  These do not necessarily need to have any preprocessing methods called before classification
    """
    # tuning = []
    for obj in preproArr:
        #Conducts classification with 10-fold cross-validation
        n = NaiveBayesClassifier()
        e = execute(obj.df)
        results = e.crossvalidate(n.driver, 10, obj.truthColIndex)

        x = []
        y = []
        #Prints out precision and recall for each fold
        print("***" + obj.dfName + "***")
        for fold in results:
            e = Evaluation(fold[0], fold[1], np.array(obj.truthCol))
            prec = e.precision()
            rec = e.recall()
            # e.printConfusionMatrix()
            for i in range(len(prec)):
                x.append(prec[i])
                y.append(rec[i])

        print()
        print("Average precision: " + str(sum(x) / len(x)))
        print("Average recall: " + str(sum(y) / len(y)))
        print()
        # tuning.append([(sum(x) / len(x)) + (sum(y) / len(y))])
    # tune(tuning)
    #Prints scatter plots
    # plot(np.array(x), np.array(y), obj.dfName)

if __name__ == '__main__':
    preProcessedArray = []
    breastCancer = pd.read_csv("Data/breast-cancer-wisconsin.csv", header=None)
    glass = pd.read_csv('Data/glass.csv', header=None)
    houseVotes = pd.read_csv('Data/house-votes-84.csv', header=None)
    iris = pd.read_csv('Data/iris.csv', header=None)
    soyBean = pd.read_csv('Data/soybean-small.csv', header=None)

    #Each dataset is put into a Preprocessor object before classification
    #A dataset is not preprocessed unless a method has been explicit called on the Preprocessor object.
    #The creation of a preprocessor object does not imply that the dataset has been modified in any way.

    # breastCancerNoNoise = Preprocessor(copy.copy(breastCancer), 10, "Breast Cancer Wisconsin")
    # breastCancerNoNoise.deleteFeature(0)
    # breastCancerNoNoise.removesmissingvalues()
    # preProcessedArray.append(breastCancerNoNoise)
    #
    # breastCancerNoise = Preprocessor(copy.copy(breastCancer), 10, "Breast Cancer Wisconsin - Noise Introduced")
    # breastCancerNoise.deleteFeature(0)
    # breastCancerNoise.removesmissingvalues()
    # breastCancerNoise.shuffle()
    # preProcessedArray.append(breastCancerNoise)

    glassNoNoise = Preprocessor(copy.copy(glass), 10, "Glass")
    glassNoNoise.deleteFeature(0)
    glassNoNoise.binning(list(range(1, 9)), 7)
    preProcessedArray.append(glassNoNoise)

    glassNoise = Preprocessor(copy.copy(glass), 10, "Glass - Noise Introduced")
    glassNoise.deleteFeature(0)
    glassNoise.binning(list(range(1, 9)), 7)
    glassNoise.shuffle()
    preProcessedArray.append(glassNoise)

    # houseVotesNoNoise = Preprocessor(copy.copy(houseVotes), 0, "House Votes 84")
    # preProcessedArray.append(houseVotesNoNoise)
    #
    # houseVotesNoise = Preprocessor(copy.copy(houseVotes), 0, "House Votes 84 - Noise Introduced")
    # houseVotesNoise.shuffle()
    # preProcessedArray.append(houseVotesNoise)

    irisNoNoise = Preprocessor(copy.copy(iris), 4, "Iris")
    irisNoNoise.binning([0, 1, 2, 3], 6)
    preProcessedArray.append(irisNoNoise)

    irisNoise = Preprocessor(copy.copy(iris), 4, "Iris - Noise Introduced")
    irisNoise.binning([0, 1, 2, 3], 6)
    irisNoise.shuffle()
    preProcessedArray.append(irisNoise)
    #
    # soyBeanNoNoise = Preprocessor(copy.copy(soyBean), 35, "Soy bean")
    # preProcessedArray.append(soyBeanNoNoise)
    #
    # soyBeanNoise = Preprocessor(copy.copy(soyBean), 35, "Soy bean - Noise Introduced")
    # soyBeanNoise.shuffle()
    # preProcessedArray.append(soyBeanNoise)

    experiment(preProcessedArray)