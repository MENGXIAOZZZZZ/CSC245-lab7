# CSC245-lab7

Question 0 (Example) How many features does the breast cancer dataset have? This function should return an integer.
import numpy  
import pandas 
def answer_one(): 
     
 data = numpy.c_[cancer.data,cancer.target] 
 columns = numpy.append(cancer.feature_names, ["target"]) 
     
 return pandas.DataFrame(data, columns=columns) 
 answer_one()


  Question 1 Scikit-learn works with lists, numpy arrays, scipy-sparse matrices, and pandas DataFrames, so converting the dataset to a DataFrame is not necessary for training this model. Using a DataFrame does however help make many things easier such as munging data, so let's practice creating a classifier with a pandas DataFrame.
import numpy  
import pandas 
def answer_one(): 
     
 data = numpy.c_[cancer.data,cancer.target] 
 columns = numpy.append(cancer.feature_names, ["target"]) 
     
 return pandas.DataFrame(data, columns=columns) 
 answer_one()


  Question 2 What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?) This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']
def answer_two(): 
    """calculates number of malignent and benign 
 
    Returns: 
     pandas.Series: counts of each 
    """ 
    cancerdf = answer_one() 
    counts = cancerdf.target.value_counts(ascending=True) 
    counts.index = "malignant benign".split() 
    return counts 
 
answer_two()

  Question 3 Split the DataFrame into X (the data) and y (the labels). This function should return a tuple of length 2: (X, y), where X, a pandas DataFrame, has shape (569, 30) y, a pandas Series, has shape (569,).
 def answer_three(): 
    """splits the data into data and labels 
 
    Returns: 
     (pandas.DataFrame, pandas.Series): data, labels 
    """ 
    cancerdf = answer_one() 
    X = cancerdf[cancerdf.columns[:-1]] 
    y = cancerdf.target 
    return X, y 
answer_three()
  

  Question 4 Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test)

from sklearn.model_selection import train_test_split 
 
def answer_four(): 
    """splits data into training and testing sets 
 
    Returns: 
     tuple(pandas.DataFrame): x_train, y_train, x_test, y_test 
    """ 
    X, y = answer_three() 
    return train_test_split(X, y, train_size=426, test_size=143, random_state=0) 
answer_four()


  Question 5 Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1). This function should return a sklearn.neighbors.classification.KNeighborsClassifier.

from sklearn.neighbors import KNeighborsClassifier 
 
def answer_five(): 
    """Fits a KNN-1 model to the data 
 
    Returns: 
     sklearn.neighbors.KNeighborsClassifier: trained data 
    """ 
    X_train, X_test, y_train, y_test = answer_four() 
    model = KNeighborsClassifier(n_neighbors=1) 
    model.fit(X_train, y_train) 
    return model 
answer_five()


  Question 6 Using your knn classifier, predict the class label using the mean value for each feature. Hint: You can use cancerdf.mean()[:-1].values.reshape(1, -1) which gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier). This function should return a numpy array either array([ 0.]) or array([ 1.])

def answer_six(): 
    """Predicts the class labels for the means of all features 
 
    Returns: 
     numpy.array: prediction (0 or 1) 
    """ 
    cancerdf = answer_one() 
    means = cancerdf.mean()[:-1].values.reshape(1, -1) 
    model = answer_five() 
    return model.predict(means) 
answer_six()


  Question 7 Using your knn classifier, predict the class labels for the test set X_test. This function should return a numpy array with shape (143,) and values either 0.0 or 1.0

def answer_seven(): 
    """predicts likelihood of cancer for test set 
 
    Returns: 
     numpy.array: vector of predictions 
    """ 
    X_train, X_test, y_train, y_test = answer_four() 
    knn = answer_five() 
    return knn.predict(X_test) 
answer_seven()


  Question 8 Find the score (mean accuracy) of your knn classifier using X_test and y_test. This function should return a float between 0 and 1

def answer_eight(): 
    """calculates the mean accuracy of the KNN model 
 
    Returns: 
     float: mean accuracy of the model predicting cancer 
    """ 
    X_train, X_test, y_train, y_test = answer_four() 
    knn = answer_five() 
    return knn.score(X_test, y_test) 
answer_eight()
