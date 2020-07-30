import numpy as np

class GaussianNBClassifier:

    def __init__(self):
        pass

    def separate_classes(self, X, y):
        """
        Separates the dataset in a subset of data for each class.

        Parameters:
        ----------
        X: list of features
        y: target variable

        Returns:
        dictionnary with y as keys, and the assigned X as values
        """
        separated_classes = {}
        for i in range(len(X)):
            feature_values = X[i]
            class_name = y[i]
            if class_name not in separated_classes:
                separated_classes[class_name] = []
            separated_classes[class_name].append(feature_values)
        return separated_classes
    
    def summarize(self, X):
        """
        Creates a sequence of mean and standard deviation for each column of X.
        Parameters:
        ----------
        X: array-like, dataset
        """
        for feature in zip(*X):
            yield {
                'stdev' : np.std(feature),
                'mean' : np.mean(feature)
            }
            
    def fit(self, X, y):
        """
        Trains the model.
        Parameters:
        ----------
        X: training features
        y: target variable

        Returns:
        Dictionary with the prior probability, mean, and standard deviation of each class
        """
        separated_classes = self.separate_classes(X, y)
        self.class_summary = {}
        for class_name, feature_values in separated_classes.items():
            self.class_summary[class_name] = {
                'prior_proba': len(feature_values)/len(X),
                'summary': [i for i in self.summarize(feature_values)],
            }     
        return self.class_summary
    
    def gauss_distribution_function(self, x, mean, stdev):
        """
        Gaussian Distribution Function(GDF)
        Parameters:
        ----------
        x: value of feature
        mean:  the average value of feature
        stdev: the standard deviation of feature
        Returns:
        A value of Normal Probability
        """
        exponent = np.exp(-((x-mean)**2 / (2*stdev**2)))
        return exponent / (np.sqrt(2*np.pi)*stdev)
    
    def predict(self, X):
        """
        Predicts the class.
        Parameters:
        ----------
        X: test data set

        Returns:
        List of predicted class for each row of data set
        """
        MAPs = []
        for row in X:
            joint_proba = {}
            for class_name, features in self.class_summary.items():
                total_features = len(features['summary'])
                likelihood = 1
                for idx in range(total_features):
                    feature = row[idx]
                    mean = features['summary'][idx]['mean']
                    stdev = features['summary'][idx]['stdev']
                    normal_proba = self.gauss_distribution_function(feature, mean, stdev)
                    likelihood *= normal_proba
                prior_proba = features['prior_proba']
                joint_proba[class_name] = prior_proba * likelihood
            MAP = max(joint_proba, key=joint_proba.get)
            MAPs.append(MAP)
        return MAPs
    
    def accuracy(self, y_test, y_pred):
        """
        Calculates model's accuracy.

        Parameters:
        y_test: actual values
        y_pred: predicted values
        
        Returns:
        A number between 0-1, representing the percentage of correct predictions.
        """
        true_true = 0
        for y_t, y_p in zip(y_test, y_pred):
            if y_t == y_p:
                true_true += 1
        return true_true / len(y_test)
