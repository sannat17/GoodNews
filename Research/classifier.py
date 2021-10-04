from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from math import log2
import numpy as np

def load_data():

    # Load cleaned text data
    with open("real_data.txt", "r") as readfile:
        clean_real = readfile.read().split("\n")
    with open("fake_data.txt", "r") as readfile:
        clean_fake = readfile.read().split("\n")
    
    # Preprocessing
    all_data = clean_real + clean_fake
    y = [1 for i in range(len(clean_real))] + [0 for i in range(len(clean_fake))]

    vectorizer = CountVectorizer().fit(all_data)
    x = vectorizer.transform(all_data)

    # 7:3 :: training:testing split (from all data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
    # 1:1 :: validation:testing split (from testing) => 7:1.5:1.5 :: training:validation:testing split
    x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=100)


    return x_train, x_validate, x_test, y_train, y_validate, y_test, vectorizer


def select_model() -> DecisionTreeClassifier:
    """ Tests and compares decision tree classifiers with combination of 
    different max_depth values from the and split criteria (information gain and
    Gini coefficient). Prints the results of each model with their validation
    accuracy and then which one is the best.

    Returns
    -------
    model : DecisionTreeClassifier 
        A trained DecisionTreeClassifier with the hyperparameters that had the
        best validation accuracy.
    """

    x_train, x_validate, x_test, y_train, y_validate, y_test, vectorizer = load_data()
    best = {"max_depth": 0, "criterion": "gini", "accuracy": 0, "classifier": None}

    for max_depth in [15, 30, 45, 60, 75]:
        
        # Model with gini criterion
        DTC_gini = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=100)
        DTC_gini.fit(x_train, y_train)
        gini_score = DTC_gini.score(x_validate, y_validate)
        print(f"Split criteria: gini -- max_depth: {max_depth} -- Validation accuracy: {round(gini_score, 3)}")
        ## Check if it is best
        if gini_score > best["accuracy"]:
            best["max_depth"] = max_depth
            best["criterion"] = "gini"
            best["accuracy"] = gini_score
            best["classifier"] = DTC_gini

        # Model with information gain criterion
        DTC_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=100)
        DTC_entropy.fit(x_train, y_train)
        entropy_score = DTC_entropy.score(x_validate, y_validate)
        print(f"Split criteria: information gain -- max_depth: {max_depth} -- Validation accuracy: {round(entropy_score, 3)}")
        ## Check if it is best
        if entropy_score > best["accuracy"]:
            best["max_depth"] = max_depth
            best["criterion"] = "information gain"
            best["accuracy"] = entropy_score
            best["classifier"] = DTC_entropy
    
    # Report the model with hyperparameters tuned for best validation accuracy
    test_accuracy = best["classifier"].score(x_test, y_test)
    print(f"===== The best results are with: =====")
    print("Split criteria: '{}' -- max_depth: '{}' -- Test Accuracy:'{}'".
            format(best['criterion'], best['max_depth'], round(test_accuracy, 3))
            )

    return best["classifier"]

def compute_information_gain(x_train, y_train, vectorizer, keyword):

    #  Finding the index of the keyword in the vectorizer's vocab
    word_index = vectorizer.vocabulary_[keyword]

    # Finding Entropy of y_train
    p_real = (np.count_nonzero(y_train)/len(y_train))
    entropy = -1 * (p_real*log2(p_real) + (1 - p_real)*log2(1 - p_real))

    # Finding conditional entropy of y_train given keyword
    keyword_list = [x for x in list(zip(x_train.toarray(), y_train)) if x[0][word_index] > 0]
    real_list = [x for x in keyword_list if x[1] == 1]
    p_real = (len(real_list)/len(keyword_list))
    conditional_entropy = -1 * (p_real*log2(p_real) + (1-p_real)*log2(1-p_real))

    return entropy - conditional_entropy

if __name__ == "__main__":
    x_train, x_validate, x_test, y_train, y_validate, y_test, vectorizer = load_data()
    clf = select_model()
    tree.export_graphviz(clf, "tree.dot", max_depth=2,  
        class_names=["fake", "real"], filled=True, rounded=True,
        feature_names=vectorizer.get_feature_names()
    )

    compute_information_gain(x_train, y_train, vectorizer, "donald")
    compute_information_gain(x_train, y_train, vectorizer, "the")
    compute_information_gain(x_train, y_train, vectorizer, "victory")