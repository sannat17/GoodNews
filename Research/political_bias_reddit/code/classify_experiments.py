import argparse
import os
from scipy.stats import ttest_ind
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

from tqdm import tqdm

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

VERBOSE = False

# Store the classifiers and their arguments as key value pairs
CLASSIFIERS = [
    (SGDClassifier, {}),
    (GaussianNB, {}),
    (RandomForestClassifier, {"max_depth": 5, "n_estimators": 10}),
    (MLPClassifier, {"max_iter": 1000}),
    (AdaBoostClassifier, {})
]

def accuracy(C):
    """ Compute accuracy given NumPy array confusion matrix C. Returns a floating point value. """
    return np.trace(C, dtype="float") / np.sum(C, dtype="float")


def recall(C):
    """ Compute recall given NumPy array confusion matrix C. Returns a list of floating point values. """
    return np.diag(C) / np.sum(C, axis=1, dtype="float")


def precision(C):
    """ Compute precision given NumPy array confusion matrix C. Returns a list of floating point values. """
    return np.diag(C) / np.sum(C, axis=0, dtype="float")
    

def classify_1(output_dir, X_train, y_train, X_test, y_test):
    """ 
    Experiment 1: Classify the data with all features.
    
    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.

    Returns:      
    - best_index: int, the index of the supposed best classifier.
    """
    
    accuracies = []

    with open(f"{output_dir}/exp1.txt", "w") as outf:

        pbar = tqdm(enumerate(CLASSIFIERS), desc="Classifiers exp 1") if VERBOSE else enumerate(CLASSIFIERS)

        for i, (classifier, model_args) in pbar:
            pbar.set_postfix_str(f"Evaluating: {classifier.__name__}") if VERBOSE else None

            model = classifier(**model_args)
            y_pred = model.fit(X_train, y_train).predict(X_test)

            conf_matrix = confusion_matrix(y_test, y_pred)
            acc = accuracy(conf_matrix)
            recall_test = recall(conf_matrix)
            precision_test = precision(conf_matrix)

            accuracies.append(acc)

            outf.write(f'Results for {classifier.__name__}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall_test]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision_test]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

            print(f"Classifiers exp 1 | Done: {classifier.__name__}") if VERBOSE else None
        
        pbar.close() if VERBOSE else None

    return np.argmax(accuracies)


def classify_2(output_dir, X_train, y_train, X_test, y_test, best_index):
    """ 
    Experiment 2: Classify the data with all features, after scaling.
    
    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index:   int, the index of the supposed best classifier (from classify_1).

    Returns:
       X_1k: NumPy array, just 1K rows of X_train.
       y_1k: NumPy array, just 1K rows of y_train.
    """
    
    classifier, model_args = CLASSIFIERS[best_index]
    data_volumes = [1, 5, 10, 15, 20]
    data_volumes = [d * 1000 for d in data_volumes]

    with open(f"{output_dir}/exp2.txt", "w") as outf:

        pbar = tqdm(data_volumes, desc="Classifiers exp 2") if VERBOSE else data_volumes

        for num_train in pbar:
            pbar.set_postfix_str(f"Evaluating training size: {num_train}") if VERBOSE else None

            X_subsample, _, y_subsample, _ = train_test_split(X_train, y_train, train_size=num_train)

            model = classifier(**model_args)
            y_pred = model.fit(X_subsample, y_subsample).predict(X_test)
            acc = accuracy(confusion_matrix(y_test, y_pred))
            
            outf.write(f'{num_train}: {acc:.4f}\n')

            if num_train == 1000:
                X_1k, y_1k = X_subsample, y_subsample

            print(f"Classifiers exp 2 | Finished training size: {num_train}") if VERBOSE else None

    return (X_1k, y_1k)


def classify_3(output_dir, X_train, y_train, X_test, y_test, best_index, X_1k, y_1k):
    """ 
    Experiment 3: Classify the data after feature selection.
    
    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index: int, the index of the supposed best classifier (from classify_1).
    - X_1k:    NumPy array, just 1K rows of X_train (from classify_2).
    - y_1k:    NumPy array, just 1K rows of y_train (from classify_2).
    """

    classifier, model_args = CLASSIFIERS[best_index]

    with open(f"{output_dir}/exp3.txt", "w") as outf:
        
        for k_feat in [5, 50]:
            selector = SelectKBest(f_classif, k=k_feat)
            selector.fit(X_train, y_train)

            k_feat_mask = selector.get_support()
            p_values = selector.pvalues_[k_feat_mask]

            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        
        selector = SelectKBest(f_classif, k=5)
        
        # Train on 1k
        selector.fit(X_1k, y_1k)
        features_1k = selector.get_support(indices=True)
        X_1k_new, X_test_1k_new = selector.transform(X_1k), selector.transform(X_test)
        y_1k_pred =  classifier(**model_args).fit(X_1k_new, y_1k).predict(X_test_1k_new)
        accuracy_1k = accuracy(confusion_matrix(y_test, y_1k_pred))
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')

        # Train on full
        selector.fit(X_train, y_train)
        features_full = selector.get_support(indices=True)
        X_train_new, X_test_new = selector.transform(X_train), selector.transform(X_test)
        y_full_pred =  classifier(**model_args).fit(X_train_new, y_train).predict(X_test_new)
        accuracy_full = accuracy(confusion_matrix(y_test, y_full_pred))
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')

        # Feature intersection
        feature_intersection = np.intersect1d(features_1k, features_full)
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')

        # Top-5 at higher
        top_5 = features_full
        outf.write(f'Top-5 at higher: {top_5}\n')
    
    return


def classify_4(output_dir, X_train, y_train, X_test, y_test, best_index):
    """ 
    Experiment 4: Classify the data with the best classifier, after feature selection.
    
    Parameters:
    - output_dir:   path of directory to write output to.
    - X_train:  NumPy array, with the selected training features.
    - y_train:  NumPy array, with the selected training classes.
    - X_test:   NumPy array, with the selected testing features.
    - y_test:   NumPy array, with the selected testing classes.
    - best_index:   int, the index of the best classifier from classify_1.
    """
    
    X_full = np.concatenate((X_train, X_test))
    y_full = np.concatenate((y_train, y_test))

    with open(f"{output_dir}/exp4.txt", "w") as outf:
        
        kf = KFold(n_splits=5, shuffle=True)
        classifier_accuracies = [[] for _ in range(len(CLASSIFIERS))]

        split_pbar = tqdm(enumerate(kf.split(X_full)), desc="Classifiers exp 4") if VERBOSE else enumerate(kf.split(X_full))
        for i, (train_index, test_index) in split_pbar:
            split_pbar.set_postfix_str(f"Evaluating split: {i}") if VERBOSE else None

            X_split_train, X_split_test = X_full[train_index], X_full[test_index]
            y_split_train, y_split_test = y_full[train_index], y_full[test_index]

            kfold_accuracies = []
            classifier_pbar = tqdm(enumerate(CLASSIFIERS), desc="Classifiers exo=p 4") if VERBOSE else enumerate(CLASSIFIERS)
            for j, (classifier, model_args) in classifier_pbar:
                classifier_pbar.set_postfix_str(f"Evaluating: {classifier.__name__}") if VERBOSE else None

                model = classifier(**model_args)
                y_pred = model.fit(X_split_train, y_split_train).predict(X_split_test)
                acc = accuracy(confusion_matrix(y_split_test, y_pred))

                classifier_accuracies[j].append(acc)
                kfold_accuracies.append(acc)
                
                print("Classifiers exp 4 | Done: Split", i, classifier.__name__) if VERBOSE else None

            classifier_pbar.close() if VERBOSE else None


            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')

        split_pbar.close() if VERBOSE else None
        
        # Perform t-tests
        p_values = []
        for i in range(len(CLASSIFIERS)):
            if i != best_index:
                _, p_val = ttest_ind(classifier_accuracies[best_index], classifier_accuracies[i])
                p_values.append(p_val)
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The input npz file from extract_features", required=True)
    parser.add_argument(
        "-o", "--output-dir",
        help="The directory to write output exp*.txt files from the experiments.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # Create output dir if it is new
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with np.load(args.input) as feats_file:
        feats = feats_file['arr_0'][:, :173]
        labels = feats_file['arr_0'][:, 173]
    # Creating train test split
    X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=0.2)

    best_index_31 = classify_1(args.output_dir, X_train, y_train, X_test, y_test)
    X_1k, y_1k = classify_2(args.output_dir, X_train, y_train, X_test, y_test, best_index_31)
    classify_3(args.output_dir, X_train, y_train, X_test, y_test, best_index_31, X_1k, y_1k)
    classify_4(args.output_dir, X_train, y_train, X_test, y_test, best_index_31)