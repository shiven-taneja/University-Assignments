import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def load_data():
    fake_lines = open('clean_fake.txt').readlines()
    real_lines = open('clean_real.txt').readlines()
    vectorizer = CountVectorizer()
    combined = fake_lines + real_lines
    combined_vector = vectorizer.fit_transform(combined)
    labels = []
    i = 0
    while i < 1298:
        labels.append('fake')
        i = i+1
    i = 0
    while i < 1968:
        labels.append('real')
        i = i+1
    train_interim, test, train_interim_y, test_y = train_test_split(
        combined_vector, labels, test_size=0.15)
    train, val, train_y, val_y = train_test_split(train_interim, train_interim_y
                                                  , test_size=0.17647058823)
    return train, test, val, train_y, test_y, val_y, vectorizer


def select_model(train, val, train_y, val_y, vectorizer):
    gini_model_2 = tree.DecisionTreeClassifier(criterion='gini', max_depth=2)
    gini_model_3 = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
    gini_model_5 = tree.DecisionTreeClassifier(criterion='gini', max_depth=5)
    gini_model_7 = tree.DecisionTreeClassifier(criterion='gini', max_depth=7)
    gini_model_10 = tree.DecisionTreeClassifier(criterion='gini', max_depth=10)

    ig_model_2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
    ig_model_3 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    ig_model_5 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
    ig_model_7 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
    ig_model_10 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)

    gini_model_2.fit(train, train_y)
    gini_model_3.fit(train, train_y)
    gini_model_5.fit(train, train_y)
    gini_model_7.fit(train, train_y)
    gini_model_10.fit(train, train_y)

    ig_model_2.fit(train, train_y)
    ig_model_3.fit(train, train_y)
    ig_model_5.fit(train, train_y)
    ig_model_7.fit(train, train_y)
    ig_model_10.fit(train, train_y)

    actual = val_y
    predict = gini_model_2.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Gini Coefficient and max_depth = 2 is:',
          correct_count/count)

    predict = gini_model_3.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Gini Coefficient and max_depth = 3 is:',
          correct_count/count)

    predict = gini_model_5.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Gini Coefficient and max_depth = 5 is:',
          correct_count/count)

    predict = gini_model_7.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Gini Coefficient and max_depth = 7 is:',
          correct_count/count)

    predict = gini_model_10.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Gini Coefficient and max_depth = 10 is:',
          correct_count/count)

    predict = ig_model_2.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Information Gain and max_depth = 2 is:',
          correct_count/count)

    predict = ig_model_3.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Information Gain and max_depth = 3 is:',
          correct_count/count)

    predict = ig_model_5.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Information Gain and max_depth = 5 is:',
          correct_count/count)

    predict = ig_model_7.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Information Gain and max_depth = 7 is:',
          correct_count/count)

    predict = ig_model_10.predict(val)
    count = 0
    correct_count = 0
    i = 0
    while i < len(actual):
        if actual[i] == predict[i]:
            correct_count = correct_count + 1
        count = count + 1
        i = i + 1
    print('The accuracy with Information Gain and max_depth = 10 is:',
          correct_count/count)

    return ig_model_10


def compute_information_gain(vectorizer, train, train_y, keyword):
    entropy = calculate_entropy(train_y)
    keyword_index = vectorizer.vocabulary_[keyword]
    train_left = []
    train_y_left = []
    train_right = []
    train_y_right = []
    train = train.toarray()
    for i in range(0, len(train)):
        if train[i][keyword_index] < 0.5:
            train_left.append(train[i])
            train_y_left.append(train_y[i])
        else:
            train_right.append(train[i])
            train_y_right.append((train_y[i]))
    entropy_left = calculate_entropy(train_y_left)
    entropy_right = calculate_entropy(train_y_right)
    information_gain = entropy - ((float(len(train_y_left))/len(train_y))*
                                  entropy_left + (float(len(train_y_right))/len(train_y)
                                                  )*entropy_right)
    return information_gain


def calculate_entropy(train_y):
    num_real = 0
    num_fake = 0
    total = 0
    for i in train_y:
        if i == 'fake':
            num_fake += 1
        elif i == 'real':
            num_real += 1
    total = num_fake + num_real
    entropy = 0
    p1 = float(num_real)/total
    p2 = float(num_fake)/total
    entropy -= p1*(math.log2(p1)) + p2*(math.log2(p2))
    return entropy


if __name__ == '__main__':
    train, test, val, train_y, test_y, val_y, vectorizer= load_data()
    ig_model_10 = select_model(train, val, train_y, val_y, vectorizer)
    fig = plt.figure(figsize=(4, 4), dpi=300)
    tree.plot_tree(ig_model_10, max_depth=2, feature_names=vectorizer.
                   get_feature_names(), class_names=['false', 'true'])
    fig.savefig('tree_visualization.png',bbox_inches='tight', dpi=300)
    ig_words = ['the', 'donald', 'trumps', 'le', 'it', 'market']
    for word in ig_words:
        information_gain = compute_information_gain(vectorizer, train, train_y,
                                                    word)
        print('the Information Gain for the word', word, 'is:', information_gain)



