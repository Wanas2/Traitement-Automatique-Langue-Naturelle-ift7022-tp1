# -*- coding: utf-8 -*-
import json

from nltk.stem.snowball import EnglishStemmer

import numpy as np
# import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import spacy

reviews_dataset = {
    'train_pos_fn' : "./data/senti_train_positive.txt",
    'train_neg_fn' : "./data/senti_train_negative.txt",
    'test_pos_fn' : "./data/senti_test_positive.txt",
    'test_neg_fn' : "./data/senti_test_negative.txt"
}


def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list


def tag_data(dataset, s):
    c = {"train_pos_fn": '+', "train_neg_fn": '-', "test_pos_fn": '+', "test_neg_fn": '-'}
    
    res = list()
    for x in load_reviews(dataset[s]):
        res.append([x, c[s]])
        
    return res


def tokenize(corpus):
    analyzer_en = spacy.load("en_core_web_sm")
    return analyzer_en(corpus)
    

def stem_normalization(X):
    stemmer = EnglishStemmer()

    for idx, corpus in enumerate(X):
        stemmed_corpus = ""
        
        for word in tokenize(str(corpus)):
            stemmed_corpus += " {}".format(stemmer.stem(word.text))

        X[idx] = stemmed_corpus


def lemma_normalization(X):
    for idx, corpus in enumerate(X):
        lemmatized_corpus = ""
            
        for word in tokenize(str(corpus)):
            lemmatized_corpus += " {}".format(word.lemma_)

        X[idx] = lemmatized_corpus


def train_and_test_classifier(dataset, model='NB', normalization='words'):
    """
    :param dataset: un dictionnaire contenant le nom des 4 fichiers utilisées pour entraîner et tester les classificateurs. Voir variable reviews_dataset.
    :param model: le type de classificateur. NB = Naive Bayes, LR = Régression logistique.
    :param normalization: le prétraitement appliqué aux mots des critiques (reviews)
                 - 'words': les mots des textes sans normalization.
                 - 'stem': les racines des mots obtenues par stemming.
                 - 'lemma': les lemmes des mots obtenus par lemmatisation.
    :return: un dictionnaire contenant 3 valeurs:
                 - l'accuracy à l'entraînement (validation croisée)
                 - l'accuracy sur le jeu de test
                 - la matrice de confusion calculée par scikit-learn sur les données de test
    """

    # Votre code...
    print("Set train and test dataset.")
    train_data = np.array(tag_data(dataset, 'train_pos_fn') + tag_data(dataset, 'train_neg_fn'))
    test_data = np.array(tag_data(dataset, 'test_pos_fn') + tag_data(dataset, 'test_neg_fn'))

    np.random.seed(777)

    print("Shuffle dataset")
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    print("Split X_train, y_train_ X_test, y_test")
    X_train, y_train = train_data[:, 0], train_data[:, 1]
    X_test, y_test = test_data[:, 0], test_data[:, 1]

    print("Normalization with {}...".format(normalization))
    print("This operation may take somme time to finish!")
    
    if normalization == 'words':
        pass
    elif normalization == 'stem':
        stem_normalization(X_train)
        stem_normalization(X_test)
        
    elif normalization == 'lemma':
        lemma_normalization(X_train)
        lemma_normalization(X_test)
    else: 
        raise ValueError("normalization should be 'words', 'stem' or 'lemma'")

    print("One hot enconding")
    vectorizer = CountVectorizer(lowercase=True)
    vectorizer.fit(X_train)

    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("Clf = {}".format(model))
    if model == 'NB':
        clf = MultinomialNB()
    elif model == 'LR':
        clf = LogisticRegression(max_iter=200)
    else: 
        raise ValueError("model should be NB or LR")

    print("Fit model")
    clf.fit(X_train_vectorized, y_train)
    
    print("Predict train and test")
    y_train_pred = clf.predict(X_train_vectorized)
    y__test_pred = clf.predict(X_test_vectorized)

    # Les résultats à retourner 
    results = dict()
    results['accuracy_train'] = accuracy_score(y_train, y_train_pred)
    results['accuracy_test'] = accuracy_score(y_test, y__test_pred)
    results['confusion_matrix'] = confusion_matrix(y_test, y__test_pred)
    return results


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme vous le souhaitez.
    # Contenu des fichiers de données
    splits = ['train_pos_fn', 'train_neg_fn', 'test_pos_fn', 'test_neg_fn']
    print("Taille des partitions du jeu de données")
    partitions = dict()
    for split in splits:
        partitions[split] = load_reviews(reviews_dataset[split])
        print("\t{} : {}".format(split, len(partitions[split])))

    # Entraînement et évaluation des modèles
    results = train_and_test_classifier(reviews_dataset, model='LR', normalization='lemma')
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: ", results['confusion_matrix'])

