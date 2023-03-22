# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from distutils.log import Log
import glob
import os
import string
import unicodedata
import json
import random

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
# le fichier contenant les données de test pour évaluer vos modèles
test_filename = './data/test_names.txt'

# un dictionnaire qui contient une liste de noms pour chaque langue d'origine
names_by_origin = {}
all_origins = []  # la liste des 18 langues d'origines de noms

# Fonctions utilitaires pour lire les données d'entraînement et de test - NE PAS MODIFIER


def load_names():
    """Lecture des noms et langues d'origine d'un fichier. Par la suite,
       sauvegarde des noms pour chaque origine dans le dictionnaire names_by_origin."""
    for filename in find_files(datafiles):
        origin = get_origin_from_filename(filename)
        all_origins.append(origin)
        names = read_names(filename)
        names_by_origin[origin] = names


def find_files(path):
    """Retourne le nom des fichiers contenus dans un répertoire.
       glob fait le matching du nom de fichier avec un pattern - par ex. *.txt"""
    return glob.glob(path)


def get_origin_from_filename(filename):
    """Passe-passe qui retourne la langue d'origine d'un nom de fichier.
       Par ex. cette fonction retourne Arabic pour "./data/names/Arabic.txt". """
    return os.path.splitext(os.path.basename(filename))[0]


def read_names(filename):
    """Retourne une liste de tous les noms contenus dans un fichier."""
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def unicode_to_ascii(s):
    """Convertion des caractères spéciaux en ascii. Par exemple, Hélène devient Helene.
       Tiré d'un exemple de Pytorch. """
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def load_test_names(filename):
    """Retourne un dictionnaire contenant les données à utiliser pour évaluer vos modèles.
       Le dictionnaire contient une liste de noms (valeurs) et leur origine (clé)."""
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data


# ---------------------------------------------------------------------------
# Fonctions à développer pour ce travail - Ne pas modifier les signatures et les valeurs de retour

vectorizers = [
    CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 1)),
    CountVectorizer(lowercase=True, analyzer='char', ngram_range=(2, 2)),
    CountVectorizer(lowercase=True, analyzer='char', ngram_range=(3, 3)),
    CountVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 3))
]

models = {"NB": [], "LR": []}
ngram_lengths = [1, 2, 3, 'multi']


def tag_data(raw_data):
    res = list()

    for origin, names in raw_data.items():
        for name in names:
            res.append([name, origin])

    return res


def train_classifiers():
    load_names()
    # Vous ajoutez à partir d'ici tout le code dont vous avez besoin
    # pour construire les différentes versions de classificateurs de langues d'origines.
    # Voir les consignes de l'énoncé du travail pratique pour déterminer les différents modèles à entraîner.
    #
    # On suppose que les données d'entraînement ont été lues (load_names) et sont disponibles (names_by_origin).
    #
    # Vous pouvez ajouter au fichier toutes les fonctions que vous jugerez nécessaire.
    # Assurez-vous de sauvegarder vos modèles pour y accéder avec la fonction get_classifier().
    # On veut éviter de les reconstruire à chaque appel de cette fonction.
    # Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    #
    print("\n======= Train classifiers")
    train_data = np.array(tag_data(names_by_origin))

    print("Shuffle data")
    random.seed(777)
    random.shuffle(train_data)

    print("Split data")
    X_train, y_train = train_data[:, 0], train_data[:, 1]

    vectorized_data = list()

    print("Fit vectorizers for ngram_lengths 1, 2, 3 and multi")
    for vectorizer in vectorizers:
        vectorizer.fit(X_train)
        vectorized_data.append(vectorizer.transform(X_train))

    print("Fit NB models")
    for X_vectorized in vectorized_data:
        nb_model = MultinomialNB()
        nb_model.fit(X_vectorized, y_train)

        models["NB"].append(nb_model)

    print("Fit LR models")
    for X_vectorized in vectorized_data:
        lr_model = LogisticRegression(max_iter=200)
        lr_model.fit(X_vectorized, y_train)

        models["LR"].append(lr_model)


def get_classifier(type, n=3):
    # Retourne le classificateur correspondant. On peut appeler cette fonction
    # après que les modèles ont été entraînés avec la fonction train_classifiers
    #
    # type = 'NB' pour naive bayes ou 'LR' pour régression logistique
    # n = 1,2,3 ou multi
    #
    if type != 'NB' and type != 'LR':
        raise ValueError("Unknown model type")

    if n not in ngram_lengths:
        raise ValueError("Unknow n")

    return models[type][ngram_lengths.index(n)]


def origin(name, type, n=3):
    # Retourne la langue d'origine prédite pour le nom.
    #   - name = le nom qu'on veut classifier
    #   - type = 'NB' pour naive bayes ou 'LR' pour régression logistique
    #   - n désigne la longueur des N-grammes de caractères. Choix possible = 1, 2, 3, 'multi'
    #

    clf = get_classifier(type, n)

    vectorizer = vectorizers[ngram_lengths.index(n)]
    name_origin = clf.predict(vectorizer.transform([name]))

    return name_origin


def evaluate_classifier(test_fn, type, n=3):
    test_data = load_test_names(test_fn)

    return predict_score(test_data, type, n)


def predict_score(data, type, n):
    true_pos_neg = 0
    total = 0
    for true_origin, names in data.items():
        for name in names:
            pred_origin = origin(name, type, n)[0]

            if true_origin == pred_origin:
                true_pos_neg += 1

            total += 1

    return true_pos_neg / total
    

if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    load_names()
    print("Les {} langues d'origine sont: \n{}".format(len(all_origins), all_origins))

    train_classifiers()

    test_names = load_test_names(test_filename)
    print("\nLes données pour tester vos modèles sont:")
    for org, name_list in test_names.items():
        print("\t{} : {}".format(org, name_list))
    
    for model in ['NB', 'LR']:
        for ngram_length in ngram_lengths:
            classifier = get_classifier(model, n=ngram_length)
            print("\nType de classificateur: ", classifier, ", ngram_length: ", ngram_length)

            train_evaluation = predict_score(names_by_origin, model, ngram_length)
            test_evaluation = evaluate_classifier(test_filename, model, n=ngram_length)
            
            print("\nTrain accuracy = {}\nTest accuracy = {}".format(train_evaluation, test_evaluation))

