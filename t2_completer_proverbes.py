import json
import string
from collections import Counter
from nltk import word_tokenize
from nltk.util import pad_sequence, ngrams
from nltk.lm import MLE
from nltk.lm.models import Laplace

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes.txt"

# Variables
models = {
    1: Laplace(1),
    2: Laplace(2),
    3: Laplace(3)
}


def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
    return test_data


# Function to remove punctuations and make lowercase
def clean_proverbs(native_proverbs):
    punctuations = string.punctuation + "’"
    cleaned_proverbs = []
    for proverb in native_proverbs:
        new_string = proverb
        for char in punctuations:
            if char in proverb:
                new_string = new_string.replace(char, ' ')
        new_string.lower()
        cleaned_proverbs.append(" ".join(new_string.split()))
    return cleaned_proverbs


# Function to get ngram.
def get_ngrams(sentence_list, n=1):
    left_tag = '<BOS>'
    right_tag = '<EOS>'
    all_ngrams = list()
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        padded_sent = list(pad_sequence(tokens,
                                        pad_left=True,
                                        left_pad_symbol=left_tag,
                                        pad_right=True, right_pad_symbol=right_tag, n=n))
        all_ngrams.extend(list(ngrams(padded_sent, n=n)))
    return all_ngrams


def ngram_from_sentence(sentence, n=1):
    left_tag = '<BOS>'
    right_tag = '<EOS>'
    tokens = word_tokenize(sentence)
    padded_sent = list(pad_sequence(tokens,
                                    pad_left=True,
                                    left_pad_symbol=left_tag,
                                    pad_right=True, right_pad_symbol=right_tag, n=n))
    return padded_sent


# Function to get the vocabulary from all proverbs
def get_vocabulary(sentence_list):
    start_tag = '<BOS>'
    end_tag = '<EOS>'
    all_tokens = []
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        all_tokens.extend(tokens)
    vocabulary = list(set(all_tokens))
    vocabulary.append(start_tag)
    vocabulary.append(end_tag)
    return vocabulary


# Function to clean proverb test
def clean_sentence(sentence, n=1):
    left_tag = '<BOS>'
    right_tag = '<EOS>'
    answer = sentence
    punctuations = string.punctuation + "’"
    for char in punctuations:
        if char != "*":
            answer = answer.replace(char, ' ')
    result = " ".join(answer.split())
    tokens = word_tokenize(result.lower())
    return list(pad_sequence(tokens,
                             pad_left=True,
                             left_pad_symbol=left_tag, pad_right=True, right_pad_symbol=right_tag, n=n))


def train_models(filename):
    proverbs = load_proverbs(filename)
    """ Vous ajoutez à partir d'ici tout le code dont vous avez besoin
        pour construire les différents modèles N-grammes.
        Voir les consignes de l'énoncé du travail pratique concernant les modèles à entraîner.

        Vous pouvez ajouter au fichier les classes, fonctions/méthodes et variables que vous jugerez nécessaire.
        Il faut au minimum prévoir une variable (par exemple un dictionnaire) 
        pour conserver les modèles de langue N-grammes après leur construction. 
        Merci de ne pas modifier les signatures (noms de fonctions et arguments) déjà présentes dans le fichier.
    """

    # Votre code à partir d'ici...

    cleaned_proverbs = clean_proverbs(proverbs)
    vocabulary = get_vocabulary(cleaned_proverbs)
    corpus_uni = get_ngrams(cleaned_proverbs)
    corpus_bigram = get_ngrams(cleaned_proverbs, 2)
    corpus_trigram = get_ngrams(cleaned_proverbs, 3)
    models.get(1).fit([corpus_uni], vocabulary_text=vocabulary)
    models.get(2).fit([corpus_bigram], vocabulary_text=vocabulary)
    models.get(3).fit([corpus_trigram], vocabulary_text=vocabulary)


def cloze_test(incomplete_proverb, choices, n=3, criteria="perplexity"):
    """ Fonction qui complète un texte à trous (des mots masqués) en ajoutant le bon mot.
        En anglais, on nomme ce type de tâche un "cloze test".

        Le paramètre criteria indique la mesure qu'on utilise pour choisir le mot le plus probable: "logprob" ou "perplexity".
        La valeur retournée est l'estimation sur le proverbe complet (c.-à-d. toute la séquence de mots du proverbe).

        Le paramètre n désigne le modèle utilisé.
        1 - unigramme NLTK, 2 - bigramme NLTK, 3 - trigramme NLTK
    """

    # Votre code à partir d'ici.Vous pouvez modifier comme bon vous semble.
    missing_word = 'atrouver'
    sentence = incomplete_proverb.replace('***', missing_word)
    test_sentence_token = clean_sentence(sentence, n)
    missing_word_index = test_sentence_token.index(missing_word)
    history = test_sentence_token[(missing_word_index-(n-1)):missing_word_index]
    logprob_result = {}
    perplexity_result = {}
    for word in choices:
        logprob = models.get(n).logscore(word, history)
        logprob_result[word] = logprob

        sentence_sequence = sentence.replace(missing_word, word)
        text_sequence = ngram_from_sentence(sentence_sequence, n)
        # text_sequence = [(history.append(word))]
        perplexity = models.get(n).perplexity(text_sequence)
        perplexity_result[word] = perplexity

    if criteria == "perplexity":
        sorted_preplexity = dict(sorted(perplexity_result.items(), key=lambda x: x[1], reverse=True))
        perplexity_value = next(iter(sorted_preplexity.items()))[1]
        word_from_perplexity = next(iter(sorted_preplexity.items()))[0]
        score = perplexity_value
        complete_proverb = sentence.replace(missing_word, word_from_perplexity)
        return complete_proverb, score
    else:
        sorted_logprob = dict(sorted(logprob_result.items(), key=lambda x: x[1], reverse=True))
        logprob_value = next(iter(sorted_logprob.items()))[1]
        word_from_logprob = next(iter(sorted_logprob.items()))[0]
        score = logprob_value
        complete_proverb = sentence.replace(missing_word, word_from_logprob)
        return complete_proverb, score


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    proverbs = load_proverbs(proverbs_fn)
    print("\nNombre de proverbes pour entraîner les modèles : ", len(proverbs))
    train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    for partial_proverb, options in test_proverbs.items():
        solution, valeur = cloze_test(partial_proverb, options, n=3, criteria="logprob")
        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
        print("\tSolution = {} , Valeur = {}".format(solution, valeur))
