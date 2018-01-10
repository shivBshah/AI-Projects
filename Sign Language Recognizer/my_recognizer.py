import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for seq in test_set.get_all_sequences():
        item_X, item_len = test_set.get_item_Xlengths(seq)
        scores = dict()
        best_guess = None
        max_score = float('-inf')
        for word in models:
            model = models[word]
            try:
                this_score = model.score(item_X, item_len)
                if this_score > max_score:
                    max_score = this_score
                    best_guess = word
                scores[word] = this_score
            except:
                scores['word'] = None
        probabilities.append(scores)
        guesses.append(best_guess)
    return probabilities, guesses
