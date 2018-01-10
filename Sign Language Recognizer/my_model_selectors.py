import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_num = None
        best_score = float('inf')
        for num_comp in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_comp)
                logL = model.score(self.X, self.lengths)
                p = num_comp + num_comp * (num_comp-1) + len(self.X[0]) * num_comp * 2
                bic_score = -2 * logL + p * np.log(len(self.X))

                if bic_score < best_score:
                    best_score = bic_score
                    best_num = num_comp
            except:
                pass

        return self.base_model(best_num)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num = None
        best_score = float('-inf')
        for num_comp in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_comp)
                self_logL = model.score(self.X, self.lengths)
                
                sum_other_logL = 0
                count = 0
                for word in self.words:
                    if word != self.this_word:
                        other_X, other_len = self.hwords[word]
                        sum_other_logL += model.score(other_X, other_len)
                        count += 1
                
                avg_other_logL = 0.
                if count != 0:
                    avg_other_logL = sum_other_logL/count
                dic_score = self_logL - avg_other_logL
                if best_score is None or dic_score > best_score:
                    best_score = dic_score
                    best_num = num_comp
            except:
                pass
        if best_num is None:
            return None
        return self.base_model(best_num)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # TODO implement model selection using CV
        best_score = float('-inf')
        best_num = None
        for num_comp in range(self.min_n_components, self.max_n_components+1):
            num_splits = len(self.sequences)
            if num_splits == 1:
                continue
            
            if len(self.sequences) >= 3:
                num_splits = 3

            split_method = KFold(num_splits)

            sum_logL = 0
            count = 0
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train, len_train = combine_sequences(cv_train_idx, self.sequences)
                    model = GaussianHMM(n_components=num_comp, covariance_type="diag", n_iter=1000, 
                                        random_state=self.random_state, verbose=False).fit(X_train, len_train)
                    X_test, len_test = combine_sequences(cv_test_idx, self.sequences)
                    sum_logL += model.score(X_test, len_test)
                    count += 1
            except:
                pass    

            avg_logL = 0.
            if count != 0:
                avg_logL = sum_logL/count
            if best_score is None or avg_logL > best_score:
                best_num = num_comp
                best_score = avg_logL
            
        if best_num is None:            
            return None

        return self.base_model(best_num)
