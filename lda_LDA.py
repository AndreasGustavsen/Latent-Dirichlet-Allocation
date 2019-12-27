import logging
import numpy as np
import random
from lda_corpus import *
corpus = Corpus()

logging.basicConfig( level = logging.DEBUG, format="[%(asctime)s] %(levelname)s: %(message)s")


def categorical_sampler(probs): 
    """
    input: condistional probabilities calcluated by gibbs sampling: P(z_n = k | z_-n , w)
    output: an index between [0, K), K=number of topics
    """
    rand = random.random()           
    for idx in range(len(probs)):   
        rand -= probs[idx]
        if rand < 0:
            return idx               



class LDA(object):
    
    def __init__(self, n_topic, alpha, beta, valid_split):
        self.V = corpus.V
        self.M = corpus.M
        self.K = n_topic
        self.alpha = alpha
        self.beta = beta
        self.valid_split = valid_split

        assert alpha > 0 and beta > 0, 'Alpha and beta should be larger than zero'
        assert valid_split >= 0 and valid_split < 1, 'valid_split should be in interval: [0, 1)'
        assert isinstance(self.K, int), 'n_topic should be an integer'

        self.logger = logging.getLogger('LDA')
        

        
        
      
    def fit_initialize(self, docs):        
        """ initial all count-matrices as zeros and parameters randomly
            The Gibbs samoling will converge to the true topics smoothly in each iteration
        """
        self.N_sum = 0
        self.z_mn = []
        self.n_mk = np.zeros((self.M_valid, self.K), dtype='intc')
        self.n_m = np.zeros(self.M_valid, dtype='intc')
        self.n_kv = np.zeros((self.K, self.V), dtype='intc')
        self.n_k = np.zeros(self.K, dtype='intc')
        
        
        self.phi = np.empty((self.K, self.V), dtype='float64')                # P(w_n | z_k)
        self.theta = np.empty((self.M_valid, self.K), dtype='float64')        # P(z_k | d), d=[1,M], M = number of documents

 
        for doc_index, doc in enumerate(docs):
            self.N_sum += len(doc)
            z_m = np.empty(len(doc), dtype='intc')
            for word_index, word_in_idform in enumerate(doc):
                init_k = int(np.random.random(1) * self.K)  # generate a random number in range(n_topics)
                z_m[word_index] = init_k
                self.n_mk[doc_index][init_k] += 1
                self.n_kv[init_k][word_in_idform] += 1
                self.n_m[doc_index] += 1
                self.n_k[init_k] += 1
            self.z_mn.append(z_m)
            
           
        
          
            
            
            
            
            
    """ The algorithm is based on Gibbs-sampling not on variational inference """    
    def fit(self, corpus, n_iter, verbose): 
        self.V = corpus.V
        self.n_gibbs_iteration = n_iter
        self.verbose = verbose
        
        
        # check whether the given values fulfit the requirements or not
        
        assert isinstance(corpus, Corpus), 'Input should be Corpus type'
        assert isinstance(self.n_gibbs_iteration, int), 'n_iter should be an integer'
        assert isinstance(self.verbose, bool), 'verbose should be boolian'


        # if we dont want to use whole of docs
        
        self.M_valid = int(corpus.M * self.valid_split)     
        docs = corpus.docs_idform[: self.M_valid]
        print('The total number of documents is: ', corpus.M)
        print('The number of documents used for topic modeling is: ', len(docs))
        
        self.fit_initialize(docs)                                  
        
        
        for it in range(self.n_gibbs_iteration):
            self.collapsed_gibbs_sampling(docs)
            if self.verbose:
                self.logger.info('<iteration number: {0}>, perplexity: {1:.6g}, update rate after each iteration: {2:.6g}'.format(it + 1, self.perplexity(docs), float(self.update_k_count) / self.N_sum))
                
            
    def collapsed_gibbs_sampling(self, docs):
           
        self.update_k_count = 0
        for doc_index, doc in enumerate(docs):                         
            for word_index, word in enumerate(doc):
                old_k = self.z_mn[doc_index][word_index]
                    
                self.n_mk[doc_index][old_k] -= 1
                self.n_kv[old_k][word] -= 1
                self.n_m[doc_index] -= 1
                self.n_k[old_k] -= 1
                
                """ sample a new topic by weighted topic-probs """
                new_k = self.sample_a_topic_by_WeightedTopicProbabilites(doc_index, word)
                
                """ update the counts given the new topic """
                self.n_mk[doc_index][new_k] += 1
                self.n_kv[new_k][word] += 1
                self.n_m[doc_index] += 1
                self.n_k[new_k] += 1

                self.z_mn[doc_index][word_index] = new_k
                    
                self.update_counter(new_k, old_k)

        """ At the end of each iteration, update topic parameters: phi and theta """
        self.updates_parameters_phi_theta_after_each_iteration() 
        
  
           
    def update_counter(self, new_k, old_k):
        if new_k != old_k:
            self.update_k_count += 1   
            
        
            
    def sample_a_topic_by_WeightedTopicProbabilites(self, m, word): #  one can say: probs ∼ Dirichlet() in generative model 
        """
        input:   
            dirichlet parameters: alpha and beta
            counts: n_kv and n_mk
            Vocabulary size: V
            n_topics: K
        output:  
            posterior distirbution or prob_k: p(z_n = k|z_-n, w)
        """

        prob_k = ((self.n_kv[:, word] + self.beta)/(self.n_k[:] + self.V * self.beta))*((self.n_mk[m, :] + self.alpha)/(self.n_m[m] + self.alpha * self.K))
        
        prob_k = prob_k / prob_k.sum()             # normalizing the probabilities to sum of one

        return categorical_sampler(prob_k)         # generate a random topic by weigthed topic probabilities

    
    
    
    
            
    def updates_parameters_phi_theta_after_each_iteration(self): 
        """             
        OBS: the real topic parameters have been collapsed out. These phi and theta are approximate phi and theta which we get by gibbs sampling. The idea behind gibbs sampling is that the sampling converges to the true parameters with the long and enough number of iterations.
        """
        for k in range(self.K):
            self.phi[k] = (self.n_kv[k] + self.beta) / (self.n_k[k] + self.V * self.beta)

        for m in range(self.n_m.size):
            self.theta[m] = (self.n_mk[m] + self.alpha) / (self.n_m[m] + self.K * self.alpha)
            
            
            
            
            
    def perplexity(self, docs):      
        """
        Both theta and phi are needed to calculaate perplexity
        phi =  P(w_d | z) is a K*V matrix
        theta =  P(z | d) is a M*K matrix
        P(W_d) = P(w_n|z)*P(z|d)
        sum( log( P(W_d) ) )
        exp[ -sum/total_sum ] 
        """
        
        
        if not hasattr(self, 'theta') or not hasattr(self, 'phi'):      
            raise Exception('You should fit model first')


        M = self.theta.shape[0]           

        docs = docs[:M]                                                   
        expindex = 0.0                                                    
        for doc_index, doc in enumerate(docs):  
            for word in doc:
                p = np.dot(self.theta[doc_index, :], self.phi[:, word])   
                expindex += np.log(p)                                     
        perplexity_D_test = np.exp(-expindex / self.N_sum)                
        
        return perplexity_D_test     
    
            
            
    """ 
    ##################################################################################################
    #################################### FITTING Process is done #####################################  
    ##################################################################################################
    """
    
    
    
    """ shows n-top words for corresponding topic """
    
    def topic_word(self, n_top_word, corpus): 
        self.n_top_word = n_top_word
        
        if not hasattr(self, 'phi'):                                    
            raise Exception('You should fit model first')
            
        else:
            topic_word_list = []
            for k in range(self.K):                                     # phi is a K.V matrix
                word_list = []
                for index in self.phi[k].argsort()[-self.n_top_word:]:     
                    if corpus is not None:
                        word_list.append(corpus.id2word[index])         # give words with n_top highest probabilities.
                    else:
                        word_list.append(index)                         # give word_id instead (word_id in Vocabulary)
                topic_word_list.append(word_list)
            return topic_word_list
    


        
