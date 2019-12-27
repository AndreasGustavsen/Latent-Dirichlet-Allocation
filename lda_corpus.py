import numpy as np
class Corpus(object):

    def __init__(self):
        self.id2word = {}
        self.word2id = {}
        self.index2doc = {}
        self.docs_idform = []        
        self.V = 0 
        self.M = 0
    
    """ Each document is considered as a line and words in a documents are separated by whitespace. """            
    
    def load_text(self, filepath):                             
        input_file = open(filepath, 'r') 

        for index_doc, doc in enumerate(input_file):       
            """  making docs in id form , index2doc, word2id and id2word  """  
            
            doc = doc.strip().split(' ')         
            self.index2doc[index_doc] = doc                       # docs in tokenized form   
            doc_id = np.empty(len(doc), dtype='intc')           
            
            
            for index, word in enumerate(doc):          
                if word not in self.word2id:                      # if unseen
                    current_id = len(self.word2id)    
                    self.word2id[word] = current_id               # word2id is a dictionary with word and id
                    self.id2word[current_id] = word               # id2word is a dictionay with id and word 

                    doc_id[index] = current_id 
                else:                                             # if seen
                    doc_id[index] = self.word2id[word]            
            self.docs_idform .append(doc_id)
            

        self.V = len(self.word2id)                                # number of unique words in whole of corpus
        self.M = len(self.docs_idform )                           # number of docs in corpus
        print("Done processing corpus with {} documents".format(len(self.docs_idform )))

        input_file.close() 
        
 


""" Example on some output 

input = ["apple orange mango melon", "dog cat bird rat",]

outputs: 
        
1. id2word = {0: 'apple', 1: 'orange', 2: 'mango', 3: 'melon', 4: 'dog', 5: 'cat', 6: 'bird', 7: 'rat'}

2. word2id = {'apple': 0, 'orange': 1, 'mango': 2, 'melon': 3, 'dog': 4, 'cat': 5, 'bird': 6, 'rat': 7}

3. docs_idform = [array([0, 1, 2, 3], dtype=int32), array([4, 5, 6, 7], dtype=int32)]

4. index2doc = {0: ['apple', 'orange', 'mango', 'melon'], 1: ['dog', 'cat', 'bird', 'rat']}

"""
        
        