from lda_LDA import *
from lda_corpus import *
from lda_filtering import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))


def read_documents_reviews(doc_file):
    docs = []; labels = []; 
    with open(doc_file, encoding = 'utf-8') as file:
        for line in file:
            label, _, _, doc = line.strip().split( maxsplit = 3 )
            docs.append(doc)
            labels.append(label)
    return docs, labels


docs, labels = read_documents_reviews('all_sentiment_shuffled.txt')
f = filtering(stemmer, stop_words)
filtered_docs = f.preprocessing(docs)

with open("temporary_file.txt", "w") as output:
    for doc in filtered_docs:
        output.write('%s\n' % doc)

        
procentage_use_data = 0.1
numberOfTopic = 6
numberOfGibbsIteration = 2
alpha = 0.05 
beta = 0.01    # alternative 1/numberOfTopic
numberOfPrintedTopWordsPerTopic = 10
numnerOfExisitingTopTopicInADocument = 1
corpus = Corpus()
corpus.load_text('temporary_file.txt')
model = LDA(n_topic = numberOfTopic, alpha = alpha, beta = beta, valid_split = procentage_use_data)



model.fit(corpus, n_iter = numberOfGibbsIteration, verbose = True)

print('The top words for corresponding topic in ascending order is as follow: ')
print(model.topic_word(n_top_word = numberOfPrintedTopWordsPerTopic, corpus = corpus))








