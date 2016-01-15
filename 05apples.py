import string as string #lib for string manipulations
import nltk as nltk #Natural Language processing ToolKit
import re as re #regular expressions (for tokenizing)
import fileinput #for reading from standard input
from nltk.corpus import brown

#functions from nltk tutorial for feeding nBayes classifier data in proper format
# takes in: line of text made of WORDs
# returns: dictionary in form {'contains(WORD)': True}
def features(sentence):
     words = sentence.lower().split()
     return dict(('contains(%s)' % w, True) for w in words)
# takes in: line of text made of WORDs
# returns: dictionary in form {'contains(WORD)': None}
def notfeatures(sentence):
     words = sentence.lower().split()
     return dict(('contains(%s)' % w, None) for w in words)

def labelFruit(sentence):
    words = sentence.lower().split()
    return dict((w, 'fruit') for w in words)

def labelComputers(sentence):
    words = sentence.lower().split()
    return dict((w, 'computers') for w in words)

#reading wikipedia articles on apples and Apple
fruitText = open("apple-fruit.txt").read()
fruitTokens = re.split(r'\W+',fruitText)
fruitWords= nltk.Text(fruitTokens)

computersText = open("apple-computers.txt").read()
computersTokens= re.split(r'\W+', computersText)
computersWords= nltk.Text(computersTokens)

#calculating frequency distributions of words in wiki articles
#this should be followed by other processes like getting rid of common words like "the" or "and"
fruitWordsFreq = nltk.FreqDist(w.lower() for w in fruitWords)
computersWordsFreq = nltk.FreqDist(w.lower() for w in computersWords)

#choosing arbitrlly only a subset of words for training the model
fruitFeatures = list(map(features, fruitWordsFreq.keys()[2:100]))
computersFeatures= list(map(notfeatures, computersWordsFreq.keys()[2:2000]))

fruitLabels = list(map(labelFruit, fruitWordsFreq.keys()[5:40]))
computersLabels = list(map(labelComputers, computersWordsFreq.keys()[5:40]))
allLabels = fruitLabels + computersLabels
print(allLabels)

#creating and training Positive Naive Bayes model with features from wiki articles
naiveB = nltk.PositiveNaiveBayesClassifier.train(fruitFeatures, computersFeatures)

#dTree = nltk.classify.decisiontree.DecisionTreeClassifier.train(allLabels)

#main part - loop going through the standard input and calssiffing lines
i=0
for line in fileinput.input():
    i+=1 #controlling i for omitting the first line
    if i == 1:
        pass
    else:
        nbResult = naiveB.classify(notfeatures(line))
        #result= dTree.classify(line)
        if nbResult == True:
            print('fruit')
        elif nbResult == False:
            print('computer-company')
        else:
            print('something is wrong')
        #if result == True:
        #    print('tree fruit')
        #elif result == False:
        #    print('tree computer-company')
        #else:
        #    print('something is wrong')
            

suffix_fdist = nltk.FreqDist()
for word in brown.words():
     word = word.lower()
     suffix_fdist.inc(word[-1:])
     suffix_fdist.inc(word[-2:])
     suffix_fdist.inc(word[-3:])
common_suffixes = suffix_fdist.keys()[:100]
def pos_features(word):
     features = {}
     for suffix in common_suffixes:
         features['endswith(%s)' % suffix] = word.lower().endswith(suffix)
     return features

tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
print(featuresets)
