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
fruitFeatures = list(map(features, fruitWordsFreq.keys()[:100]))
computersFeatures= list(map(notfeatures, computersWordsFreq.keys()[:100]))

print(fruitFeatures)
print(computersFeatures)

#creating and training Positive Naive Bayes model with features from wiki articles
naiveB = nltk.PositiveNaiveBayesClassifier.train(fruitFeatures, computersFeatures)


#main part - loop going through the standard input and calssiffing lines
i=0
for line in fileinput.input():
    i+=1 #controlling i for omitting the first line
    if i == 1:
        pass
    else:
        nbResult = naiveB.classify(notfeatures(line))
        if nbResult == True:
            print('fruit')
        elif nbResult == False:
            print('computer-company')
        else:
            print('something is wrong')
  


