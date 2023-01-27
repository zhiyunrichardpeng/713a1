import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
from math import log2
import math

class Entry():
    def __init__(self, word, end, logprob, backptr):
        self.word = word
        self.end = end
        self.logprob = logprob
        self.backptr = backptr
    def __lt__(self, other):
        if self.logprob>other.logprob:
            return True
        else:
            return False



class Segment:

    def __init__(self, Pw, P2w):
        self.Pw = Pw
        self.P2w = P2w
        
    def inputMatch(self, position, text, prevEntry):
        entryList = []
        for i in range( min((self.Pw).maxLength, len(text)-position)):  #(self.Pw).maxLength  5 8
            word = text[position:position+i+1]
            if word in (self.Pw).keys():
                logprob = log10(self.Pw(word))
#                 if prevEntry is None:
#                     logprob = log10(self.Pw(word))
#                 else:
#                     logprob = log10(self.cPw(word, prevEntry.word))
                end = position + i+1
                entryList.append(Entry(word, end, logprob, None))
                
            if len(entryList)==0:
                for i in range(len(text)-position):
                    if(i<3):
                        end=position+i+1
                        word = text[position:end] #-1
                        logprob = log10(self.Pw(word))
                        entryList.append(Entry(word,end,logprob,None))
            return entryList

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
#         flag = 1
        
        if not text: return []
        entryHeap = self.inputMatch(0,text,None)
        heapq.heapify(entryHeap)
        chart=[None]*len(text)
        while len(entryHeap)!=0:
            entry = heapq.heappop(entryHeap)
            endindex = entry.end-1
#         self.word = word
#         self.end = end
#         self.logprob = logprob
#         self.backptr = backptr            
#             print('Adding : ',entry[0], log10(self.Pw(entry[0])))
#             print('toprocess: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (entry[0],entry[1]+1, entry[2] ,str(entry[3])))              
#             print('pop: word=%s', entry[0], 'logProb',log10(self.Pw(entry[0])))
#             print('endIndex= %d : newEntry: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (endindex, entry[0],entry[1]+1, entry[2] ,str(entry[3])))              
            if chart[endindex] is not None:
                preventry = chart[endindex]
                if entry.logprob > preventry.logprob:
                    chart[endindex] = entry
                else:
                    continue
            else:
                chart[endindex] = entry
                if (endindex+1)<len(text):
                    entryList = self.inputMatch(endindex+1,text,chart[endindex])
                    for newEntry in entryList:
                        newStandardEntry = Entry(newEntry.word,newEntry.end,newEntry.logprob+entry.logprob, endindex)
                        if any(newStandardEntry==existEntry for existEntry in entryHeap) is not True:
                            heapq.heappush(entryHeap,newStandardEntry)
                            
        finalEntry = chart[-1]
#         print("the final result: \n",finalentry)        
        currentEntry = finalEntry
        segmentation = []
        segmentation.append(currentEntry.word)
        while currentEntry.backptr is not None:
            currentEntry = chart[currentEntry.backptr]
            segmentation.append(currentEntry.word)
        segmentation.reverse()
        return segmentation
    
    def cPw(self, word, prev):
        bigramProb = 0
        unigramProb = 0
        if prev + ' ' + word in self.P2w:
            bigramProb = self.P2w[prev + ' ' + word]/float(self.Pw[prev])
        if word in self.Pw:
            unigramProb = self.Pw(word)
        bigram_lambda = 0.5
                                                           
        unigram_lambda=0.3
        return bigram_lambda*bigramProb+unigram_lambda*unigramProb+(1-bigram_lambda-unigram_lambda)*(1/float(len(self.Pw)))
        


    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.maxLength = max(map(len, self.keys()))
#         self.missingfn = missingfn or (lambda k, N: 1./N)
        self.missingfn = missingfn or (lambda k, N: 1./(N*10000**len(k)))        
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

# def avoid_long_words(word, N):####################
#     return 1./(N*10000**len(word)) #10./(N * 10**len(word))####################



if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w))#, missingfn=avoid_long_words)
    P2w = Pdist(data=datafile(opts.counts2w))#, missingfn=avoid_long_words)
    segmenter = Segment(Pw, P2w)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
