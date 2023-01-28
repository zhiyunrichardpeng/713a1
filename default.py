import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
from math import log2
import math


class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."

        if not text: return []
        # segmentation = [w for w in text]  # segment each char into a word
        # word_init = text[0]
        
        ## Initialize the heap ##
        heap=[]
        heap = self.MatchWords(0, text)
        heap.sort(key=lambda a: a[2])

        chart = [None] * len(text)

        ## Iteratively fill in chart[i] for all i ##
        while (len(heap) != 0):
            
            # pop the largest prob entry from heap
            entry = heap[-1]
            heap = heap[:-1]
            
            # find endindex of current text segmentation / startindex of next text segmentation
            endindex = entry[1] + len(entry[0]) - 1
            
            # check chart[endindex] has preventry or not
            if chart[endindex] is not None:
                prevEntry = chart[endindex]
                if entry[2] > prevEntry[2]:    # check the prob of two entries
                    chart[endindex] = entry
                else:
                    continue
            else:
                chart[endindex] = entry
                
                entryList = self.MatchWords(endindex + 1, text)
                for currEntry in entryList:
                    newEntry = (currEntry[0], currEntry[1], currEntry[2] + entry[2], endindex) # update prob 
                    if newEntry not in heap:
                        heap.append(newEntry)

        # use back pointer to trace the segmentation
        finalEntry = chart[-1]
        segmentation = []
        segmentation.append(finalEntry[0])
        while finalEntry[3] is not None:
            finalEntry = chart[finalEntry[3]]
            segmentation.append(finalEntry[0])
        segmentation.reverse()
            
        return segmentation

    def Pwords(self, words):
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)
    
    # insert new entries into a list which matches the position in input
    def MatchWords(self, startPosition, text):
        entryList = []
        L = min((self.Pw).maxLength, len(text) - startPosition)  # L = min(maxlen, j) in order to avoid long words
        for i in range(L):
            currword = text[startPosition : startPosition + i + 1] # the current text segmentation
            if currword in self.Pw:
                entryList.append((currword, startPosition, log10(self.Pw(currword)), None)) # add the current entry
        
        # if words not in dictionary
        if len(entryList) == 0:
            for i in range(len(text) - startPosition):
                if(i < 4):
                    currword = text[startPosition : startPosition + i + 1]  
                    entryList.append((currword, startPosition, log10(self.Pw(currword)), None)) 
        return entryList


#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)


class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, data=[], N=None, missingfn=None):
        for key, count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.maxLength = max(map(len, self.keys()))
        self.missingfn = missingfn or (lambda k, N: 1./(N*8000**len(k))) # #same reason as A0, avoid long words

    def __call__(self, key):
        if key in self:
            return self[key] / self.N
        else:
            return self.missingfn(key, self.N)


def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name,'r', encoding='UTF-8') as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'),
                         help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'),
                         help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'),
                         help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w))
    segmenter = Segment(Pw)
    with open(opts.input,'r', encoding='UTF-8') as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
