import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
from math import log2
import math

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw
    
    # insert new entries into a list which matches the position in input
    def MatchWords(self, endposition, text, prevEntry):
        words = []
        #find L = min(maxlen, j)
        L = min((self.Pw).maxLength, len(text)-endposition)
        for i in range (L):
            word = text[endposition:endposition + i + 1]
            if word in (self.Pw).keys():
                logp = log10(self.Pw(word))
                end = endposition + i + 1
                words.append((word, end, logp, None))  
                print('Adding : ', word, log10(self.Pw(word)))
        if len(words) == 0:
            for i in range(len(text)-endposition):
                if(i<3):
                    end = endposition + i + 1
                    word = text[endposition:end]  # -1
                    logp = log10(self.Pw(word))
                    words.append((word, end, logp, None)) 
                    print('Adding : ', word, log10(self.Pw(word)))
        return words

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."

        #if not text: return []
        #segmentation = [w for w in text] 
        
        ## Initialize the heap ##
        Heap = []
        Heap = self.MatchWords(0, text, None)
        Heap.sort(key=lambda a: a[2])
        chart = [None] * len(text)
        

        ## Iteratively fill in chart[i] for all i ##
        while (len(Heap) != 0):
            #pop the largest prob from heap
            entry = Heap[-1]
            Heap = Heap[:-1]

            print('toprocess: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (
                entry[0], entry[1], entry[2], str(entry[3])))
            print('pop: word=%s', entry[0], 'logProb', log10(self.Pw(entry[0]))  )
            
            #get the endindex
            endindex = entry[1] - 1
            
            #check chart[endindex] has preventry or not
            if chart[endindex] is not None:
                preventry = chart[endindex]  #has preventry
                if entry[2] > preventry[2]:
                    chart[endindex] = entry  
                else:
                    continue
            else:
                chart[endindex] = entry
                if endindex < len(text) - 1:
                    entryList = self.MatchWords(endindex+1, text, chart[endindex])     #get the new entry list
                    for currEntry in entryList:
                        newEntry = (currEntry[0], currEntry[1], currEntry[2]+entry[2], endindex)   #update the prob of current entry
                        if any(newEntry == existEntry for existEntry in Heap) is not True:
                            print('endIndex= %d : currEntry: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (
                                endindex, entry[0], entry[1], entry[2], str(entry[3])   ))
                            Heap.append(newEntry)
                    Heap.sort(key=lambda a: a[2])
            
            #print output
            for i1 in range(len(chart)):
                if chart[i1] != None:
                    print('chart[ %s ]: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (
                    str(i1), chart[i1][0], chart[i1][1], chart[i1][2], str(chart[i1][3])))

        # return segmentation
        finalEntry = chart[-1]
        for i1 in range(len(chart)):
            if chart[i1] != None:
                print('final chart[ %s ]: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (
                    str(i1), chart[i1][0], chart[i1][1], chart[i1][2], str(chart[i1][3])))
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
        self.missingfn = missingfn or (lambda k, N: 1./(N*8000**len(k)))   #same reason as A0, avoid long words

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
