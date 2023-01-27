import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math, heapq
from functools import reduce
from collections import defaultdict
from math import log10

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

    # find the matched word start with char
    def inputMatch(self, position, text, prevEntry):
        entryList = []
        #for all the words within the max length
        for i in range( min((self.Pw).maxLength, len(text)-position) ):
            word = text[position:position+i+1]
            if word in (self.Pw).keys():
                if prevEntry is None:
                    logprob = log10(self.Pw(word))
                else:
                    logprob = log10(self.cPw(word, prevEntry.word))
                end = position + i+1
                entryList.append(Entry(word, end, logprob, None))
        if len(entryList) == 0:
            for i in range(len(text)-position):
                if(i<3):
                    end = position + i + 1
                    word = text[position:end]  # -1
                    logprob = log10(self.Pw(word))
                    entryList.append(Entry(word, end, logprob, None))
            # entryList.append(Entry(char, end, logprob, None))
            # entryList.append(Entry(char, end, logprob, None))
        return entryList

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        "initialize the heap"
        entryHeap = self.inputMatch(0, text, None)
        heapq.heapify(entryHeap)
        "Iteratively fill in chart[i] for all i"
        chart = [None]*len(text)
        while len(entryHeap) != 0:
            entry = heapq.heappop(entryHeap)
            endindex = entry.end - 1
            if chart[endindex] is not None:
                preventry = chart[endindex]
                if entry.logprob > preventry.logprob:
                    chart[endindex] = entry
                else:
                    continue
            else:
                chart[endindex] = entry
                #if the text does not finish
                if (endindex+1) < len(text):
                    entryList = self.inputMatch(endindex+1, text, chart[endindex])
                    for newEntry in entryList:
                        newStandardEntry = Entry(newEntry.word, newEntry.end, newEntry.logprob+entry.logprob, endindex)
                        if any(newStandardEntry == existEntry for existEntry in entryHeap) is not True:
                            heapq.heappush(entryHeap, newStandardEntry)
        "Get the best segmentation"
        finalEntry = chart[-1]
        currentEntry = finalEntry
        segmentation = []
        segmentation.append(currentEntry.word)
        while currentEntry.backptr is not None:
            currentEntry = chart[currentEntry.backptr]
            segmentation.append(currentEntry.word)
        segmentation.reverse()
        return segmentation

    def cPw(self, word, prev):
        "Conditional probability of word, given previous word."
        #Laplace Add-1 smoothing
        # if prev + ' ' + word in self.P2w:
        #     if prev in self.Pw:
        #         return (1 + self.P2w[prev + ' ' + word])/(len(self.Pw)+float(self.Pw[prev]))
        #     else:
        #         return (1 + self.P2w[prev + ' ' + word])/(len(self.Pw))
        # else:
        #     if prev in self.Pw:
        #         return 1 / (len(self.Pw)+float(self.Pw[prev]))
        #     else:
        #         return 1 / len(self.Pw)
        # Jelinek-Mercer smoothing
        bigramProb = 0
        unigramProb = 0
        if prev + ' ' + word in self.P2w:
            bigramProb = self.P2w[prev + ' ' + word] / float(self.Pw[prev])
        if word in self.Pw:
            unigramProb = self.Pw(word)
        bigram_lambda = 0.5
        unigram_lambda = 0.3
        return bigram_lambda*bigramProb + unigram_lambda*unigramProb + (1-bigram_lambda-unigram_lambda)*(1/float(len(self.Pw)))


    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)

### Support functions (p. 224)

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
        self.missingfn = missingfn or (lambda k, N: 1./(N*10000**len(k)))
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name,'r', encoding='UTF-8') as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    # relative path for debug within the .py file
    # optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('hw1', 'data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    # optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('hw1', 'data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    # optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('hw1', 'data','input', 'dev.txt'), help="file to segment")

    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data','input', 'dev.txt'), help="file to segment")
    # optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data','input', 'test.txt'), help="file to segment")

    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w))
    P2w = Pdist(data=datafile(opts.counts2w))
    segmenter = Segment(Pw, P2w)
    with open(opts.input,'r', encoding='UTF-8') as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
