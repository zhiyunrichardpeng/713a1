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
        flag = 1
        
        if not text: return []
        segmentation = [ w for w in text ] # segment each char into a word
        word_init = text[0]
        
        if flag ==1:
            flag=0
            ## Initialize the heap ##
            
            chart = [None]*len(text)
            
            Phy= None
#             print('word_init', word_init)
#             print('self.Pw(word_init)', log10(self.Pw(word_init)))
            heap = []
    
    
            newWordList = []
            #count=0
            for i in range(0,len(text)):
#                     print('text[i:i+count]',text[endindex+1:i+1])
                newWordList.append(text[0:i+1])

#             newWordList = newWordList[::-1]    
            for word in newWordList:
                newentry = (word, 0, log10(self.Pw(word)),  None) 
                if newentry not in heap:
                    heap.append(newentry)    
            heap.sort(key=lambda a: a[2])            
            #print('heap after init', heap)
#             heap.insert(0,(word_init, 0, log10(self.Pw(word_init)), Phy))
            chart[0] = (heap[0])
#             print('heap',heap)
#             print('heap top one',heap[-1])
            
            ## Iteratively fill in chart[i] for all i ##
            while(len(heap)!=0):
                entry = heap[-1]
                heap = heap[:-1]
               
#                 chart = [entry]

                currentWord = entry[0]
                endindex = entry[1]+ len(currentWord) -1
#                 entry[1] = endindex
                print('Adding : ',currentWord, log10(self.Pw(currentWord)))
                print('toprocess: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (entry[0],entry[1]+1, entry[2] ,str(entry[3])))  
                print('pop: word=%s', entry[0], 'logProb',log10(self.Pw(entry[0])))

#                 print('endIndex=', endindex, 'newEntry=chartEntry', chart)
                print('endIndex= %d : newEntry: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (endindex, entry[0],entry[1]+1, entry[2] ,str(entry[3])))  
#                 print('chart No.', endindex, 'chartEntry', chart)
               
                if chart[endindex]!=None and chart[endindex][3]!= None and endindex!=0: # endindex!=0:  # entry[3]!=-1  #and chart[endindex][3]!= 0
#                     index = chart[endindex][3]
                    preventry = chart[chart[endindex][3]]
                    #if entry[2]>chart[endindex-1][2]:
                    if entry[2]>preventry[2]:
                        chart[endindex] = entry
                    if entry[2]<=preventry[2]:
                        continue
                else:
                    chart[endindex] = entry
                    
                # for each newword that matches input starting at position endindex+1    
                
#                 for 
                newWordList = []
                #count=0
                for i in range(endindex+1,len(text)):
#                     print('text[i:i+count]',text[endindex+1:i+1])
                    newWordList.append(text[endindex+1:i+1])
                
                newWordList = newWordList[::-1]
                    
                    #count+=1

# for each newword that matches input starting at position endindex+1

#     newentry = Entry(newword, endindex+1, entry.log-probability + logPw

# (newword), entry)
# if newentry does not exist in heap:

#     insert newentry into heap
                
                for word in newWordList:
                    newentry = (word, endindex+1, entry[2] + log10(self.Pw(word)),  entry[1]) 
                    if newentry not in heap:
                        heap.append(newentry)
                        
                heap.sort(key=lambda a: a[2])
#                         print('heap after append',heap)
                
#                 newWord = text[endindex+2]
#                 print('newWord', newWord)
#                 backptr = entry[1] # endindex - len(currentWord)
#                 newEntry = (newWord, endindex+1, chart[backptr][2] + log10(self.Pw(newWord)), backptr)
#                 print('heap', heap, 'newEntry',newEntry)
#                 if newEntry not in heap:
#                     print('we append heap really')
#                     heap.append(newEntry)
#                 print('heap after append',heap)
            
#             finalindex = len(text)
#             print('finalindex', finalindex)
#             finalentry = chart[finalindex]
#             print('finalentry', finalentry)       
            finalindex = len(text) - 1
            finalentry = chart[finalindex]
    
            while(finalentry[3]!=None):
                print("the final result: \n",finalentry)
                finalentry = chart[finalentry[3]]
            
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
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
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
#     return 10./(N * 10**len(word))####################



if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w))
#     Pw = Pdist(data=datafile(opts.counts1w), missingfn=avoid_long_words) ####################    
    segmenter = Segment(Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
