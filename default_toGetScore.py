import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
from math import log2
import math

# class Entry():
#     def __init__(self, word, end, logprob, backptr):
#         self.word = word
#         self.end = end
#         self.logprob = logprob
#         self.backptr = backptr
#     def __lt__(self, other):
#         if self.logprob>other.logprob:
#             return True
#         else:
#             return False

class Segment:

    def __init__(self, Pw,P2w):
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
                    logprob = log10(self.cPw(word, prevEntry[0]))
                end = position + i+1
                entryList.append((word, end, logprob, None))  #Entry
        if len(entryList) == 0:
            for i in range(len(text)-position):
                if(i<3):
                    end = position + i + 1
                    word = text[position:end]  # -1
                    logprob = log10(self.Pw(word))
                    entryList.append((word, end, logprob, None)) #Entry
            # entryList.append(Entry(char, end, logprob, None))
            # entryList.append(Entry(char, end, logprob, None))
        return entryList

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        # flag = 1

        if not text: return []
        # segmentation = [w for w in text]  # segment each char into a word
        # word_init = text[0]
        entryHeap=[]
        entryHeap = self.inputMatch(0, text, None)
        # heapq.heapify(entryHeap)
        entryHeap.sort(key=lambda a: a[2])

        # if flag == 1:
        # flag = 0
        ## Initialize the heap ##

        chart = [None] * len(text)


        # Phy = None
        #             print('word_init', word_init)
        #             print('self.Pw(word_init)', log10(self.Pw(word_init)))
        # heap = []

        # newWordList = []
        # # count=0
        # for i in range(0, len(text)):
        #     #                     print('text[i:i+count]',text[endindex+1:i+1])
        #     newWordList.append(text[0:i + 1])
        #
        # #             newWordList = newWordList[::-1]
        # for word in newWordList:
        #     newentry = (word, 0, log10(self.Pw(word)), None)
        #     if newentry not in heap:
        #         heap.append(newentry)
        # heap.sort(key=lambda a: a[2])
        # # print('heap after init', heap)
        # #             heap.insert(0,(word_init, 0, log10(self.Pw(word_init)), Phy))
        # chart[0] = (heap[0])
        #             print('heap',heap)
        #             print('heap top one',heap[-1])

        ## Iteratively fill in chart[i] for all i ##
        while (len(entryHeap) != 0):
            # entry = heapq.heappop(entryHeap)
            entry = entryHeap[-1]
            entryHeap = entryHeap[:-1]

            endindex = entry[1] - 1
            if chart[endindex] is not None:
                preventry = chart[endindex]
                if entry[2] > preventry[2]:
                    chart[endindex] = entry
                else:
                    continue
            else:
                chart[endindex] = entry
                #if the text does not finish
                if (endindex+1) < len(text):
                    entryList = self.inputMatch(endindex+1, text, chart[endindex])
                    for newEntry in entryList:
                        newStandardEntry = (newEntry[0], newEntry[1], newEntry[2]+entry[2], endindex)
                        if any(newStandardEntry == existEntry for existEntry in entryHeap) is not True:
                            # heapq.heappush(entryHeap, newStandardEntry)
                            entryHeap.append(newStandardEntry)

            # entry = heap[-1]
            # heap = heap[:-1]
            #
            # #                 chart = [entry]
            #
            # currentWord = entry[0]
            # endindex = entry[1] + len(currentWord) - 1
            #                 entry[1] = endindex
            # print('Adding : ', currentWord, log10(self.Pw(currentWord)))
            # print('toprocess: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (
            # entry[0], entry[1] + 1, entry[2], str(entry[3])))
            # print('pop: word=%s', entry[0], 'logProb', log10(self.Pw(entry[0])))
            #
            # #                 print('endIndex=', endindex, 'newEntry=chartEntry', chart)
            # print('endIndex= %d : newEntry: chartEntry (start=%s, end=%d logprob=%f, backptr=%s)' % (
            # endindex, entry[0], entry[1] + 1, entry[2], str(entry[3])))
            # #                 print('chart No.', endindex, 'chartEntry', chart)
            #
            # if chart[endindex] != None and chart[endindex][
            #     3] != None and endindex != 0:  # endindex!=0:  # entry[3]!=-1  #and chart[endindex][3]!= 0
            #     #                     index = chart[endindex][3]
            #     preventry = chart[chart[endindex][3]]
            #     # if entry[2]>chart[endindex-1][2]:
            #     if entry[2] > preventry[2]:
            #         chart[endindex] = entry
            #     if entry[2] <= preventry[2]:
            #         continue
            # else:
            #     chart[endindex] = entry
            #
            # # for each newword that matches input starting at position endindex+1
            #
            # #                 for
            # newWordList = []
            # # count=0
            # for i in range(endindex + 1, len(text)):
            #     #                     print('text[i:i+count]',text[endindex+1:i+1])
            #     newWordList.append(text[endindex + 1:i + 1])
            #
            # newWordList = newWordList[::-1]
            #
            # # count+=1
            #
            # # for each newword that matches input starting at position endindex+1
            #
            # #     newentry = Entry(newword, endindex+1, entry.log-probability + logPw
            #
            # # (newword), entry)
            # # if newentry does not exist in heap:
            #
            # #     insert newentry into heap
            #
            # for word in newWordList:
            #     newentry = (word, endindex + 1, entry[2] + log10(self.Pw(word)), entry[1])
            #     if newentry not in heap:
            #         heap.append(newentry)
            #
            # heap.sort(key=lambda a: a[2])
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
        # finalindex = len(text) - 1
        # finalentry = chart[finalindex]
        #
        # while (finalentry[3] != None):
        #     print("the final result: \n", finalentry)
        #     finalentry = chart[finalentry[3]]

        # return segmentation
        finalEntry = chart[-1]
        currentEntry = finalEntry
        segmentation = []
        segmentation.append(currentEntry[0])
        while currentEntry[3] is not None:
            currentEntry = chart[currentEntry[3]]
            segmentation.append(currentEntry[0])
        segmentation.reverse()
        return segmentation

    def Pwords(self, words):
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)


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
        self.missingfn = missingfn or (lambda k, N: 1./(N*10000**len(k)))

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


# def avoid_long_words(word, N):####################
#     return 10./(N * 10**len(word))####################


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
    P2w = Pdist(data=datafile(opts.counts2w))
    #     Pw = Pdist(data=datafile(opts.counts1w), missingfn=avoid_long_words) ####################
    segmenter = Segment(Pw,P2w)
    with open(opts.input,'r', encoding='UTF-8') as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
