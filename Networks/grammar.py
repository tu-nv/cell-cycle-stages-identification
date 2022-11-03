#!/usr/bin/python3
import os
import numpy as np


class Grammar(object):
    
    # @context: tuple containing the previous label indices
    # @label: the current label index
    # @return: the log probability of label given context p(label|context)
    def score(self, context, label): # score is a log probability
        return 0.0

    # @return: the number of classes
    def n_classes(self):
        return 0

    # @return sequence start symbol
    def start_symbol(self):
        return -1

    # @return sequence end symbol
    def end_symbol(self):
        return -2

    # @context: tuple containing the previous label indices
    # @return: list of all possible successor labels for the given context
    def possible_successors(context):
        return set()

# grammar containing all action transcripts seen in training
# used for inference
class PathGrammar(Grammar):
    
    def __init__(self, outpath , n_classes):
        self.num_classes = n_classes
        transcripts = self._read_transcripts(outpath)
        # generate successor sets
        self.successors = dict()
        for transcript in transcripts:
            transcript = transcript + [self.end_symbol()]
            for i in range(len(transcript)):
                context = (self.start_symbol(),) + tuple(transcript[0:i])
                self.successors[context] = set([transcript[i]]).union( self.successors.get(context, set()) )

    def _read_transcripts(self, outpath):
        transcripts = []
        with open(os.path.join(outpath,'grammar.txt')) as f:
            trans_str = f.read().split('\n')[0:-1]
        for transcript in trans_str:
            trans = transcript.replace(' ','').replace('[','').replace(']','').replace('\n','')
            transcripts.append([int(s) for s in trans.split(',')])
        #print (transcripts)
        self.transcripts = transcripts
        return transcripts

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf


# grammar that generates only a single transcript
# use during training to align frames to transcript
class SingleTranscriptGrammar(Grammar):

    def __init__(self, transcript, n_classes):
        self.num_classes = n_classes
        transcript = transcript + [self.end_symbol()]
        self.successors = dict()
        for i in range(len(transcript)):
            context = (self.start_symbol(),) + tuple(transcript[0:i])
            self.successors[context] = set([transcript[i]]).union( self.successors.get(context, set()) )

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf

