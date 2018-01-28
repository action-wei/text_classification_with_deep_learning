#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import multiprocessing

from time import time
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import BrownCorpus

if __name__== "__main__":

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s' )
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    inp, outp1, outp2= sys.argv[1:4]
    
    begin = time()
#    model = Word2Vec(BrownCorpus(inp), size=100, window=5, min_count=5,
#                     workers=multiprocessing.cpu_count())
    model = Word2Vec(LineSentence(inp),sg=1, size=200, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=True)
    
    end = time()
    print("total processing time:%d seconds" %(end-begin))
