#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import sys
import jieba

if __name__=='__main__':

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    if len(sys.argv) < 3:
        print(globals()['__doc__'] %locals())
        sys.exit(1)

    inp, outp = sys.argv[1:3]
    space = ' '

    output = open(outp, 'w')
    inputer = open(inp, 'r')

    for line in inputer.readlines():
        seg_list = jieba.cut(line)
        output.write(space.join(seg_list) + '\n')

