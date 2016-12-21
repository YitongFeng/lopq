import numpy as np
import logging
from python.lopq.search import LOPQSearcher
from python.lopq.model import *
import struct

def load_data(feature_file, name_file):
    names = []

    with open(name_file) as nf:
        file_names = [i.strip() for i in nf.readlines()]

    with open(feature_file, 'rb') as f:
        num = np.uint32(struct.unpack('i', f.read(4)) )[0]
        fea_dim = np.uint32(struct.unpack('i', f.read(4)))[0]
        A = np.zeros((num, fea_dim), 'float32')
        for i in xrange(num):
            print i
            for j in xrange(fea_dim):
                A[i, j] = np.float32(struct.unpack('f', f.read(4)))[0]

    return A, file_names

#train, train_names = load_data('../Data/FV/codefile_sample.bin', '../Data/FV/names_sample.txt')

m = LOPQModel(V=16, M=8)
m.fit(train, n_init=1)
searcher = LOPQSearcher(m)
train_codes = np.load('lopq_c9.txt.npy')
searcher.add_codes(train_codes)

print "success"
