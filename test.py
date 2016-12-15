import struct
import numpy as np
feature_file = r'D:\c_sift_fv\sift_fv\x64\9\indexfile_all_new.bin'

data = {}
with open(feature_file, 'rb') as f:
    # db_num = np.uint32(struct.unpack('i', f.read(4)) )[0]
    # db_fea_size = np.uint32(struct.unpack('i', f.read(4)))[0]
    num = 2201301
    fea_dim = 4480
    A = np.zeros((num, fea_dim), 'float')
    for i in xrange(num):
        print i
        for j in xrange(fea_dim):
            A[i, j] = np.float32(struct.unpack('f', f.read(4)))[0]
        data[i] = A[i]

print "success"