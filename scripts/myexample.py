# Copyright 2016, YitongFeng, CASIA.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import sys
import os
import struct
import time
import logging

# Add the lopq module - not needed if they are available in the python environment
sys.path.append(os.path.abspath('../python'))

import numpy as np
from sklearn.cross_validation import train_test_split

from python.lopq import LOPQModel, LOPQSearcher
from python.lopq.eval import compute_all_neighbors, get_recall
from python.lopq.model import eigenvalue_allocation


def load_oxford_data():
    from python.lopq.utils import load_xvecs
    data = load_xvecs('../data/oxford/oxford_features.fvecs')
    return data

def load_data(feature_file):
    data = {}

    # with open(path_file) as pf:
    #     file_paths = [i.strip() for i in pf.readlines()]
    with open(feature_file, 'rb') as f:
        num = np.uint32(struct.unpack('i', f.read(4)) )[0]
        fea_dim = np.uint32(struct.unpack('i', f.read(4)))[0]
        num = 65968
        A = np.zeros((num, fea_dim), 'float32')
        for i in xrange(num):
            print i
            for j in xrange(fea_dim):
                A[i, j] = np.float32(struct.unpack('f', f.read(4)))[0]
            data[i] = A[i]
    return A

# def load_data(feature_file, name_file, index_file):
#     names = []
#
#     with open(name_file) as nf:
#         file_names = [i.strip() for i in nf.readlines()]
#     with open(index_file) as indf:
#         indices = [i.strip() for i in indf.readlines()]
#         indices = [int(i) - 1 for i in indices]
#
#     with open(feature_file, 'rb') as f:
#         num = np.uint32(struct.unpack('i', f.read(4)) )[0]
#         fea_dim = np.uint32(struct.unpack('i', f.read(4)))[0]
#         A = np.zeros((num, fea_dim), 'float32')
#         for i in xrange(num):
#             print i
#             for j in xrange(fea_dim):
#                 A[i, j] = np.float32(struct.unpack('f', f.read(4)))[0]
#
#     return A, file_names, indices

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

def load_gt(gt_file):
    data = {}
    with open(gt_file) as f:
        d = [i.strip() for i in f.readlines()]
        d = [i.split(':') for i in d]
    for i in d:
        if data.has_key(i[0].strip()):
            data[i[0].strip()] += i[1].strip().split()
        else:
            data[i[0].strip()] = i[1].strip().split()
    for i, j in data.items():
        data[i] = set(j)

    return data

def pca(data):
    """
    A simple PCA implementation that demonstrates how eigenvalue allocation
    is used to permute dimensions in order to balance the variance across
    subvectors. There are plenty of PCA implementations elsewhere. What is
    important is that the eigenvalues can be used to compute a variance-balancing
    dimension permutation.
    """

    # Compute mean
    count, D = data.shape
    mu = data.sum(axis=0) / float(count)

    # Compute covariance
    summed_covar = reduce(lambda acc, x: acc + np.outer(x, x), data, np.zeros((D, D)))
    A = summed_covar / (count - 1) - np.outer(mu, mu)

    # Compute eigen decomposition
    eigenvalues, P = np.linalg.eigh(A)

    # Compute a permutation of dimensions to balance variance among 2 subvectors
    permuted_inds = eigenvalue_allocation(2, eigenvalues)

    # Build the permutation into the rotation matrix. One can alternately keep
    # these steps separate, rotating and then permuting, if desired.
    P = P[:, permuted_inds]

    return P, mu

def main():
    """
    A brief demo script showing how to train various LOPQ models with brief
    discussion of trade offs.
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='test.log',
                        filemode='w')

    # Get the oxford dataset
    # data = load_oxford_data()
    train, train_names = load_data('../../Data/FV/codefile_sample.bin', '../../Data/FV/names_sample.txt')
    ground_truth = load_gt('../../Data/FV/gt_file.txt')
    test, test_names = load_data('../../Data/FV/codefile_sample.bin', '../../Data/FV/names_sample.txt')
    # Compute PCA of oxford dataset. See README in data/oxford for details
    # about this dataset.
    # P, mu = pca(data)

    # Mean center and rotate the data; includes dimension permutation.
    # It is worthwhile see how this affects recall performance. On this
    # dataset, which is already PCA'd from higher dimensional features,
    # this additional step to variance balance the dimensions typically
    # improves recall@1 by 3-5%. The benefit can be much greater depending
    # on the dataset.
    # data = data - mu
    # data = np.dot(data, P)

    # Create a train and test split. The test split will become
    # a set of queries for which we will compute the true nearest neighbors.
    # train, test = train_test_split(data, test_size=1.0)


    # Compute distance-sorted neighbors in training set for each point in test set.
    # These will be our groundtruth for recall evaluation.
    # nns = compute_all_neighbors(test, train)

    # Fit model
    m = LOPQModel(V=16, M=8)
    m.fit(train, n_init=1)

    # Note that we didn't specify a random seed for fitting the model, so different
    # runs will be different. You may also see a warning that some local projections
    # can't be estimated because too few points fall in a cluster. This is ok for the
    # purposes of this demo, but you might want to avoid this by increasing the amount
    # of training data or decreasing the number of clusters (the V hyperparameter).

    # With a model in hand, we can test it's recall. We populate a LOPQSearcher
    # instance with data and get recall stats. By default, we will retrieve 1000
    # ranked results for each query vector for recall evaluation.
    searcher = LOPQSearcher(m)
    searcher.add_data(train)

    pic = test_names[0]
    results_total = {}
    results_one_img = []
    result_quota = 20

    # query process
    # Todo: Code need optimization
    start = time.clock()
    for i in xrange(len(test)):
        result, _ = searcher.search(test[i], result_quota * 2, with_dists=True)
        if test_names[i] != pic:
            results_one_img.sort(key=lambda x : x[2])
            results_one_img_new = []
            for rr in results_one_img:
                if len(results_one_img_new) == result_quota:
                    break
                img_name = train_names[rr[0]]
                if img_name not in results_one_img_new:
                    results_one_img_new.append(img_name)
            results_total[pic] = results_one_img_new
            results_one_img = []
            pic = test_names[i]
        results_one_img += result

    if(len(results_one_img) > 0):
        results_one_img.sort(key=lambda x: x[2])
        results_one_img_new = []
        for rr in results_one_img:
            if len(results_one_img_new) == result_quota:
                break
            img_name = train_names[rr[0]]
            if img_name not in results_one_img_new:
                results_one_img_new.append(img_name)
        results_total[pic] = results_one_img_new

    query_time = time.clock() - start
    logging.info("query time is %f", query_time)
    # calc recall
    total_recall = 0
    for q, res in results_total.items():
        recall_i = 0
        if ground_truth.has_key(q):
            gts = ground_truth[q]
        else: continue
        for g in gts:
            if g in res:
                recall_i += 1
        total_recall += recall_i * 1.0 / len(gts)
    mean_recall = total_recall*1.0 / len(results_total)
    # recall, _ = get_recall(searcher, test, nns)
    print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m.V, m.M, m.subquantizer_clusters, str(mean_recall))
    logging.info('Total Recall is: %s, query_num is: %d' % (str(total_recall), len(results_total)))
    logging.info('Recall (V=%d, M=%d, subquants=%d): %s' % (m.V, m.M, m.subquantizer_clusters, str(mean_recall)))
    # We can experiment with other hyperparameters without discarding all
    # parameters everytime. Here we train a new model that uses the same coarse
    # quantizers but a higher number of subquantizers, i.e. we increase M.
    # m2 = LOPQModel(V=16, M=16, parameters=(m.Cs, None, None, None))
    # m2.fit(train, n_init=1)
    #
    # # Let's evaluate again.
    # searcher = LOPQSearcher(m2)
    # searcher.add_data(train)
    # recall, _ = get_recall(searcher, test, nns)
    # print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m2.V, m2.M, m2.subquantizer_clusters, str(recall))
    #
    # # The recall is probably higher. We got better recall with a finer quantization
    # # at the expense of more data required for index items.
    #
    # # We can also hold both coarse quantizers and rotations fixed and see what
    # # increasing the number of subquantizer clusters does to performance.
    # m3 = LOPQModel(V=16, M=8, subquantizer_clusters=512, parameters=(m.Cs, m.Rs, m.mus, None))
    # m3.fit(train, n_init=1)
    #
    # searcher = LOPQSearcher(m3)
    # searcher.add_data(train)
    # recall, _ = get_recall(searcher, test, nns)
    # print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m3.V, m3.M, m3.subquantizer_clusters, str(recall))

    # The recall is probably better than the first but worse than the second. We increased recall
    # only a little by increasing the number of model parameters (double the subquatizer centroids),
    # the index storage requirement (another bit for each fine code), and distance computation time
    # (double the subquantizer centroids).


if __name__ == '__main__':
    main()
