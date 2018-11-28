import os
import sys
import argparse

import cPickle as pickle
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='saving intelcaffe model to file: ')
    parser.add_argument('-r', '--caffe_root', dest='caffe_root',
                        help='the path to the intel caffe root directory',
                        default=None, type=str, required=True)
    parser.add_argument('-p', '--prototxt', dest='prototxt',
                        help='the path to the train prototxt',
                        default=None, type=str, required=True)
    parser.add_argument('-w', '--weight', dest='weight',
                        help='the path to the corresponding weight generated from the train prototxt with intel caffe',
                        default=None, type=str, required=True)
    parser.add_argument('-o', '--output', dest='output',
                        help='the path to the saved weight binary file',
                        default=None, type=str, required=True)

    #print len(sys.argv)

    #if len(sys.argv) != 5:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()

    print args.caffe_root
    print args.prototxt
    print args.weight

    if not os.path.exists(args.caffe_root) or not os.path.exists(args.prototxt) or not os.path.exists(args.weight):
        parser.print_help()
        sys.exit(1)

    return args

if __name__ == '__main__':
    args = parse_args()

    sys.path.insert(0, args.caffe_root + '/python')
    import caffe
    
    weights = []
    net = caffe.Net(args.prototxt, args.weight, caffe.TRAIN)
    for param in net.params:
        data = []
        for i in range(len(net.params[param])):
            data.append(net.params[param][i].data)
        weights.append([param, data])
    
    with open(args.output, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)
