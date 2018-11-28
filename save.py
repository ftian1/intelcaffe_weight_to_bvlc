import os
import sys
import argparse

import cPickle as pickle
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='convert intelcaffe model to be compatible with bvlc caffe: ')
    parser.add_argument('-r', '--caffe_root', dest='caffe_root',
                        help='the path to the bvlc caffe root directory',
                        default=None, type=str, required=True)
    parser.add_argument('-p', '--prototxt', dest='prototxt',
                        help='the path to the train prototxt',
                        default=None, type=str, required=True)
    parser.add_argument('-w', '--weight', dest='weight',
                        help='the path to the corresponding weight generated from the train prototxt with intel caffe',
                        default=None, type=str, required=True)
    parser.add_argument('-o', '--output', dest='output',
                        help='the path to the converted weight binary file which is compatible with bvlc caffe',
                        default=None, type=str, required=True)

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
    
    with open(args.weight, 'rb') as f:
        weights = pickle.load(f)
    net = caffe.Net(args.prototxt, caffe.TEST)

    for i in range(len(weights)):
        name = weights[i][0]
        weight = weights[i][1]
        #assert name in net.params, name + 'param does not exist in bvlc caffe'
        for j in range(len(net._layer_names)):
            if net._layer_names[j] != name:
                continue
            if len(net.params[name]) != len(weight):
                print name, net.layers[j].type, net.layers[j + 1].type, len(net.params[name]), len(weight)
                assert net.layers[j].type == 'BatchNorm' and net.layers[j + 1].type == 'Scale', 'batch_norm layer is not following a scale layer, it\'s wrong for pure caffe engine'
                assert len(net.params[name]) == 3 and len(net.params[net._layer_names[j + 1]]) == 2, 'batch_norm layer should have 3 learnable parameters, scale layer should have 2 learnable parameters'
                for m in range(len(net.params[name])):
                    net.params[name][m].data = weight[m]
                for n in range(len(net.params[net._layer_names[j + 1]])):
                    net.params[net._layer_names[j + 1]][n].data = weight[3 + n]
                continue
            for k in range(len(net.params[name])):
                print name, k
                net.params[name][k].data[...] = weight[k][...]
    
    net.save(args.output)
    
