# intelcaffe_weight_to_bvlc
This is used to convert intelcaffe weight file to be compatible with bvlc caffe

steps:
1. dump weight binary from intelcaffe

   - python dump.py -r /path/to/intelcaffe/root -p /path/to/train/prototxt -w /path/to/weight -o /path/to/dump/output

2. save weight to the one being compatible with bvlc caffe

   - python save.py -r /path/to/bvlccaffe/root -p /path/to/inference/prototxt -w /path/to/dump/output -o /path/to/final/bvlc/weight

NOTE:
1. The prototxt used in dump phase should be the one used to train in IntelCaffe. The prototxt used in save phase should be the one used to inference in BVLC Caffe.
2. User have to mannually remove those unsupported fields from prototxt in BVLC, such as bn_stats_batch_size and batch_norm_param.filler fields.
