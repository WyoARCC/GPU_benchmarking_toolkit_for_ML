# SE(3)-Transformers For PyTorch

Most of the files in this folder are from [Nvidia’s Deep Learning Examples repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer). It was cloned from there on 2023/03/20. Additional files have been added to this repository to create a Singularity container that can be used to train the SE(3)-Transformer model and helper scripts to run and log the outputs of training the model when launching the Docker container in a Kubernetes environment.
For the specifics of how this implementation of the SE(3)-Transformer model works please refer to Nvidia’s repository. The goal of this repository is to provide a framework for benchmarking the performance of GPU hardware by training Machine Learning algorithms.

## Table Of Contents
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Release notes

### Changelog
2023/05/12
- Updated to use ARCC's fork of [memprof.](https://github.com/WyoARCC/memprof)

2023/04/07:
- Added support for tracking resource utilization using [memprof](https://github.com/IGBIllinois/memprof) and [gpustat](https://github.com/wookayin/gpustat).

2023/03/27:
- Initial commit

### Known issues

If you encounter `OSError: [Errno 12] Cannot allocate memory` during the Dataloader iterator creation (more precisely during the `fork()`, this is most likely due to the use of the `--precompute_bases` flag. If you cannot add more RAM or Swap to your machine, it is recommended to turn off bases precomputation by removing the `--precompute_bases` flag or using `--precompute_bases false`.
