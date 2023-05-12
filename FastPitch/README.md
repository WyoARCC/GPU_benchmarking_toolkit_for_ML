# FastPitch 1.1 for PyTorch

Most of the files in this folder are from [Nvidia’s Deep Learning Examples repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch). It was cloned from there on 2023/03/20. Additional files have been added to this repository to create a Singularity container that can be used to train the FastPitch model and helper scripts to run and log the outputs of training the model when launching the Docker container in a Kubernetes environment. For the specifics of how this implementation of the FastPitch model works please refer to Nvidia’s repository. The goal of this repository is to provide a framework for benchmarking the performance of GPU hardware by training Machine Learning algorithms.

## Table Of Contents
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Release notes

### Changelog
2023/05/12
- Updated to use ARCC's fork of [memprof.](https://github.com/WyoARCC/memprof)
- Updated to use a shared dataset with other TTS benchmarks in this repo.

2023/04/07:
- Added support for tracking resource utilization using [memprof](https://github.com/IGBIllinois/memprof) and [gpustat](https://github.com/wookayin/gpustat).

2023/03/30:
- Updates to default values to better suit the environments we are looking to benchmark.

2023/03/28:
- Initial commit

### Known issues

There are no known issues in this release.
