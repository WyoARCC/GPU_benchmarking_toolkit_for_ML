# Coqui-ai TTS

Most of the files in this folder are from [coqui-ai's TTS repository](https://github.com/coqui-ai/TTS). It was cloned from there on 2023/03/20. Additional files have been added to this repository to create a Singularity container that can be used to train the TTS and VOCODER models, an alternative Dockerfile, and helper scripts to run and log the outputs of training the model when launching the Docker container in a Kubernetes environment. For the specifics of how the implementation of the TTS and VOCODER models work please refer to Coqui-ai’s repository. The goal of this repository is to provide a framework for benchmarking the performance of GPU hardware by training Machine Learning algorithms. 

## Table Of Contents
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Release notes

### Changelog
2023/05/15
- Added additional headings to run_test.sh to help with searching through the output.
- Updated run_test.sh to stop incase of a step failing.
 
2023/05/12
- Updated to use ARCC's fork of [memprof.](https://github.com/WyoARCC/memprof)
- Added additional training models.
- Updated to use a shared dataset with other TTS benchmarks in this repo.

2023/04/07
- Added support for tracking resource utilization using [memprof](https://github.com/IGBIllinois/memprof) and [gpustat](https://github.com/wookayin/gpustat). 

2023/03/30:
- Initial commit.

### Known issues

There are no known issues in this release.
