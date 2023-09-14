# GPU_benchmarking_toolkit_for_ML
Collection of Machine Learning algorithms for providing benchmarks of GPU hardware.

Abstract
========

The rapid evolution of machine learning (ML) and artificial intelligence (AI) has spurred the demand for advanced computing tools to meet researchers’ evolving needs. ARCC benchmarking toolkit, aims to equip researchers with crucial resources to enhance their ML/AI projects and expand their domains. Our toolkit, designed for both bare metal and Kubernetes-based HPC clusters, offers diverse ML/AI workloads, ensuring scalability and providing GPU performance insights. It displays system performance through intuitive graphs, employing advanced stats. Metrics like CPU utilization, Disk IOPS, and GPU Usage aid task assessment and bottleneck identification. These metrics deepen understanding of computing environment performance and ML/AI task efficiency, enhancing researchers’ insights and aiding optimization.

The rapid advancement of machine learning (ML) and artificial intelligence (AI) has sparked a demand for sophisticated
computing tools to meet the evolving needs of researchers. In response, we have embarked on a comprehensive ARCC
benchmarking toolkit project, aiming to equip researchers with essential resources and tools to enhance their ML/AI
projects and push the boundaries of what is achievable in their respective fields. Existing AI benchmarks, such as MLPerf [ 13], suffer from limited scalability due to fixed problem sizes, while others like AIPerf [14 ] focus on scalability but lack specificity in the ML/AI models used. Additionally, some toolkits, like iMLBench [ 15], are designed for CPU-GPU integrated architectures, but they have limited workloads and do not emphasize specific ML/AI models. Other commonly used toolkits like Rodina benchmark [3], Mixbench [ 8]are not built with the focus on AI/ML models. Our benchmarking toolkit is designed explicitly for ML/AI algorithms with a specific emphasis on both bare metal (ex: SLURM workload manager) and Kubernetes-based HPC clusters. It offers a diverse range of ML/AI workloads and ensures scalability across different computing platforms, providing researchers with valuable insights into GPU performance under varying scenarios.

As the ARCC benchmarking toolkit is designed to be compatible with both bare metal (e.g., SLURM workload manager)
and Kubernetes-based high-performance computing (HPC) clusters, accommodating the dataset organization structure
of open-source repositories like HuggingFace. It covers a wide range of GPUs, including A10, A30, A40, A100, RTX
A6000, and V100-32gb, commonly used in ML/AI workloads and representing the state-of-the-art in CUDA-based
computing hardware. Furthermore, The toolkit encompasses a variety of ML/AI methods and algorithms, such as
natural language processing algorithms (BERT [5], GPT-2 [12], DNABERT2[16]), image recognition and classification
algorithms (YOLOV8[1]), text-to-speech algorithms (FastPitch[17], Coqui-ai TTS [6], Deep Voice 3 [11]), and text-to-image conversion. Notably, a wide variety of Datasets are supported for LLMs given the numerous standard Corpus
datasets; OpenWebtext [2], ThePile[7], Red Pajamma [4], Oscar [10], and Starcoder [9]. This diversity allows an in-depth understanding of each CPU/GPU’s performance under different ML/AI workloads. A crucial aspect of the ARCC benchmarking toolkit is its efficient portrayal of system performance data through auto-generated, user-intuitive graphical representations. The toolkit employs advanced statistical analysis to convert raw data into understandable, actionable information. Critical computational metrics such as Central Processing Unit (CPU) utilization, Disk Input/Output Operations Per Second (IOPS), Graphics Processing Unit (GPU) power, and core
utilization are included in the comprehensive data presentation. Each of these metrics provides a unique insight into
the performance of the computing environment and the efficiency of the ML/AI tasks being run. By visualizing these
metrics, the benchmarking tools provide researchers with a clear and intuitive understanding of how their tasks are
performing and where potential bottlenecks may lie in the development pipeline. This understanding can be invaluable
in developing proper performance expectations for various model types and in optimizing the performance of their
tasks. Additionally, collating and analyzing the aforementioned, our benchmarking suite provides a comprehensive
and objective understanding of each GPU’s performance characteristics enabling students and researchers to make
well-informed decisions when selecting hardware for their specific ML/AI tasks, ensuring optimal utilization of the
computing resources.
For instance, when examining the Disk IOP performance for GPT-style models, a notable observation emerged during
the Docker Container-based test of GPT2. This model was chosen because it represents a modern early development
unoptimized Language Model (LLM). During the fine-tuning process of GPT2 for three epochs on an NVIDIA A100
GPU, utilizing the OpenWebtext dataset[2], a consistent decrease in IOPS of approximately 13 percent was identified
between each epoch. This observation sheds light on potential bottlenecks within the Data loading pipeline, providing
valuable insights that can be used to optimize future workloads.
By empowering researchers with transparent and accurate performance data, the benchmarking suite assists in informed
GPU selection for specific algorithms, resulting in influential research outcomes. This initiative reflects our dedication
to supporting and advancing ML/AI research, equipping researchers with the necessary tools to drive innovation in the
field.

Acknowledgments
===============
We would like to thank UW REDD for supporting the ARCC Internship program, The ARCC Infrastructure Team for
their technical support, The University of Wyoming School of Computing, and the National Research Platform for
allowing us to conduct research on the Nautilus Cluster

## References
[1] [n. d.].

[2] Ellie Pavlick Stefanie Tellex Aaron Gokaslan, Vanya Cohen. 2019. OpenWebText Corpus. (2019). http://Skylion007.github.io/OpenWebTextCorpus

[3] Shuai Che, Michael Boyer, Jiayuan Meng, David Tarjan, Jeremy W. Sheaffer, Sang-Ha Lee, and Kevin Skadron. 2009. Rodinia: A benchmark suite for
heterogeneous computing. In 2009 IEEE International Symposium on Workload Characterization (IISWC). 44–54. https://doi.org/10.1109/IISWC.2009.
5306797

[4] Together Computer. 2023. RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset. https://github.com/togethercomputer/RedPajama-
Data

[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding. CoRR abs/1810.04805 (2018). arXiv:1810.04805 http://arxiv.org/abs/1810.04805
 
[6] Gölge Eren and The Coqui TTS Team. 2021. Coqui TTS. https://doi.org/10.5281/zenodo.6334862
 
[7] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al.
2020. The Pile: An 800GB dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027 (2020).
 
[8] Elias Konstantinidis and Yiannis Cotronis. 2017. A quantitative roofline model for GPU kernel performance estimation using micro-benchmarks
and hardware metric profiling. J. Parallel and Distrib. Comput. 107 (2017), 37–56. https://doi.org/10.1016/j.jpdc.2017.04.002
 
[9] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny
Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro,
Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham
Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour
Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero,
Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson,
Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz
Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023. StarCoder: may the source be with you! (2023).
arXiv:2305.06161 [cs.CL]
 
[10] Pedro Javier Ortiz Su’arez, Laurent Romary, and Benoit Sagot. 2020. A Monolingual Approach to Contextualized Word Embeddings for Mid-Resource
Languages. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics,
Online, 1703–1714. https://www.aclweb.org/anthology/2020.acl-main.156
 
[11] Wei Ping, Kainan Peng, Andrew Gibiansky, Sercan O. Arik, Ajay Kannan, Sharan Narang, Jonathan Raiman, and John Miller. 2018. Deep Voice 3:
Scaling Text-to-Speech with Convolutional Sequence Learning. arXiv:1710.07654 [cs.SD]
 
[12] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners.
(2019).
 
[13] Vijay Janapa Reddi, Christine Cheng, David Kanter, Peter Mattson, Guenther Schmuelling, Carole-Jean Wu, Brian Anderson, Maximilien Breughe,
Mark Charlebois, William Chou, Ramesh Chukka, Cody Coleman, Sam Davis, Pan Deng, Greg Diamos, Jared Duke, Dave Fick, J. Scott Gardner, Itay
Hubara, Sachin Idgunji, Thomas B. Jablin, Jeff Jiao, Tom St. John, Pankaj Kanwar, David Lee, Jeffery Liao, Anton Lokhmotov, Francisco Massa, Peng
Meng, Paulius Micikevicius, Colin Osborne, Gennady Pekhimenko, Arun Tejusve Raghunath Rajan, Dilip Sequeira, Ashish Sirasao, Fei Sun, Hanlin
Tang, Michael Thomson, Frank Wei, Ephrem Wu, Lingjie Xu, Koichi Yamada, Bing Yu, George Yuan, Aaron Zhong, Peizhao Zhang, and Yuchen
Zhou. 2020. MLPerf Inference Benchmark. arXiv:1911.02549
 
[14] Zhixiang Ren, Yongheng Liu, Tianhui Shi, Lei Xie, Yue Zhou, Jidong Zhai, Youhui Zhang, Yunquan Zhang, and Wenguang Chen. 2021. AIPerf:
Automated machine learning as an AI-HPC benchmark. Big Data Mining and Analytics 4, 3 (2021), 208–220. https://doi.org/10.26599/BDMA.2021.9020004
 
[15] Chenyang Zhang, Feng Zhang, Xiaoguang Guo, Bingsheng He, Xiao Zhang, and Xiaoyong Du. 2021. iMLBench: A Machine Learning Benchmark
Suite for CPU-GPU Integrated Architectures. IEEE Transactions on Parallel and Distributed Systems 32, 7 (2021), 1740–1752. https://doi.org/10.1109/
TPDS.2020.3046870
 
[16] Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana Davuluri, and Han Liu. 2023. DNABERT-2: Efficient Foundation Model and Benchmark
For Multi-Species Genome. arXiv:2306.15006 [q-bio.GN]
 
[17] Adrian Łańcucki. 2021. FastPitch: Parallel Text-to-speech with Pitch Prediction. arXiv preprint arXiv:2006.06873 (2021). https://doi.org/10.48550/
arXiv.2006.06873
 
