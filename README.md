## Improving sequence-based modeling of protein families using secondary structure quality assessment

**Motivation** 
These last years, major breakthroughs have been made in Machine Learning regarding generative models 
able to generate new data with the same statistics as a given training set. These technics have led 
to conclusive results with a variety of data such as images, audios or texts. Following these 
successes, data-driven design of new varieties of proteins has been a field of growing interest. 
Until now, most attempts have relied only on the sequences of proteins. In particular,
Direct Coupling Analysis models drove a lot of attention for their ability to model 
important correlation in protein sequences. Building on these models, we introduce a new 
framework to assess the quality of a secondary structure for a better use of structural 
information in protein design.

**Results**
 Working with previously established experimental data, we showed improvement in the detection of 
non-functional generated sequences with our two frameworks named Dot Product and Pattern Matching.
Our results have been established on previous data-driven protein design work relying on Direct
Coupling Analysis and on previous mutational effects datasets.

Take a look at our [paper]()

In this repository, we provide :
- Scripts to format data from FASTA file in `data/` as well as all necessary data structure
- Scripts for performing the diverse Secondary Structure Quality Assessment in `ssqa/` and generative
  algorithms taking into account SSQA in `generation/`
- Implementation of RBM, adaptable to more complex Markov Random Fields model in `pgm/`
- Tutorials and Notebooks to get the results of the paper in `notebooks/`

To use these scripts please install all required Python libraries from `requirements.txt` as well as
[MMSEQS](https://github.com/soedinglab/MMseqs2) and [HH-suite](https://github.com/soedinglab/hh-suite). 
Also, download the working [data]() and add yours easily following the tutorials. Once done, update
the `config.py` with your own absolute path to your data folder to avoid any bug.

If you have any question please feel free to contact me at [cyril.malbranke@ens.fr](mailto:cyril.malbranke@ens.fr)