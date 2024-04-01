## Building a Biomedical Entity Linker with LLMs [[Blog Article](https://towardsdatascience.com/building-a-biomedical-entity-linker-with-llms-d385cb85c15a)]
This directory contains the code and resources for running the experiments described in the blog article.

## Dataset
I've used the [BC5CDR dataset](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/) for evaluating the LLMs and fine-tuning the model for entity extraction. The original dataset is distributed under the license mentioned in this [file](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/README.txt). Please cite the original dataset and papers if you use this dataset as part of your research work:

1. Wei CH, Peng Y, Leaman R, Davis AP, Mattingly CJ, Li J, Wiegers TC, Lu Z. Overview of the BioCreative V Chemical Disease Relation (CDR) Task, Proceedings of the Fifth BioCreative Challenge Evaluation Workshop, p154-166, 2015 

2. Li J, Sun Y, Johnson RJ, Sciaky D, Wei CH, Leaman R, Davis AP, Mattingly CJ, Wiegers TC, Lu Z. Anotating chemicals, diseases and their interactions in biomedical literature, Proceedings of the Fifth BioCreative Challenge Evaluation Workshop, p173-182, 2015 

3. Leaman R, Dogan RI, Lu Z. DNorm: disease name normalization with pairwise learning to rank, Bioinformatics 29(22):2909-17, 2013
 
4. Leaman R, Wei CH, Lu Z. tmChem: a high performance approach for chemical named entity recognition and normalization. J Cheminform, 7:S3, 2015

along with:

5. Jiao Li, Yueping Sun, Robin J. Johnson, Daniela Sciaky, Chih-Hsuan Wei, Robert Leaman, Allan Peter Davis, Carolyn J. Mattingly, Thomas C. Wiegers, Zhiyong Lu, BioCreative V CDR task corpus: a resource for chemical disease relation extraction, Database, Volume 2016, 2016, baw068, https://doi.org/10.1093/database/baw068
