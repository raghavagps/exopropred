# **ExoProPred**
A computational approach to predict the subcellular localisation of exosomal proteins using the sequence information of the proteins.
## Introduction
ExoProPred is a webserver to predict exosomal proteins based on hybrid model that combines machine learning model with motif-search approach. The models are trained on a dataset comprising of 2831 exosomal proteins and 2831 non-exosomal proteins. The performance of the models were evaluated using 5-fold cross-validation. The models were trained on top 70 best features comprising of composition-based and evolutionary information based features as well as on hybrid features(Top 70 features + Motif-search) by implementing random-forest classifier from the scikit library of python. In the standalone version, random-forerst classifier based model is implemented alongwith the motif-search usinf MERCI tool, named it as hybrid approach.
ExoProPred is also available as web-server at https://webs.iiitd.edu.in/raghava/exopropred. Please read/cite the content about the ExoProPred for complete information including algorithm behind the approach.
## Reference
Arora A, Patiyal S, Sharma N, Devi NL, Kaur D, Raghava GPS. A random forest model for predicting exosomal proteins using evolutionary information and motifs. <a href="https://pubmed.ncbi.nlm.nih.gov/37525341/">Proteomics. 2023 Jul 31:e2300231. doi: 10.1002/pmic.202300231. Epub ahead of print. PMID: 37525341. </a>

## PIP Installation
PIP version is also available for easy installation and usage of this tool. The following command is required to install the package 
```
pip install exopropred
```
To know about the available option for the pip package, type the following command:
```
exopropred -h
```
## Standalone
The Standalone version of transfacpred is written in python3 and following libraries are necessary for the successful run:
- scikit-learn
- Pandas
- Numpy

## Minimum USAGE
To know about the available option for the stanadlone, type the following command:
```
python3 exopropred.py -h
```
To run the example, type the following command:
```
python3 exopropred.py -i example_input.fa
```
This will predict if the submitted sequences are exososomal proteins or non-exososomal proteins. It will use other parameters by default. It will save the output in "outfile.csv" in CSV (comma seperated variables).

## Full Usage
```
usage: exopropred.py [-h] -i INPUT [-o OUTPUT] [-m {1,2}] [-t THRESHOLD]
                     [-d {1,2}]
```
```
Please provide following arguments

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence(s) in FASTA format
                        or single sequence per line in single letter code
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -m {1,2}, --model {1,2}
                        Model Type: 1: Composition based model, 2: Hybrid
                        Model, by default 1
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold: Value between 0 to 1 by default 0.51
  -d {1,2}, --display {1,2}
                        Display: 1:Exosomal Proteins only, 2: All Proteins, by
                        default 1
```

**Input File:** It allow users to provide input in the FASTA format.

**Output File:** Program will save the results in the CSV format, in case user do not provide output file name, it will be stored in "outfile.csv".

**Threshold:** User should provide threshold between 0 and 1, by default its 0.51.

**Model:** User is allowed to choose between two different models, such as, 1 for composition-based model, 2 for hybrid model, by default its 1.

**Display type:** This option allow users to fetch either only exososomal proteins by choosing option 1 or prediction against all proteins by choosing option 2.

ExoProPred Package Files
=========================
It contantain following files, brief descript of these files given below

INSTALLATION                              : Installations instructions

LICENSE                                   : License information

README.md                                 : This file provide information about this package

model.zip                                 : This zipped file contains the compressed version of model

envfile                                   : This file compeises of paths for the PSI-BLAST, MERCI_motif_locator.pl, Motifs, and Swiss-Prot database.

exopropred.py                             : Main python program

MERCI_motif_locator.pl                    : Perl script for locating motifs using MERCI

swissprot                                 : Swiss-Prot database for calculating PSSM profile

motifs                                    : Folder containing the motif files

src                                       : Folder containing the python scripts for PSSM based composition features

Data                                      : Folder containing the files to calculate the features using Pfeature

example_input.fa                          : Example file contain peptide sequenaces in FASTA format

example_composition_model_output.csv      : Example output file for composition-based model

example_hybrid_model_output.csv           : Example output file for hybrid model
