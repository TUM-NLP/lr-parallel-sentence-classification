# Parallel Sentence Filtering for Low-Resource Language Pairs: A Case Study for Upper Sorbian, German, and Czech

## Description
This repository contains the core scripts used in the thesis:  
**Multilingual Embedding-Based Sentence Pair Analysis in Low-Resource Languages**.

This repository presents the dataset and code from our LREC 2026 article *Parallel Sentence Filtering for Low-Resource Language Pairs: A Case Study for Upper Sorbian, German, and Czech* (Jiang et al., 2026). 
The work stems from Ruiyang Jiang's bachelor's thesis (dataset, code, and experiments): **Multilingual Embedding-Based Sentence Pair Analysis in Low-Resource Languages**.


## Files
Below are the files in the `script` folder:
- `main.py` - experiment pipeline
- `sentence_embedding.py` — encode sentences with multilingual models (XLM-R, Glot500m, LaBSE)  
- `sentence_transformation.py` — apply transformations (negation, antonym, etc.)  
- `cbie_main.py` — run experiments with CBIE  
- `cbie_transformation.py` — CBIE-related preprocessing  
- `cbie_visual.py` — visualization of embedding spaces  
- `evaluation.py` — compute classification metrics (accuracy, precision, recall, F1, etc.)  
- `plot_curve.py` — plotting curves (e.g., ROC, PR)
- `prefiltering.py`- dataset pre-processing pipeline
- `antonyms_de.json` — resource file for antonym-based transformations  



## Acknowledgements
This work was funded by the European Research Council (ERC) under grant agreements No. 101113091 - Data4ML (a Proof of Concept Grant) and No. 101141712 - EPICAL.