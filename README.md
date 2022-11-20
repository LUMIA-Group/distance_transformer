# distance_transformer
The official implementation of the paper "Syntax-guided Localized Self-attention by Constituency Syntactic Distance"(findings of EMNLP 2022)

We provide the code for reproducing our model's result and data preprocessing.

## Requirements
You can directly download this code and run a requirement installation:
`pip install -r requirements.txt` on your conda environment.(our python version is 3.6)

Then it's necessary to run the setting up code `pip install --editable ./` and it's camera-ready now to run shell scripts in the `run` directory.

## Data Preparation
The pre-processing scripts for every dataset can be found in the corresponding `preprocessing/` folder, and rely on our code (`scripts/`). Take notice that scripts with `prepare` prefix aims to download, split and clean the dataset, while scripts with `preprocess` prefix aims to binarize the indexed dataset.Third-party software toolkits are automatically downloaded in the script.  

Totally six machine translation datasets are entailed as follows
- `IWSLT14 German to English`
- `IWSLT14 English to German`
- `NC11 German to English`
- `NC11 English to German`
- `ASPEC Chinese to Japanese`
- `WMT14 English to German`

For instance, if you want to prepare dataset iwslt14de2en/en2de, run the corresponding data preparation script

`cd preprocess`
`bash prepare-iwslt14.sh`

## Distance Preparation
To run our syntactic based model, the sytactic distance of sourcce language sentence must be firstly generated, and the scripts here could be directly run.

For instance, if you want to prepare syntactic distance of source language iwslt14de2en, which is German, run the corresponding distance preparation script(data preparation must be completed first)

`bash distance_iwslt_de2en.sh`

## Training and Evaluation
Scripts for training each model are provided in the folder `run/`. Each script is suffixed with corresponding task name including source language and target language.
Each script is binded with training and testing process altogether.
For running the script, enter the folder `run` first and use `bash` command. For example,

`cd run`

`bash train_iwslt_de2en.sh`
 
The final BLEU score for the test set will be logged into a .txt file.

## Description of this repository
- `data-bin/`<br>
  Contains the binarized dataset for the fairseq toolkit to read in. Dataset implementations are set to memory mapped.

- `distance_prior/`<br>
  Contains the calculated syntactic distance for every task. Only source language sentence is calculated. The syntactic distance is stored in the form of numpy files for each sentence pair.  
  
- `run/`<br>
  Shell scripts for training, validation and test.
  
- `log/`<br>
  Directory for storing the running result, including tensorboard log directory, saved checkpoint and training/test log.
  
- `preprocess/`<br>
  Shell scripts for training, validation and test.

- `fairseq/`  Model definition folder, crucial files are:
    - `models/distance_transformer.py` Define the overall architecture of Transformer.
    - `modules/transformer_layer_distance.py` Define the encoder layer and decoder layer respectively.
    - `modules/multihead_attention_distance.py` Define the multi-head attention guided by constituency based syntactic distance,
    - `models/distance_transformer.py`: Transformer baseline from [Vaswani et al. (NIPS'17)](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
    - `modules/transformer_layer_distance.py`Transformer baseline encoder and decoder layer.
    - `modules/multihead_attention_distance.py` Transformer baseline multi-head self-attention.
    - `criterions/labeL_smoothed_cross_entropy.py`
  
- `scripts/ tests/ helpers/ docs/ build/`<br>
  Other auxiliary folders for compilation and running.


