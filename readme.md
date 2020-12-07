# Overview

This project is to review and implement an unsupervised representation learning algorithm called Deep InfoMax (DIM), proposed in [Learning deep representations by mutual information estimation and maximization](https://arxiv.org/abs/1808.06670).


## Requirements
The code is written in Python 3.7 and Pytorch 1.7


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the rest of the requirements.
```bash
pip install -r requirements.txt
```

## Folder Structure
Experiment/
|
|-- data/ -default directory for storing datasets
|-- output/ - default directory for storing trained DIM models
|-- task_result/ - default directory for storing downstream tasks
|-- clustering.py - script to run the clustering task
|-- evaluation.py - script to evaluate trained classifiers
|-- train_classifier.py - script to run image classification task
|-- train_DIM.py - script to train the DIM
|-- config.py -- helper functions
|-- data_loader.py
|-- loss_func.py 
|-- models.py
|-- requirements.txt 
|-- readme.md



## Example usage
Experiments can be run following the steps below:

1. Train DIM encoder
2. Run downstream task(s) using the trained encoder
    - Downstream task 1: Image classification
    - Downstream task 2: Image clustering




### Train DIM

    ```python
    python train_DIM.py -e 500 -a 0.5 -b 1 -g 0.1 -lr 1e-5
    ```
Training on a 4-GPU machine takes about 1min30s/epoch.

### Downstream task 1: Image classification: 
1. Train classifiers: 
        ```python
        python train_classifier -t 2 -p A0_B1.0_G0.1_1205_0145 -c 500 -e 200)
    
        ```
    
2. Evaluate classifiers:
        ```python
        python evaluation.py -p conv1203_2236 -c 200 -l Y
        ```
    

### Downstream task 2: Image clustering: 

    ```python
    python clustering.py -n 4 -k 10 -p1 A0.5_B1.0_G0.1_1202_1059 -p2 A0_B1.0_G0.1_1202_1059
    ```
This script can compare up to to DIM encoders by giving the folder name of where the checkpoints are store. e.g. 'A0.5_B1.0_G0.1_1202_1059'. 


## Note

To view the full set of commands, try:
    ```python
    python train_DIM.py --help
    ```
    
## Code Reference
All codes are implemented on the author's own after reading and understanding the [original Pytorch implementation](https://github.com/rdevon/DIM) provided by the paper authors as well as [Duane Nielsen's implementation](https://github.com/DuaneNielsen/DeepInfomaxPytorch).