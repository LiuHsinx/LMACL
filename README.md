# LMACL
This is the PyTorch implementation for LMACL proposed in the paper LMACL: Improving Graph Collaborative Filtering with Learnable Model Augmentation Contrastive Learning.

### 1. Note on datasets and directories
Due to the large size of datasets  *Amazon* and *Tmall*, we have compressed them into zip files. Please unzip them before running the model on these datasets. For *Gowalla*,  keeping the current directory structure is fine.

Before running the codes, please ensure that two directories `log/` and `saved_model/` are created under the root directory. They are used to store the training results and the saved model and optimizer states.

### 2. Running environment

We develope our codes in the following environment:

```
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```

### 3. How to run the codes

* Tmall
```
python main.py --data tmall --temp 0.15
```

* Gowalla

```
python main.py --data gowalla --epoch 150
```

* Amazon

```
python main.py --data amazon --num-heads 8 --gnn_layer 3
```

### 4. Some configurable arguments

* `--cuda` specifies which GPU to run on if there are more than one.
* `--data` selects the dataset to use.
* `--lambda1` specifies $\lambda_1$, the regularization weight for CL loss.
* `--lambda2` is $\lambda_2$, the L2 regularization weight.
* `--temp` specifies $\tau$, the temperature in CL loss.
* `--num-heads` is the number of hidden attention heads.



