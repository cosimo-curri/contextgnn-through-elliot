# Rel-DeepLearning-RecSys

This is the official GitHub repository of the paper: *__ContextGNN goes to Elliot: Towards Benchmarking Relational
Deep Learning for Static Link Prediction (aka Personalized Item
Recommendation)__*, currently on arXiv.

The code contained in this repository allows to run rigorous and extensive **reproducibility** and **benchmarking** analyses with ContextGNN for static link prediction (aka personalized item recommendation) within Elliot, a popular framework for the reproducibility, benchmarking, and evaluation of recommender systems. 

While you may refer to the official GitHub repository for Elliot (https://github.com/sisinflab/elliot), here we used a reduced version of the framework adopted in this work:

- Paper: https://arxiv.org/abs/2308.00404
- GitHub repository: https://github.com/sisinflab/Graph-RSs-Reproducibility

### Disclaimer and acknowledgements
This repository has been forked from: https://github.com/kumo-ai/ContextGNN. Thus, a substantial portion of the code contained here is directly taken from that repository, and specifically from the branch: https://github.com/kumo-ai/ContextGNN/tree/xinwei_add_static_data_and_model_v1. Our purpose was to integrate the code of ContextGNN and make it available to run within Elliot (https://github.com/sisinflab/elliot).

We wish to thank the authors of ContextGNN for their amazing work and for publicly releasing their code!

If you use this code for your research, please do not forget to cite the original GitHub repository and paper of ContextGNN (accepted at [ICLR 2025](https://openreview.net/forum?id=nzOD1we8Z4&noteId=fSCflOd5my) and currently on arXiv):

```
@article{DBLP:journals/corr/abs-2411-19513,
  author       = {Yiwen Yuan and
                  Zecheng Zhang and
                  Xinwei He and
                  Akihiro Nitta and
                  Weihua Hu and
                  Dong Wang and
                  Manan Shah and
                  Shenyang Huang and
                  Blaz Stojanovic and
                  Alan Krumholz and
                  Jan Eric Lenssen and
                  Jure Leskovec and
                  Matthias Fey},
  title        = {ContextGNN: Beyond Two-Tower Recommendation Systems},
  journal      = {CoRR},
  volume       = {abs/2411.19513},
  year         = {2024}
}
```

And, of course, please do not forget to cite this repository and our paper :-)

```
@article{DBLP:journals/corr/abs-2503-16661,
  author       = {Alejandro Ariza-Casabona and
                  Nikos Kanakaris and
                  Daniele Malitesta},
  title        = {ContextGNN goes to Elliot: Towards Benchmarking Relational Deep Learning for Static Link Prediction (aka Personalized Item Recommendation)},
  journal      = {CoRR},
  volume       = {abs/2503.16661},
  year         = {2025}
}
```

### Requirements

Please, make sure to install the following packages to run our code:

- Cython
- hyperopt (you may need to install the version 0.2.5)
- torch
- Pillow
- PyYAML
- scikit_learn
- sparsesvd (you may need to install the version 0.2.0)
- torch_geometric
- pyg_lib 
- torch_scatter 
- torch_sparse
- torch-frame
- relbench==1.1.0
- sentence-transformers

Be careful to install the proper version of torch to be compatible with your acceleration hardware (if any) and with torch_geometric.

### STEP 1: Prepare your recommendation data

To work properly, the framework needs (in the minimum setting) the following files: 

- The list of users and items, two separate files formatted as tsv files with the original user (item) IDs and their numerical mapped IDs. The two files are named user_list.txt and item_list.txt, and their column headers are named org_id and remap_id.
- The training and test files (optionally, the validation file). They are formatted as tsv files where each row contains the user ID and the IDs of all the items the user has interacted with. Thus, the total number of rows will reflect the number of users in the dataset. All IDs in these files are intended to be the mapped IDs (i.e., reported in the remap_id columns from the user_list.txt and item_list.txt files).

### STEP 2: Prepare the dataset for Elliot and RelBench

In this step, the input dataset is processed to be used by RelBench within Elliot by running the Python script map_rel_bench.py.

```
python map_rel_bench.py --dataset <dataset_name>
```

This script reads the files from the previous step and creates the following files:

- Two tsv files train_elliot.tsv and test_elliot.tsv, formatted with two columns, one for the user and the other for the items, where each row corresponds to an interaction. These files are useful for Elliot to execute properly.
- Two tsv files, src_df.tsv and dst_df.tsv, that represent the user and item tables within the database, with a column for the user (item) IDs and the other containing a dummy timestamp value (as required in RelBench). Then, other two files are train_df.tsv and test_df.tsv, formatted with three columns, one indicating the user, the other the list of interacted items, and the final one (again) as a dummy timestamp value. Finally, a tsv file named target_table.tsv, whose content is quite similar to that of train_elliot.tsv, with an additional column for the dummy timestamp value. These five files are required by RelBench to execute properly.

### STEP 3: Prepare the configuration file for ContextGNN

From this step on, the process to perform the benchmark experiments is exactly the same as in Elliot. While we invite the readers to refer to the original ContextGNN's paper and code for a proper explanation and setting of all the hyper-parameters, as well as the official Elliot's documentation for further details regarding how to prepare a YAML configuration file, here we report (as an example) the configuration file we used to reproduce the results of ContextGNN on Gowalla.

**WARNING** Please, remember that the validation_rate parameter should always be less than the epochs parameter for the code to run without any problems.

```yaml
experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{0}/train_elliot.tsv
    test_path: ../data/{0}/test_elliot.tsv
  dataset: gowalla
  top_k: 20
  evaluation:
    cutoffs: [20]
    simple_metrics: [Recall, nDCG]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.ContextGNN:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: True
        validation_rate: 20
        validation_metric: Recall@20
        restore: False
        write_best_iterations: True
      lr: 0.001
      epochs: 20
      factors: 128
      batch_size: 128
      n_layers: 4
      aggr: sum
      channels: 128
      max_steps: 2000
      neigh: (16,16,16,16)
      seed: 42
```

### STEP 4: Run Elliot to benchmark ContextGNN

If all the previous steps run smoothly, we are all set to execute the experiment by running:

```
python start_experiments.py --dataset <dataset_name> --model <model_name>
```

### Reproducibility results

**Gowalla**

| Models     | Recall@20 | nDCG@20 |
|------------|-----------|---------|
| NGCF       | 0.1556    | 0.1320  |
| DGCF       | 0.1736    | 0.1477  |
| LightGCN   | 0.1826    | 0.1545  |
| SGL        | ---       | ---     |
| UltraGCN   | 0.1863    | 0.1580  |
| GFCF       | 0.1849    | 0.1518  |
| ContextGNN | 0.1712    | 0.1285  |

**Yelp 2018**

| Models     | Recall@20 | nDCG@20 |
|------------|-----------|---------|
| NGCF       | 0.0556    | 0.0452  |
| DGCF       | 0.0621    | 0.0505  |
| LightGCN   | 0.0629    | 0.0516  |
| SGL        | 0.0669    | 0.0552  |
| UltraGCN   | 0.0672    | 0.0553  |
| GFCF       | 0.0697    | 0.0571  |
| ContextGNN | 0.0543    | 0.0430  |

**Amazon Book**

| Models     | Recall@20 | nDCG@20 |
|------------|-----------|---------|
| NGCF       | 0.0319    | 0.0246  |
| DGCF       | 0.0384    | 0.0295  |
| LightGCN   | 0.0419    | 0.0323  |
| SGL        | 0.0474    | 0.0372  |
| UltraGCN   | 0.0688    | 0.0561  |
| GFCF       | 0.0710    | 0.0584  |
| ContextGNN | 0.0455    | 0.0379  |

### Contributors

In alphabetical order:

- Alejandro Ariza-Casabona (alejandro.ariza14@ub.edu)
- Nikos Kanakaris (kanakari@usc.edu)
- Daniele Malitesta (daniele.malitesta@centralesupelec.fr)