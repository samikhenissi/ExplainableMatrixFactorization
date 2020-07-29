# Explainable Matrix Factorization using Pytorch-lightning

This is an implementation of the paper [Using Explainability for Constrained Matrix Factorization](https://dl.acm.org/doi/10.1145/3109859.3109913). 
We implement both EMF amd MF with explainability metrics. 

Many of the components of this implementation  are modified from the following  [repo](https://github.com/yihong-chen/neural-collaborative-filtering)


## Requirements and installation
We used pytorch 1.5.1 and pytorch-lightning==0.8.5

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements or refer to official website for [pytorch](https://pytorch.org/) and [pytorch-lightning](https://github.com/PytorchLightning/pytorch-lightning).

```bash
pandas==1.0.1
numpy==1.18.1
torch==1.5.1
pytorch-lightning==0.8.5
scipy==1.4.1
scikit-learn==0.22.1
```

You can also use  

```bash
pip install -r requirements.txt
```

## Usage

The main training script is in train.py. You will need a training data in a pandas dataframe that has the following columns:  ['uid', 'mid', 'rating', 'timestamp']

You can try the implementation on Movielens-100K or Movielens-1m

For example, to run the training script using EMF  on the Movielens-100 data you can use:

```bash
train.py --model EMF --data movielens100
```

pytorch-lightning allows scalable training on multi gpus. For more information refer to: [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html) 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

To contact me with any concern you can also email me at sami.khenissi@louisville.edu
## License
[MIT](https://choosealicense.com/licenses/mit/)
