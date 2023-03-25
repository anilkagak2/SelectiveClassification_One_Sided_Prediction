# [SelectiveClassification_One_Sided_Prediction](https://proceedings.mlr.press/v130/gangrade21a.html)

Selective Classification via One Sided Prediction

## Brief Description  

## Installation 

Our codebase is written using [PyTorch](https://pytorch.org). You can set up the environment using [Conda](https://www.anaconda.com/products/individual) and executing the following commands.  

```
conda create --name pytorch-1.10 python=3.9
conda activate pytorch-1.10
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Please update the last command as per your system specifications (see [PyTorch-Install](https://pytorch.org/get-started/locally/)). Although we have not explicitly tested all the recent PyTorch versions, but you should be able to run this code on PyTorch>=1.7 and Python>=3.7

Please install the following packages used in our codebase.

```
pip install thop
```

## Training Scripts 

```
bash runner.sh
```

## Reference (Bibtex entry)

```

@InProceedings{gangrade21a_SC_OSP,
  title = 	 { Selective Classification via One-Sided Prediction },
  author =       {Gangrade, Aditya and Kag, Anil and Saligrama, Venkatesh},
  booktitle = 	 {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {2179--2187},
  year = 	 {2021},
  editor = 	 {Banerjee, Arindam and Fukumizu, Kenji},
  volume = 	 {130},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--15 Apr},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v130/gangrade21a/gangrade21a.pdf},
  url = 	 {https://proceedings.mlr.press/v130/gangrade21a.html},
}
```
