Summary
=======

This software can be used to reproduce the results in "SimplE Embedding for Link Prediction in Knowledge Graphs" paper. It can be also used to learn `SimplE` models for other datasets. The software can be also used as a framework to implement new tensor factorization models (implementations for `TransE` and `ComplEx` are included as two examples).

## Dependencies

* `Python` version 3.6 or higher
* `Numpy` version 1.13.1 or higher
* `PyTorch` version 1.1.0 or higher

## Usage

To run SimplE you should define the following parameters:

`ne`: number of epochs

`lr`: learning rate

`reg`:l2 regularization parameter

`dataset`: The dataset you want to run SimplE on

`emb_dim`: embedding dimension

`neg_ratio`: number of negative examples per positive example

`batch_size`: batch size

`save_each`: validate every k epochs

* Run `python main.py -ne ne -lr lr -reg reg -dataset dataset -emb_dim emb_dim -neg_ratio neg_ratio -batch_size batch_size -save_each save_each`


Running a model `M` on a dataset `D` will save the embeddings in a folder with the following address:

    $ <Current Directory>/models/D/

As an example, running the `SimplE` model on `wn18` will save the embeddings in the following folder:

    $ <Current Directory>/models/wn18/
    

## Reproducing the Results in the Paper

In order to reproduce the results presented in the paper, you should run the following commands:

### WN18

RUN `python main.py -ne 1000 -lr 0.1 -reg 0.03 -dataset WN18 -emb_dim 200 -neg_ratio 1 -batch_size 1415 -save_each 50`

### FB15K

RUN `python main.py -ne 1000 -lr 0.05 -reg 0.1 -dataset FB15K -emb_dim 200 -neg_ratio 10 -batch_size 4832 -save_each 50`

## Learned Embeddings for SimplE



## Publication

Refer to the following publication for details of the models and experiments.

- [Seyed Mehran Kazemi](https://mehran-k.github.io/) and [David Poole](http://www.cs.ubc.ca/~poole)

  [SimplE Embedding for Link Prediction in Knowledge Graphs](https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs)
  
  [Representing and learning relations and properties under uncertainty](https://open.library.ubc.ca/collections/ubctheses/24/items/1.0375812)


## Cite SimplE

If you use this package for published work, please cite one (or both) of the following:

    @inproceedigs{kazemi2018simple,
      title={SimplE Embedding for Link Prediction in Knowledge Graphs},
      author={Kazemi, Seyed Mehran and Poole, David},
      booktitle={Advances in Neural Information Processing Systems},
      year={2018}
    }
    
    @phdthesis{Kazemi_2018, 
      series={Electronic Theses and Dissertations (ETDs) 2008+}, 
      title={Representing and learning relations and properties under uncertainty}, 
      url={https://open.library.ubc.ca/collections/ubctheses/24/items/1.0375812}, 
      DOI={http://dx.doi.org/10.14288/1.0375812}, 
      school={University of British Columbia}, 
      author={Kazemi, Seyed Mehran}, 
      year={2018}, 
      collection={Electronic Theses and Dissertations (ETDs) 2008+}
    }

Contact
=======

Bahare Fatemi

Computer Science Department

The University of British Columbia

201-2366 Main Mall, Vancouver, BC, Canada (V6T 1Z4)  

<bfatemi@cs.ubc.ca>


License
=======

Licensed under the GNU General Public License Version 3.0.
<https://www.gnu.org/licenses/gpl-3.0.en.html>


Copyright (C) 2019 Bahare Fatemi
