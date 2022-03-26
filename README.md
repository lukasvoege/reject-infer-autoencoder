# Bias-Removing Autoencoder for Reject Inference
Information Systems Seminar @ HU-Berlin

Credit scoring models are trained on data from previously granted loan applications, where the borrowers' repayment behavior has been observed. This creates sampling bias: the model is trained on accepted cases only. The training data is not representative of the general population of borrowers, where the model is used to screen new applications. Reject inference comprises a set of techniques that aim at correcting the sampling bias by using data of rejected applications. 

Some bias correction methods work by transforming the data used to train a scoring model such that the transformed data is less biased. One of the promising approaches to find a suitable data transformation is to use a deep autoencoder with a distribution mismatch penalty (Atan et al, 2018).