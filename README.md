# CS5014-P2
Classification of Seal

## Instructions
To run the program, type `python3 p2.py [binary|multi] [nn|svm|linear_svm|knn] [debug|]`

The first argument is to choose either the binary or the multi-class classification task

The second argument is to choose the model to train with. `nn` is for neural networks, `svm` is for support vector machine, `linear_svm` is for linear support vector machine and `knn` is for K-nearest neighbors

The final argument `debug` is optional. Including this flag will calculate the best parameters to use based on the classification task and model using GridSearchCV before predicting the model.
