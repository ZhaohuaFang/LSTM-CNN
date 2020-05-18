When doing natural language processing, combining LSTM with CNN is an idea because machine can extract both local features and global features. Thus, in theory, neural networks can have a better sentiment classification. However, sometimes, for example in dataset of imdb, single LSTM can have better ability in judgement.

Materials below are two pieces of code showing my implementation on the above two ideas via TensorFlow 2.0.

The highest accuracy of LSTM_Dense.py is about 0.83361378.

The highest accuracy of LSTM_CNN_Regularization_BatchNormalization.py is about 0.825480769, which is lower than the model in LSTM_Dense.py.
