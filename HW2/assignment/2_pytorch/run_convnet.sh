#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 3 \
    --hidden-dim 24 \
    --epochs 6 \
    --weight-decay 0.95 \
    --momentum 0.9 \
    --batch-size 256 \
    --lr 0.0001 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
