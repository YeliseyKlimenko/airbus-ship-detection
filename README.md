# airbus ship detection

## Contents

* requirements.txt - necessary python packages
* train.py - data preparation and model fitting
* test.py - model implementation (recreates the model and loads the weights)
* logs folder - tensorboard logs from model fitting
* weights folder - files to load weights from
* airbus_data_prep.ipynb - notebook with dataset analysis and preparation

## Info

This is an attempt at building a semantic segmentation model for the Airbus Ship Detection Challenge, mostly done as a test, rather than a fully optimized solution. A lot of corners had to be cut due to execution time being way too long (took 7+ hrs just to process the images and train the model).
Masks for testing weren't provided, so no accuracy assesment has been done.

## Model Info

Basic U-Net using keras. Takes 128x128 images as an input (working with the original 768x768 images would take way too much time). I was considering using a pre-trained model, but decided to build one from the ground up and train it from scratch just to see how it goes. Training was done using only the images containing 1 or more ships, in batches of 100 over 2 epochs. Yes, not a lot of epochs - each one was taking over 3 hrs to process, but at least it's better than 1.

Note: the tensorboard logs only show 1 epoch, so they aren't very descriptive