# airbus ship detection

## Contents

* requirements.txt - necessary python packages
* train.py - data preparation and model fitting
* test.py - model implementation (recreates the model and loads the weights)
* airbus_data_prep.ipynb - notebook with dataset analysis and preparation

## Info

This is an attempt at building a semantic segmentation model for the Airbus Ship Detection Challenge, mostly done as a test, rather than a fully optimized solution. A lot of corners had to be cut due to execution time being way too long.

## Model Info

Basic U-Net using keras. Takes 128x128 images as an input (working with the original 768x768 images would take way too much time). I was considering using a pre-trained model, but decided to build one from the ground up and train it from scratch just to see how it goes. Training was done using only the images containing 1 or more ships, in batches of 100 over 4 epochs. 

## How to set up

1. Install the packages from requirements.txt
2. Place train.py into the folder with the dataset
3. Run train.py. This will train the model (takes a while). After train.py finishes, some files will be produced.
   - 'logs' folder. Contains training logs that can be used using TensorBoard.
   - weights files. Those are used to load the trained weights for testing.