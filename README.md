# numGPT
Have you ever wanted to train a Transformer that's significantly slower and probably incorrect than the many existing library? Well then you've come to the right place. This entire architecture was written in purely numpy and no other dependency (except pyyaml) is required. 

## Creating a virtual env
Simply use `pipenv install` and `pipenv shell` to create the virtual environment. But like I said, since this only uses numpy, if you already have it installed you can go ahead and use it right away. 

## Notes
This repository contains all the tools you need to construct a basic transformer using the existing layers provided. This is for LEARNING PURPOSES ONLY! Please do not try to build a production ready transformer with this code.

## Support 
Please, if you see any errors with my gradient calculations or anything that doesn't make sense, PLEASE MAKE A PULL REQUEST! I am so certain I made some mistakes in my calculations and I would love your help.

## How to train a transformer
Use the `train.py` script and checkout the configs folder for a sample training configuration.

## Downloading the vocab and merges file
Just use `get_vocab.py` to download the merges and vocab file to the specified folder.

## Contributing
Please create a pull request if you'd like to contribute to this project. I'm a busy student but I'll be sure to review it as soon as possible!

## Todo
Write clear unit tests for each module (right now each module just has testing code when ran independently.)

