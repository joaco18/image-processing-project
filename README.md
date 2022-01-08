# SD-OCT fluid segmentation

![alt text](https://github.com/joaco18/rashno-2017-fluid-oct/blob/main/summary.png)
### This repository offers an implementation of Rashno et al. 2017 paper:

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186949

### This implementation includes some modifications in order to make the proposed algorithm work on the images comming from:

https://retouch.grand-challenge.org/



To start, create a new conda enviroment:

    <conda create -n improject anaconda>

Then install the requirements:

    <pip install -r requirements.txt>

Then contact the administrators of the data and download the data. Once you get the zips the should unziped in a "data" folder inside this same repository.

There should be six folder corresponding to Train and Test sets for each manufacturer. Inside each folder several "cases" volder are axpected.

Try the code using the MAIA_ starting notebooks
