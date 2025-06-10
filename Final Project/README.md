# Project 6 - Music Instrument Classification

## Team

- Giuliano Di Lorenzo
- Carmen Frieda Franci
- Nicola Antonio Magno
- Jorge Cuartero Galindo

## Work organization

In this project you find the following notebooks:

- `SaveStem`: feature extraction after segmentation of stem files
- `SaveMultiLabel`: feature extraction after segmentation of mix files
- `ModelForStem`: 2D-CNN modelling for MIC based on stem files
- `ModelForMix`: 2D-CNN modelling for MIC based on mix files
- `ModelHybrid`: 2D-CNN modelling for MIC for both stem and mix files
- `GenreAnalysis`: evaluation between instruments presence and genre

In order to run everything correctly, it is mandatory to start from `SaveStem` and `SaveMultiLabel`, as they prepare the necessary material for the other notebooks.
Taking into account the necessary segmentation occurring in those 2 notebooks, it requires all the audio material and takes some time (around 1 hour for both notebooks).

We suggest to run the notebooks in the same order they are presented in the previous list, which is the same order we used.

## Credits

This project is part of the "Selected Topics in Music and Acoustic Engineering" exam at Politecnico di Milano.