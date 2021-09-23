### folder "face_classifier"
contains all specific code for face classification, separated from main project files on purpose.
### preprocess.py
All functions related to processing a raw dataset go here (such as lookit - mostly code from icatcher+, or any other dataset)
### options.py
Use to parse command line arguments. 
### logger.py
Logging functions should go here (including prints)
### models.py
All torch models go here
### train.py
Main training loop, keep as generic as possible and minimal
### visualize.py
All visualizations should go here
### data.py
All torch data loaders should go here
### config.py
Contains general configurations. Should be integrated into options.py
