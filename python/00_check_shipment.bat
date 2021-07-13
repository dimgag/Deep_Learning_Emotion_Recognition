
REM train architecture 9 for 20 epochs
REM - 200 epochs are used in experiments
python train.py 0 20

REM Close the figure and continue...

REM evaluate model 5
python evaluate.py 5

REM feature maps for layers in model 6
python visualize_layers.py 6
