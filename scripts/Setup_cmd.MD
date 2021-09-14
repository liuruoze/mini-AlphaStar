# use Pytorch 1.5:
```
conda create -n torch_1_5 python=3.7 pytorch=1.5 -c pytorch
```

# or install PyTorch 1.7:
```
conda create -n torch_1_7 python=3.7 pytorch=1.7 -c pytorch
```

# After install PyTorch, please run (assuming use torch_1_5):
```
conda activate torch_1_5
```
to activate the environment.

# Then run:
```
pip install -r requirements.txt
``` 
to install other packages.

# Finally, run the below python file to run the program:
```
python run.py
```
to start the program (you may use comments and uncomments to select the training process you want).