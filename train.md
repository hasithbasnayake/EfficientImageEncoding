
Okay, new plan for the automated training loop. 

Two main functions, `train_model` and `evaluate`.

`train_model`

1. Takes in hyper-parameters, including number of epochs, eval every X epochs, and whether to train decoder
2. Creates a directory, stores config.json, creates state.json to track training, begins. 
3. Instantiates model, processes dataset, loads dataset, begins training (DataLoader gets torch.Generator for shuffling), begins training
4. Handles the printing.
5. At every epoch it saves the model, `models/epoch_XXX.pt`
6. At every X epochs it calls `evaluate(snapshot)` on the last trained model, skip if decoder is false 
	1. Creating a folder `evals/epoch_XXX` which it saves that current model too 
	2. trains the decoder, tests it, and saves the result
	3. Adds epoch to `evaluated` in state.json
7. Continues training till finished, sets status in state.json to completed
8. If Ctrl+C it saves the model to models/epoch_XXX.pt, overwrites checlpoint.pt (weights, epoch, RNG states), and updates state.json

`evaluate`

1. Trains the decoder on the specified model
2. Tests it
3. Saves the result to `evals/epoch_XXX`

So the commands become....

```
python train.py --start config.json [--name baseline]
python train.py --resume runs/X                 # continue to num_epochs
python train.py --extend runs/X --epochs 150    # extend to 150 total 
python train.py --eval runs/X [--epoch 40]      # decode + test a snapshot (default: latest)
```

- `python train.py --start config.json [--name baseline]`
	- Finds config.json, loads it, creates a new folder with config.json, and a state.json, `status: epoch` which is either the epoch or "COMPLETE", begins training run, decodes, and evaluates 
	- Decoder is a key in the config.json too btw, this way resume can reference it to know whether it should resume the run and decode. 
- `python train.py --resume /trainingfolderpath`
	- Finds config.json in the /trainingfolderpath, sees what the last epoch is of status, if status is not or complete it should break because that's already a completed run and suggest `--extend`
- `python train.py --extend /trainingfolderpath --epochs 150`
	- Finds config.json in the /trainingfolderpath, adds a flag to the state.json that the run was extended to to 150 epochs. Training data stays the same obviously. Works on unfinished runs too, simply has the same effect as --extend + resume
- `python train.py --eval /trainingfolderpath --epoch 40`
	- Finds the models `models/epoch_040.pt` for that epoch and runs the evaluation into` evals/epoch_040/`. If that eval already exists it skips unless --force. Defaults to lastest epoch.

`state.json`

```
{"last_epoch": 37, "target_epochs": 100, "status": "training", "evaluated": [10, 20, 30]}
```

- state.json and checkpoint.pt are written atomically (save to a temp file, then `os.replace`).

Folder structure

- /test_run_10_17_2026
    - config.json
    - state.json
    - checkpoint.pt
    - models/
        - epoch_001.pt
        - epoch_002.pt
        - ...
        - epoch_100.pt
    - evals/
        - epoch_010/
            - decoder.pt
            - results.json
            - *.png...
        - ....

Ctrl+C during training or an evaluation gets caught. The train function exits at the last completed epoch, with checkpoint.pt and state.json already saved. Mid-epoch progress is lost; revisit this with a between-images checkpoint if an epoch turns out to take too long.