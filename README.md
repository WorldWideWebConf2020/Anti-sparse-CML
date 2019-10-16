# Anti sparse CML
Reproduce results of the paper.

### Build the docker
In the repository directory, run:

```
cd docker
```

Then run:

```
docker build -t drec .
```

### Docker command
Create a container with the following command:

```
nvidia-docker run     -ti     --rm     --memory=20g  --shm-size=15g  --name=DREC -v/path/to/DREC:/workspace/DREC drec 
```

### Download the dataset
In the docker, run:

```
python data/core.py
```

It should add a directory in ```data``` with the downloaded dataset.

### Run experiments
In the docker, run:

```
python scripts/model_name/run_exp.py
```

It produces a file results in which is saved the model and its hyperparameters as well as a ```metrics.txt``` file in which is written the score on the evaluation set.

### Evaluate model on test set
After running an experiment on a model, to select the hyperparameters with the highest score on evaluation set and test it on the test set, run:

```
python scripts/model_name/evaluate.py
```

Produces a ```metrics.txt``` file in which is written the score on the test set.
