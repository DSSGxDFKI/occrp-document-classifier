# Cheatsheet
This cheatsheed contains some commands which we found useful during development. Some of the problems here are specific to our environment, the solutions might therefore not be generalisable and working in other environments. 

# Docker

To create the image of our package:

```
pipenv lock -r > requirements.txt
docker build -t occrp-document-classifier:0.1.0 .
```
Locking the requirements can be skipped, if the dependencies were not changed.

<br>
<br>

If error `no space left on device`

```
export TMPDIR=/data/dssg/occrp/data/docker
```
<br>
<br>

To access the container when it's running:
```
docker exec -it occrp-document-classifier:0.1.0 bash
```
<br>
<br>

To run a command (example here train-document-classifier) using the container and deleting the container after the run:
```
docker run -it --rm occrp-document-classifier:0.1.0 bash -c "python main.py train-document-classifier" 
```
<br>
<br>

Using docker with a volume connected to it:
```
docker run -v /data/dssg/occrp/data/:/data/dssg/occrp/data/ -it --rm occrp-document-classifier:0.1.0 bash -c "python main.py convert-all-to-jpg /data/dssg/occrp/data/temp /data/dssg/occrp/data/temp"
```


# MLflow

Start the MLflow UI
```
mflow ui
```
The standard port for this is 5000. If it is run in Visual Studio Code, the IDE will by default forward the port. The UI will therefore be inspectable on 
http://127.0.0.1:5000.


Start the MLflow UI on a different port, e.g. 5001
```
mlflow ui --port 5001
```



Start the MLflow UI from the mlruns directory on our VM.
```
mlflow ui --backend-store-uri "/data/dssg/occrp/data/mlruns/"
```


Start MLflow in a tmux terminal to leave running in the background:
```
tmux new -s "MLflowUI"
mlflow ui --backend-store-uri "/data/dssg/occrp/data/mlruns/"
To exit without killing the tmux terminal press <Ctrl>-b, d
```
# tmux
list all current tmux sessions
```
tmux ls
```


enter session named "MLflowUI"
```
tmux attach-session -t MLflowUI
```
# Shell commands

Count how often a certain file exension is found in a directory and its subdirectories (also lists directories, confusingly)

```
ls -R | awk -F . '{print $NF}' | sort | uniq -c | sort -n -r | more
```

Save the results to a txt file:
```
ls -R | awk -F . '{print $NF}' | sort | uniq -c | sort -n -r | more > file_extensions.txt
```


The same in Windows PowerShell:
```
Get-Childitem -Recurse | WHERE { -NOT $_.PSIsContainer } | Group Extension -NoElement | Sort Count -Desc > FileExtensions.txt
```
<br>
<br>

Delete recursively all 1.jpg from current directory and subfolders

```
find . -name \*1.jpg -exec rm {} \;
```
<br>
<br>

Forward a port (useful if you don't use VS Code and want to inspect the MLflow UI running on the server), run this from your local machine:
```
ssh -L 5000:127.0.0.1:5000 username@10.30.40.120
```
<br>
<br>

# Shell
If error
```
/bin/sh: error while loading shared libraries: libc.so.6: cannot change memory protections
```
run
```
restorecon -R -v $HOME/.local/share/containers
```
<br>
<br>

If error
```
no space left on device
```
run
```
export TMPDIR=/data/dssg/occrp/data/docker
```
<br>
<br>

Convert all pdf and tifs from our input folder to the output folder (execute from root folder of the project):

```
pipenv run python src/preprocessing/preprocessing_cli.py convert-all-to-jpg /data/dssg/occrp/data/input/document_classification_data/ /data/dssg/occrp/data/processed/
```
<br>
<br>

# pytest

Running pytest:
```
pytest
```

Without warnings
```
pytest --disable-pytest-warnings
```

To check the coverage of pytest:

```
pytest --cov=src tests
```

# VS Code

Increase width for columns when debugging data frames (not tested):
```
pd.options.display.max_colwidth = 400
```

# Portainer

Start the portainer:
```
./portainer.sh
```

Example credentials:
```
User: admin
PW: admin123admin123
```

# GPU
Run a command (example here train-document-classifier) via GPU:
```
./run_gpu_model.sh train-document-classifier
```