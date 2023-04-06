# Docker usage

If you want to use the Docker version, make sure that you have [Docker](https://docs.docker.com/engine/install/) installed. Change directory to the root folder of this project and run: 

```bash
docker build -t occrp-document-classifier:0.1.0 .
```

This will take around 5 minutes and create a Docker image of around 6-7 GB size. List all Docker images on your machine via 
```
docker image list
```
and copy the IMAGE ID of `occrp-document-classifier`. 

To spin up the container, pass a command to it and remove it again, run: 
```
docker run -it --rm occrp-document-classifier:0.1.0 bash -c "<your-cli-command-here>"
```

For example, if you want to run "python main.py train-document-classifier", run:
```
docker run -it --rm occrp-document-classifier:0.1.0 bash -c "python main.py train-document-classifier"
```