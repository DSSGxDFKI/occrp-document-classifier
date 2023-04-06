# Docker GPU

To run CLI commands in a Docker container with GPU support out of the box, this project provides the script [run_gpu_model.sh](../run_gpu_model.sh). This script builds an image with the project code and runs the image with a selected CLI command.

## Step 1: set the GPU id in the config

The [config file](../src/config.py) defines a dictionary to select the id of the GPU to be used:
```python
PROCESSOR_SETTINGS: dict = {
        # "processor_type": "GPU",   # GPU or CPU
        "n_CPU": 16,  # From 0 to 16.
        "GPU_id": None,  # GPU_id = 0 or 1 or None
        "GPU_mb_memory": 38_000,  # Amount of GPU memory in MB
    }
```

To use CPU, `"GPU_id"` should be set to `None`. To use GPU, `"GPU_id"` should match the desired ID of the GPU. The GPU availables in a server and their IDs can be displayed with the command `nvidia-smi`.

## Step 2: run the gpu script
To run the current project in a detached container, use this command:

 ```console
./run_gpu_model.sh [OPTIONS] COMMAND [ARGS]...
```

The same commands defined in the [CLI README](./CLI.md) are available through GPU. In general, changing `python src/main.py` to `./run_gpu_model.sh` for any command will cause the command to run in a GPU container.

## Useful commands

To manage the running Docker containers, a useful software is [Portainer](https://www.portainer.io/). It is a UI that allows the user to run, see the logs and stop containers from a browser. To run it:
```
./portainer.sh
```


You can check the usage of GPU's and CPU's by your processes by running the following commands in any terminal inside the server:

```bash
# to check GPU usage
watch -n 1 nvidia-smi

# to check CPU usage
htop
```
