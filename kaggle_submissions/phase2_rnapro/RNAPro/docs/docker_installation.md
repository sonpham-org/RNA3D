### Run with Docker

1. Install Docker (with GPU Support)

    Ensure that Docker is installed and configured with GPU support. Follow these steps:
    *  Install [Docker](https://www.docker.com/) if not already installed.
    *  Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU support.
    *  Verify the setup with:
        ```bash
        docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
        ```
        
2. Pull the Docker image, which was built based on this [Dockerfile](../Dockerfile)
    ```bash
    docker pull nvcr.io/nvidia/pytorch:25.09-py3
    ```

3. Clone this repository and `cd` into it
    ```bash
    git clone https://github.com/NVIDIA-Digital-Bio/RNAPro
    cd ./RNAPro
    ```

4. Run Docker with an interactive shell
    ```bash
    docker run --gpus all -it -v $(pwd):/workspace -v nvcr.io/nvidia/pytorch:25.09-py3 /bin/bash
    ```
5. Install RNAPro
    ```bash
    pip install -e .
    ```
  After running the above commands, you can train and run inference with RNAPro inside the container's environment.