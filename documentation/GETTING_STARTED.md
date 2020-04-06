<p align="left"><a href="../README.md">&#8592; README</a></p>

# Getting started

## Table of contents
- [Requirements](#requirements)
- [Configuring ee-outliers](#configuring-ee-outliers)
- [Running in interactive mode](#running-in-interactive-mode)
- [Running in daemon mode](#running-in-daemon-mode)
- [Customizing your Docker run parameters](#customizing-your-docker-run-parameters) 

## Requirements
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) to build and run ee-outliers
- Access to an Elasticsearch cluster

## Running ee-outliers with Docker Compose 

Using ee-outliers is basically a four-step process:
1. Define the ee-outliers configuration file. 
2. Define your outlier detection use cases.
3. Define the docker container and the ee-outliers parameters inside a Compose file `docker-compose.yml`.
4. Build an image and run ee-outliers with `docker-compose up`.

### Step 1: Configuring ee-outliers

ee-outliers makes use of a single configuration file containing all required parameters such as connectivity 
with your Elasticsearch cluster, logging, etc.

A default configuration file with all required configuration sections and parameters, along with an explanation, can be
 found in [`defaults/outliers.conf`](../defaults/outliers.conf). We recommend starting from this file when running 
 ee-outliers yourself.
 
A full description of all configuration parameters can be found [here](CONFIG_PARAMETERS.md).  

### Step 2: Define the outlier detection use cases

Each detection use case has to be defined in a separate .conf file.

We provided 4 examples of use cases available in the [use_cases/examples](../use_cases/examples) repository.
A detailed description of these use case examples, along with information on how you can create your owns can be found 
[here](CONFIG_OUTLIERS.md).

### Step 3: Define docker container & ee-outliers parameters in the Compose file

The Compose file is located at [`docker-compose.yml`](../docker-compose.yml) and should look like this:

```
version: '3'
services:
  outliers:
    build: .
    container_name: your_outliers_container_name
    command: "python3 outliers.py RUN_MODE --config /mappedvolumes/config/outliers.conf --use-cases /use_cases/*.conf"
    volumes:
      - ./defaults/outliers.conf:/mappedvolumes/config/outliers.conf
      - ./use_cases/examples:/use_cases
    network_mode: network_name
```
It allows you to define the docker container and the ee-outliers parameters for then build and run the ee-outliers image
in one single command line. For more information about the Compose file, see the 
[Compose file reference](https://docs.docker.com/compose/compose-file/).

The main parameters of the Compose file are as follow:

- [`container_name`](https://docs.docker.com/compose/compose-file/#compose-file-structure-and-examples#container_name):
Your custom container name.

- [`command`](https://docs.docker.com/compose/compose-file/#command):
The command line that will execute `outliers.py`.

    The `--config` and `--use-cases` argument require respectively the location of the configuration and the use cases file.
    Note that the `--use-cases` argument can also contain wildcards, such as ``"/my/usecase/folder/*.conf"``.

    The `RUN_MODE` argument should be replaced by one of the 3 running modes:

    - `interactive`: In interactive mode, ee-outliers will run once and finish. 
    This is the ideal run mode to use when testing ee-outliers straight from the command line.
    If you are testing ee-outliers for the first time, we are recommending to use it.

    - `daemon`: In daemon mode, ee-outliers will continuously run based on a cron schedule defined in the outliers 
    configuration file.
    The following example from the default configuration file will run ee-outliers at 00:10 each night (standard cron format).

        ```ini
        [daemon]
        schedule=10 0 * * *
        ```
    
    - `tests`: In test mode, ee-outliers will run all unit tests and finish, providing feedback on the test results. 
    This mode, which is 
    [developer-oriented](https://github.com/NVISO-BE/ee-outliers/blob/master/documentation/DEVELOPMENT.md), is useful for 
    developing and debugging purposes.

- [`volumes`](https://docs.docker.com/compose/compose-file/#volumes):
The mapped volumes so that your configuration  and use case files can be found. In this example, the default 
configuration file in ``/defaults`` is mapped to ``/mappedvolumes/config`` and the ``/use_cases/examples`` is mapped to 
``/use_cases``.

- [`network_mode`](https://docs.docker.com/compose/compose-file/#network_mode):
The name of the docker network through which the Elasticsearch cluster is reachable.

### Step 4: Build & run ee-outliers with Docker Compose

Thanks to Docker Compose, we can build and run an image of ee-outliers with one single command line:

```
docker-compose up
```

To stop and remove the container use:

```
docker-compose down
```

## Build & run ee-outliers with Docker

For convenience, we recommend using Docker Compose but the user can also use Docker and specify the ee-outliers 
parameters straight from the command line. 

To achieve that, after following [Step 1](#step-1:-configuring-ee-outliers) and 
[Step 2](#step-2:-define-the-outlier-detection-use-cases), you can enter the following commands:

```BASH
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run \
--network=network_name \
-v "$PWD/defaults:/mappedvolumes/config" \
-i  outliers-dev:latest python3 outliers.py interactive \
--config /mappedvolumes/config/outliers.conf \
--use-cases "/my/usecase/folder/*.conf"
```

## Additional content

- [TLS beaconing detection using ee-outliers and Elasticsearch](https://blog.nviso.eu/2018/12/11/tls-beaconing-detection-using-ee-outliers-and-elasticsearch/)
- [Detecting suspicious child processes using ee-outliers and Elasticsearch](https://blog.nviso.eu/2018/12/21/detecting-suspicious-child-processes-using-ee-outliers-and-elasticsearch/)

---

<p align="right"><a href="CONFIG_OUTLIERS.md">Building detection use cases &#8594;</a></p>
