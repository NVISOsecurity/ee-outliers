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

## Running ee-outliers

Using ee-outliers is basically a four-step process:
1. Define the ee-outliers configuration file. 
2. Define your outlier detection use cases.
3. Define the docker container and the ee-outliers parameters inside a Compose file `docker-compose.yml`.
4. Build an image and run ee-outliers with `docker-compose up`.

#### Step 1: Configuring ee-outliers

ee-outliers makes use of a single configuration file containing all required parameters such as connectivity 
with your Elasticsearch cluster, logging, etc.

A default configuration file with all required configuration sections and parameters, along with an explanation, can be
 found in [`defaults/outliers.conf`](../defaults/outliers.conf). We recommend starting from this file when running 
 ee-outliers yourself.
 
A full description of all configuration parameters can be found [here](CONFIG_PARAMETERS.md).  

#### Step 2: Define the outlier detection use cases

Each detection use cases has to be defined in a separate .conf file.

TODO small description of what a ee-outliers file is responsible for.

Examples of how we define a use cases file can be found in [use_cases/examples](../use_cases/examples).
For detailed description of these examples, along with information about each existing detection model and their 
parameters can be found [here](CONFIG_OUTLIERS.md).


#### Step 3: Define docker container & ee-ouliers parameters in a Compose file

The Compose file is located at [`docker-compose.yml`](../docker-compose.yml) and should look like this:

```
version: '3'
services:
  outliers:
    build: .
    container_name: outliers_mroberti_dev
    command: "python3 outliers.py interactive --config /mappedvolumes/config/outliers.conf --use-cases /use_cases/*.conf"
    volumes:
      - ./defaults/outliers.conf:/mappedvolumes/config/outliers.conf
      - ./use_cases/examples:/use_cases
    network_mode: 
    restart: always
```


#### Step 4: Build & run ee-outliers

Thanks to docker-compose, we can build and run an image of ee-outliers with the following single command line:

```
docker-compose up
```
To stop your... TODO


## Create use cases

Before running ee-outliers

## Running with docker-compose

Docker-compose will allow you to build and run an image of ee-outliers with one single command:

```
docker-compose up
```

The docker and ee-outliers are defined in docker-compose.yml

For convenience, we recommend to use docker-compose for running ee-outliers. 
If you want to use docker and specify the parameters straight from the command line, please refer to the section [Running with 
docker](#running-with-docker). 



## Running with docker

## Configuring ee-outliers

ee-outliers makes use of a single configuration file ``-config`` containing all 
required parameters such as connectivity with your Elasticsearch cluster, logging, etc.  

Detection use cases are provided using parameter ``--use-cases``.

An example configuration file with all required configuration sections and parameters, 
along with an explanation, can be found in 
[`defaults/outliers.conf`](../defaults/outliers.conf). 
We recommend starting from this file when running ee-outliers yourself.  

A full description of all configuration parameters can be found [here](CONFIG_PARAMETERS.md).        

Visit [Building detection use cases](CONFIG_OUTLIERS.md) for information how 
to define your own outlier detection use cases.

## Running in interactive mode
In this mode, ee-outliers will run once and finish. This is the ideal run mode 
to use when testing ee-outliers straight from the command line.

Running ee-outliers in interactive mode:

```BASH
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run \
-v "$PWD/defaults:/mappedvolumes/config" \
-i  outliers-dev:latest python3 outliers.py interactive \
--config /mappedvolumes/config/outliers.conf \
--use-cases "/my/usecase/folder/*.conf"
```

## Running in daemon mode
In this mode, ee-outliers will continuously run based on a cron schedule 
defined in the outliers configuration file.

Example from the default configuration file which will run ee-outliers 
at 00:10 each night (standard cron format).

```ini
[daemon]
schedule=10 0 * * *
```

Running ee-outliers in daemon mode:

```BASH
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run \
-v "$PWD/defaults:/mappedvolumes/config" \
-d outliers-dev:latest python3 outliers.py daemon \
--config /mappedvolumes/config/outliers.conf \
--use-cases "/my/usecase/folder/*.conf"
```

## Customizing your Docker run parameters

The following modifications might need to be made to the above commands for your specific situation:
- The name of the docker network through which the Elasticsearch cluster is reachable (``--network``)
- The mapped volumes so that your configuration file can be found (``-v``). By default, the default configuration file in ``/defaults`` is mapped to ``/mappedvolumes/config``
- The path of the configuration file (``--config``)
- One or more paths to use case config files (``--use-cases``). 
Path can also contain wildcards, such as ``"/my/usecase/folder/*.conf"``.


<p align="right"><a href="CONFIG_OUTLIERS.md">Building detection use cases &#8594;</a></p>
