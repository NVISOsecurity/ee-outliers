<p align="left"><a href="../README.md">&#8592; README</a></p>

# Getting started

**Table of contents:**
- [Requirements](#requirements)
- [Configuring ee-outliers](#configuring-ee-outliers)
- [Running in interactive mode](#running-in-interactive-mode)
- [Running in daemon mode](#running-in-daemon-mode)
- [Customizing your Docker run parameters](#customizing-your-docker-run-parameters)

## Requirements
- Docker to build and run ee-outliers
- Internet connectivity to build ee-outliers (internet is not required to run ee-outliers)

## Configuring ee-outliers

ee-outliers makes use of a single configuration file, in which both technical parameters (such as connectivity with your Elasticsearch cluster, logging, etc.) are defined as well as the detection use cases.

An example configuration file with all required configuration sections and parameters, along with an explanation, can be found in [`defaults/outliers.conf`](../defaults/outliers.conf). We recommend starting from this file when running ee-outliers yourself.  Note that all configuration parameters are listed [here](CONFIG_PARAMETERS.md).        
Go to the page [Building detection use cases](CONFIG_OUTLIERS.md) if you would like to know how to define your own outlier detection use cases.

## Running in interactive mode
In this mode, ee-outliers will run once and finish. This is the ideal run mode to use when testing ee-outliers straight from the command line.

Running ee-outliers in interactive mode:

```BASH
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run --network=sensor_network -v "$PWD/defaults:/mappedvolumes/config" -i  outliers-dev:latest python3 outliers.py interactive --config /mappedvolumes/config/outliers.conf
```

## Running in daemon mode
In this mode, ee-outliers will continuously run based on a cron schedule defined in the outliers configuration file.

Example from the default configuration file which will run ee-outliers at 00:10 each night (format: Minutes Hours DayNumber Month DayName):

```init
[daemon]
schedule=10 0 * * *
```

Running ee-outliers in daemon mode:

```BASH
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run --network=sensor_network -v "$PWD/defaults:/mappedvolumes/config" -d outliers-dev:latest python3 outliers.py daemon --config /mappedvolumes/config/outliers.conf
```

## Customizing your Docker run parameters

The following modifications might need to be made to the above commands for your specific situation:
- The name of the docker network through which the Elasticsearch cluster is reachable (``--network``)
- The mapped volumes so that your configuration file can be found (``-v``). By default, the default configuration file in ``/defaults`` is mapped to ``/mappedvolumes/config``
- The path of the configuration file (``--config``)


<p align="right"><a href="CONFIG_OUTLIERS.md">Building detection use cases &#8594;</a></p>
