# Developing and debugging ee-outliers

**Table of contents:**
- [Running in test mode](#)
- [Running in profiler mode](#)
- [Checking code style, PEP8 compliance & dead code](#)

ee-outliers supports additional run modes (in addition to interactive and daemon) that can be useful for developing and debugging purposes.

## Running in test mode
In this mode, ee-outliers will run all unit tests and finish, providing feedback on the test results.

Running outliers in tests mode:

```
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run -v "$PWD/defaults:/mappedvolumes/config" -i outliers-dev:latest python3 outliers.py tests --config /mappedvolumes/config/outliers.conf
```

## Running in profiler mode
In this mode, ee-outliers will run a performance profile (``py-spy``) in order to give feedback to which methods are using more or less CPU and memory. This is especially useful for developers of ee-outliers.

Running outliers in profiler mode:

```
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run --cap-add SYS_PTRACE -t --network=sensor_network -v "$PWD/defaults:/mappedvolumes/config" -i  outliers-dev:latest py-spy -- python3 outliers.py interactive --config /mappedvolumes/config/outliers.conf
```

## Checking code style, PEP8 compliance & dead code
In this mode, flake8 is used to check potential issues with style and PEP8.

Running outliers in this mode:

```
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run -v "$PWD/defaults:/mappedvolumes/config" -i outliers-dev:latest flake8 /app
```

You can also provide additional arguments to flake8, for example to ignore certain checks (such as the one around long lines):

```
docker run -v "$PWD/defaults:/mappedvolumes/config" -i outliers-dev:latest flake8 /app "--ignore=E501"
```

To check the code for signs of dead code, we can use vulture:

```
docker run -v "$PWD/defaults:/mappedvolumes/config" -i  outliers-dev:latest python3 -m vulture /app
```

