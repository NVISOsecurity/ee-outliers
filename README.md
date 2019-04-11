# ee-outliers

```
                               __  ___
  ___  ___        ____  __  __/ /_/ (_)__  __________
 / _ \/ _ \______/ __ \/ / / / __/ / / _ \/ ___/ ___/
/  __/  __/_____/ /_/ / /_/ / /_/ / /  __/ /  (__  )
\___/\___/      \____/\__,_/\__/_/_/\___/_/  /____/

Open-source framework to detect outliers in Elasticsearch events
Developed by NVISO Labs (https://blog.nviso.be - https://twitter.com/NVISO_Labs)
```

<p align="center">
<img alt="Detecting beaconing TLS connections using ee-outliers" src="https://forever.daanraman.com/screenshots/Beaconing%20detection.png?raw=true" width="650"/><br/>
<i>Detecting beaconing TLS connections using ee-outliers</i>
</p>

## Introduction

ee-outliers is a framework to detect outliers in events stored in an Elasticsearch cluster.
The framework was developed for the purpose of detecting anomalies in security events, however it could just as well be used for the detection of outliers in other types of data.


The framework makes use of statistical models that are easily defined by the user in a configuration file. In case the models detect an outlier, the relevant Elasticsearch events are enriched with additional outlier fields. These fields can then be dashboarded and visualized using the tools of your choice (Kibana or Grafana for example).

The possibilities of the type of anomalies you can spot using ee-outliers is virtually limitless. A few examples of types of outliers we have detected ourselves using ee-outliers during threat hunting activities include:

-	Detect beaconing (DNS, TLS, HTTP, etc.)
-	Detect geographical improbable activity
-	Detect obfuscated & suspicious command execution
-	Detect fileless malware execution
-	Detect malicious authentication events
-	Detect processes with suspicious outbound connectivity
-	Detect malicious persistence mechanisms (scheduled tasks, auto-runs, etc.)
-	…

Checkout the screenshots at the end of this readme for a few examples.
Continue reading if you would like to get started with outlier detection in Elasticsearch yourself!

## Core features
- Create your own custom outlier detection use cases specifically for your own needs
- Send automatic e-mail notifications in case one of your outlier use cases hit
- Automatic tagging of asset fields to quickly spot the most interesting assets to investigate
- Fine-grained control over which historical events are checked for outliers
- ...and much more!

## Requirements
- Docker to build and run ee-outliers
- Internet connectivity to build ee-outliers (internet is not required to run ee-outliers)

## Getting started

### Configuring ee-outliers

ee-outliers makes use of a single configuration file, in which both technical parameters (such as connectivity with your Elasticsearch cluster, logging, etc.) are defined as well as the detection use cases.

An example configuration file with all required configuration sections and parameters, along with an explanation, can be found in ``defaults/outliers.conf``. We recommend starting from this file when running ee-outliers yourself.
Continue reading for all the details on how to define your own outlier detection use cases.

### Running in interactive mode
In this mode, ee-outliers will run once and finish. This is the ideal run mode to use when testing ee-outliers straight from the command line.

Running ee-outliers in interactive mode:

```
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run --network=sensor_network -v "$PWD/defaults:/mappedvolumes/config" -i  outliers-dev:latest python3 outliers.py interactive --config /mappedvolumes/config/outliers.conf
```

### Running in daemon mode
In this mode, ee-outliers will continuously run based on a cron schedule defined in the outliers configuration file.

Example from the default configuration file which will run ee-outliers at 00:00 each night:

```
[daemon]
schedule=0 0 * * *
```

Running ee-outliers in daemon mode:

```
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run --network=sensor_network -v "$PWD/defaults:/mappedvolumes/config" -d outliers-dev:latest python3 outliers.py daemon --config /mappedvolumes/config/outliers.conf
```

### Customizing your Docker run parameters

The following modifications might need to be made to the above commands for your specific situation:
- The name of the docker network through which the Elasticsearch cluster is reachable (``--network``)
- The mapped volumes so that your configuration file can be found (``-v``). By default, the default configuration file in ``/defaults`` is mapped to ``/mappedvolumes/config``
- The path of the configuration file (``--config``)

## Configuring outlier detection models

The different detection use cases can be configured in the configuration file, passed to ee-outliers. In this section we discuss all the different detection mechanisms that are available, and the options they provide to the analyst.

The different types of detection models that can be configured are listed below.

- **simplequery models**: this model will simply run an Elasticsearch query and tag all the matching events as outliers. No additional statistical analysis is done. Example use case: tag all the events that contain the string "mimikatz" as outliers.

- **metrics models**: the metrics model looks for outliers based on a calculated metric of a specific field of events. These metrics include the length of a field, its entropy, and more. Example use case: tag all events that represent Windows processes that were launched using a high number of base64 encoded parameters in order to detect obfuscated fileless malware.

- **terms models**:  the terms model looks for outliers by calculting rare combinations of a certain field(s) in combination with other field(s). Example use case: tag all events that represent Windows network processes that are rarely observed across all reporting endpoints in order to detect C2 phone home activity.

- **beaconing models**:  the beaconing model can be used to look for events that occur repeatedly with a fixed time interval. Example use case: look for signs of a piece of malware sending out beacons to a Command & Control server at fixed time intervals each minute, hour or day. 

- **word2vec models (BETA)**: the word2vec model is the first Machine Learning model defined in ee-outliers. It allows the analyst to train a model based on a set of features that are expected to appear in the same context. After initial training, the model is then able to spot anomalies in unexected combinations of the trained features. Example use case: train a model to learn which usernames, workstations and user roles are expected to appear together in order to alert on breached Windows accounts that are used to laterally move in the network.


### General model parameters
**es_query_filter**

Each model starts with an Elasticsearch query which selects which events the model should consider for analysis.
The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.
 
**trigger_on**

possible values: ``low``, ``high``. 
This parameter defines if the outliers model should trigger whenever the calculated model value of the event is lower or higher than the decision boundary. For example, a model that should trigger on users that log into a statistically high number of workstations should trigger on ``high`` values, whereas a model that should detect processes that rarely communicate on the network should trigger on ``low`` values.

**trigger_method** and **trigger_sensitivity**

Possible ``trigger_method`` values:

 - ``percentile``: percentile. ``trigger_sensitivity`` ranges from ``0-100``.
 - ``pct_of_max_value``: percentage of maximum value. ``trigger_sensitivity`` ranges from ``0-100``.
 - ``pct_of_median_value``: percentage of median value. ``trigger_sensitivity`` ranges from ``0-100``.
 - ``pct_of_avg_value``: percentage of average value. ``trigger_sensitivity`` ranges from ``0-100``.
 - ``mad``: Median Average Deviation. ``trigger_sensitivity`` defines the total number of deviations and ranges from``0-Inf.``.
 - ``madpos``: same as ``mad`` but the trigger value will always be positive. In case mad is negative, it will result 0.
 - ``stdev``: Standard Deviation. ``trigger_sensitivity`` defines the total number of deviations and ranges from``0-Inf.``.
 - ``float``: fixed value to trigger on. ``trigger_sensitivity`` defines the trigger value.

**outlier_type**

Freetext field which will be added to the outlier event as new field named ``outliers.outlier_type``.
For example: encoded commands

**outlier_reason**

Freetext field which will be added to the outlier event as new field named ``outliers.reason``.
For example: base64 encoded command line arguments

**outlier_summary**

Freetext field which will be added to the outlier event as new field named ``outliers.summary``.
For example: base64 encoded command line arguments for process {OsqueryFilter.name}

**run_model**

Switch to enable / disable running of the model

**test_model**

Switch to enable / disable testing of the model

**should_notify**

Switch to enable / disable notifications for the model

## simplequery models

This model will simply run an Elasticsearch query and tag all the matching events as outliers. No additional statistical analysis is done. Example use case: tag all the events that represent hidden powershell processes as outliers.

Each metrics model section in the configuration file should be prefixed by ``simplequery_``.

**Example model**
```
##############################
# SIMPLEQUERY - POWERSHELL EXECUTION IN HIDDEN WINDOW
##############################
[simplequery_powershell_execution_hidden_window]
es_query_filter=tags:endpoint AND "powershell.exe" AND (OsqueryFilter.cmdline:"-W hidden" OR OsqueryFilter.cmdline:"-WindowStyle Hidden")

outlier_type=powershell
outlier_reason=powershell execution in hidden window
outlier_summary=powershell execution in hidden window

run_model=1
test_model=0
```

**Parameters**

All required options are visible in the example, and are required.

## metrics models

The metrics model looks for outliers based on a calculated metric of a specific field of events. These metrics include the length of a field, its entropy, and more. Example use case: tag all events that represent Windows processes that were launched using a high number of base64 encoded parameters in order to detect obfuscated fileless malware.

Each metrics model section in the configuration file should be prefixed by ``metrics_``.

**Example model**

```
##############################
# METRICS - BASE64 ENCODED COMMAND LINE ARGUMENTS
##############################
[metrics_cmdline_containing_url]
es_query_filter=tags:endpoint AND _exists_:OsqueryFilter.cmdline

aggregator=OsqueryFilter.name
target=OsqueryFilter.cmdline
metric=url_length
trigger_on=high
trigger_method=mad
trigger_sensitivity=3

outlier_reason=cmd line args containing URL
outlier_summary=cmd line args contains URL for process {OsqueryFilter.name}
outlier_type=command execution,command & control

run_model=1
test_model=0
should_notify=0
```

**Parameters**

All required options are visible in the example, and are required.

**How it works**

The metrics model looks for outliers based on a calculated metric of a specific field of events. These metrics include the following:
- ``numerical_value``: use the numerical value of the target field as metric. Example: numerical_value("2") => 2
- ``length``: use the target field length as metric. Example: length("outliers") => 8
- ``entropy``: use the entropy of the field as metric. Example: entropy("houston") => 2.5216406363433186
- ``hex_encoded_length``: calculate total length of hexadecimal encoded substrings in the target and use this as metric.
- ``base64_encoded_length``: calculate total length of base64 encoded substrings in the target and use this as metric. Example: base64_encoded_length("houston we have a cHJvYmxlbQ==") => base64_decoded_string: problem, base64_encoded_length: 7
- ``url_length``: extract all URLs from the target value and use this as metric. Example: url_length("why don't we go http://www.dance.com") => extracted_urls_length: 20, extracted_urls: http://www.dance.com

The metrics model works as following:

 1. The model starts by taking into account all the events defined in the ``es_query_filter``
 parameter. This should be a valid Elasticsearch query. The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.
 
 2. The model then calculates the selected metric (``url_length`` in the example) for each encountered value of the ``target`` field (``OsqueryFilter.cmdline`` in the example). These values are the checked for outliers in buckets defined by the values of the ``aggregator``field (``OsqueryFilter.name`` in the example). Sensitivity for deciding if an event is an outlier is done based on the ``trigger_method`` (MAD or Mean Average Deviation in this case) and the ``trigger_sensitivity`` (in this case 3 standard deviations).

3. Outlier events are tagged with a range of new fields, all prefixed with ``outliers.<outlier_field_name>``. 

## terms models

The terms model looks for outliers by calculting rare combinations of a certain field(s) in combination with other field(s). Example use case: tag all events that represent Windows network processes that are rarely observed across all reporting endpoints in order to detect C2 phone home activity.

Each metrics model section in the configuration file should be prefixed by ``terms_``.

**Example model**
```
##############################
# TERMS - RARE PROCESSES WITH OUTBOUND CONNECTIVITY
##############################
[terms_rarely_seen_outbound_connections]
es_query_filter=tags:endpoint AND meta.command.name:"get_outbound_conns" AND -OsqueryFilter.remote_port.raw:0 AND -OsqueryFilter.remote_address.raw:127.0.0.1 AND -OsqueryFilter.remote_address.raw:"::1"

aggregator=OsqueryFilter.name
target=meta.hostname
target_count_method=across_aggregators
trigger_on=low
trigger_method=pct_of_max_value
trigger_sensitivity=5

outlier_type=outbound connection
outlier_reason=rare outbound connection
outlier_summary=rare outbound connection: {OsqueryFilter.name}

run_model=1
test_model=0
```

**Parameters**

All required options are visible in the example, and are required.

**How it works**

The terms model looks for outliers by calculting rare combinations of a certain field(s) in combination with other field(s).It works as following:

1. The model starts by taking into account all the events defined in the ``es_query_filter`` parameter. This should be a valid Elasticsearch query. The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.
 
2. The model will then count all unique instances of the ``target`` field, for each individual ``aggregator``field. In the example above, the ``OsqueryFilter.name`` field represents the process name. The target field ``meta.hostname`` represents the total number of hosts that are observed for that specific aggregator (meaning: how many hosts are observed to be running that process name which is communicating with the outside world?). Events where the communicating process is observed on less than 5 percent of all the observed hosts that contain communicating processes will be flagged as being an outlier.

3. Outlier events are tagged with a range of new fields, all prefixed with ``outliers.<outlier_field_name>``. 

The ``target_count_method`` parameter can be used to define if the analysis should be performed across all values of the aggregator at the same time, or for each value of the aggregator separately. 

## beaconing models

the beaconing model can be used to look for events that occur repeatedly with a fixed time interval. Example use case: look for signs of a piece of malware sending out beacons to a Command & Control server at fixed time intervals each minute, hour or day. 
Each metrics model section in the configuration file should be prefixed by ``beaconing_``.

**Example model**

```
##############################
# DERIVED FIELDS
##############################
[derivedfields]
# These fields will be extracted from all processed events, and added as new fields in case an outlier event is found.
# The format for the new field will be: outlier.<field_name>, for example: outliers.initials
# The format to use is GROK. These fields are extracted BEFORE the analysis happens, which means that these fields can also be used as for example aggregators or targets in use cases.
timestamp=%{YEAR:timestamp_year}-%{MONTHNUM:timestamp_month}-%{MONTHDAY:timestamp_day}[T ]%{HOUR:timestamp_hour}:?%{MINUTE:timestamp_minute}(?::?%{SECOND:timestamp_second})?%{ISO8601_TIMEZONE:timestamp_timezone}?

##############################
# BEACONING - DETECT OUTBOUND SSL BEACONING - TLS
##############################
[beaconing_ssl_outbound]
es_query_filter=BroFilter.event_type:"ssl.log" AND _exists_:BroFilter.server_name

aggregator=BroFilter.server_name,BroFilter.id_orig_h,timestamp_day
target=timestamp_hour
trigger_sensitivity=1

outlier_type=suspicious connection
outlier_reason=beaconing TLS connection
outlier_summary=beaconing TLS connection to {BroFilter.server_name}

run_model=1
test_model=0
```

**Parameters**

All required options are visible in the example, and are required.

**How it works**

The beaconing model works as following:

The model starts by taking into account all the events defined in the ``es_query_filter`` parameter.
This should be a valid Elasticsearch query. The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.
 
The model will then count all unique instances of the target field, for each individual aggregator field.
In this specific case, this means that ee-outliers will create “buckets” for each hour of the day (timestamp_hour – one of the derived fields we created earlier) and fill these buckets for each unique combination of the aggregator.

As an example: let’s say that there are events for TLS connections in the cluster to the domain “sneaky.com” that appear about 5 each hour, for a specific source IP (192.168.0.2) for a specific day (19/12).
ee-outliers will then create the following buckets in order to spot outliers:

```
Aggregator: "sneaky.com - 192.168.0.2 - 19/12"
Target: 00 (midnight)
Total count: 5

Aggregator: "sneaky.com - 192.168.0.2 - 19/12"
Target: 01 (01:00 AM)
Total count: 4

Aggregator: "sneaky.com - 192.168.0.2 - 19/12"
Target: 02 (02:00 AM)
Total count: 5
...
```

These buckets will be created for ALL combinations possible for the aggregator.
In this case, for all combinations of unique server names, source IPs and days in the range of the events processed by ee-outliers.

In order to give the model access to these timestamp fields, we need to calculate some derived fields, based on the timestamp.
For this example, this can be done as following:

```
##############################
# DERIVED FIELDS
##############################
[derivedfields]
# These fields will be extracted from all processed events, and added as new fields in case an outlier event is found.
# The format for the new field will be: outlier.<field_name>, for example: outliers.initials
# The format to use is GROK. These fields are extracted BEFORE the analysis happens, which means that these fields can also be used as for example aggregators or targets in use cases.
timestamp=%{YEAR:timestamp_year}-%{MONTHNUM:timestamp_month}-%{MONTHDAY:timestamp_day}[T ]%{HOUR:timestamp_hour}:?%{MINUTE:timestamp_minute}(?::?%{SECOND:timestamp_second})?%{ISO8601_TIMEZONE:timestamp_timezone}?
```

The trigger sensitivity finally defines how many “standard deviations” tolerance we allow in order to still consider something beaconing.
In our example above, our bucket for 01:00 AM only has 4 requests instead of 5.
Without some tolerance, these would thus not be spotted as being outliers!
By defining the trigger sensitivity and setting it to 1 (or higher for more tolerance), we allow for small changes in the bucket counts to still be considered outliers.
For example, the following 24 count values (1 for each hour of the day) would still be flagged as beaconing with a trigger_sensitivity set to 1:

```
5 5 5 4 4 5 5 5 5 3 5 5 5 2 5 5 5 5 4 5 5 5 5 5
```

In the above example, the standard deviation is 0.74; as it’s smaller than 1, all the events beloning to these 24 buckets will be flagged as outliers.
The “beaconing” model has a built-in requirement where at least 10 buckets should be available; otherwise, no beaconing will be detected
(in other words: if the series above only had 9 values instead of 24 or the minimum of 10, it would not be flagged as outliers).

Beaconing events are tagged with a range of new fields, all prefixed with ``outliers.<outlier_field_name>``. 

## Whitelisting

ee-outliers provides support for whitelisting of certain outliers. By whitelisting an outlier, you prevent them from being tagged and stored in Elasticsearch.

For events that have already been enriched and that match a whitelist later, the ``es_wipe_all_whitelisted_outliers`` flag can be used in order to remove them.
The whitelist will then be checked for hits periodically as part of the housekeeping work, as defined in the parameter ``housekeeping_interval_seconds``.

Two different whitelists are defined in the configuration file:

### Literals whitelist

This whitelist will only hit for outlier events that contain an exact whitelisted string as one of its event field values.
The whitelist is checked against all the event fields, not only the outlier fields!

Example:
```
[whitelist_literals]
slack_connection=rare outbound connection: Slack.exe
```


### Regular expression whitelist

This whitelist will hit for all outlier events that contain a regular expression match against one of its event field values.
The whitelist is checked against all the event fields, not only the outlier fields!

Example:
```
[whitelist_regexps]
scheduled_task_user_specific_2=^.*rare scheduled task:.*-.*-.*-.*-.*$
autorun_user_specific=^.*rare autorun:.*-.*-.*-.*-.*$
```

## Developing and debugging ee-outliers

ee-outliers supports additional run modes (in addition to interactive and daemon) that can be useful for developing and debugging purposes.

### Running in test mode
In this mode, ee-outliers will run all unit tests and finish, providing feedback on the test results.

Running outliers in tests mode:

```
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run -v "$PWD/defaults:/mappedvolumes/config" -i outliers-dev:latest python3 outliers.py tests --config /mappedvolumes/config/outliers.conf
```

### Running in profiler mode
In this mode, ee-outliers will run a performance profile (``py-spy``) in order to give feedback to which methods are using more or less CPU and memory. This is especially useful for developers of ee-outliers.

Running outliers in profiler mode:

```
# Build the image
docker build -t "outliers-dev" .

# Run the image
docker run --cap-add SYS_PTRACE -t --network=sensor_network -v "$PWD/defaults:/mappedvolumes/config" -i  outliers-dev:latest py-spy -- python3 outliers.py interactive --config /mappedvolumes/config/outliers.conf
```

### Checking code style, PEP8 compliance & dead code
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

## Screenshots

<p align="center"> 
<img alt="Detecting beaconing TLS connections using ee-outliers" src="https://forever.daanraman.com/screenshots/Beaconing%20detection.png?raw=true" width="650"/><br/>
<i>Detecting beaconing TLS connections using ee-outliers</i>
</p>
<br/><br/>  
<p align="center"> 
<img alt="Configured use case to detect beaconing TLS connections" src="https://forever.daanraman.com/screenshots/Configuration%20use%20case.png?raw=true" width="450"/><br/>
<i>Configured use case to detect beaconing TLS connections</i>
</p>
<br/><br/>
<p align="center"> 
<img alt="Detected outlier events are enriched with new fields in Elasticsearch" src="https://forever.daanraman.com/screenshots/Enriched%20outlier%20event%202.png?raw=true" width="650"/><br/>
<i>Detected outlier events are enriched with new fields in Elasticsearch</i>
</p>

## License

See the [LICENSE](LICENSE) file for details

## Contact

You can reach out to the developers of ee-outliers by creating an issue in github.
For any other communication, you can reach out by sending us an e-mail at research@nviso.be.

Thank you for using ee-outliers and we look forward to your feedback! 🐀

## Acknowledgements

We are grateful for the support received by [INNOVIRIS](https://innoviris.brussels/) and the Brussels region in funding our Research & Development activities. 


