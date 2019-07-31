<p align="left"><a href="INSTALL.md">&#8592; Getting started</a></p>

# Building detection use cases

**Table of contents**
- [Existing detection models](#existing-detection-models)
- [General model parameters](#general-model-parameters)
- [simplequery models](#simplequery-models)
- [metrics models](#metrics-models)
- [terms models](#terms-models)
- [beaconing models](#beaconing-models)


## Existing detection models
The different detection use cases can be configured in the configuration file, passed to ee-outliers. In this section we discuss all the different detection mechanisms that are available, and the options they provide to the analyst.

The different types of detection models that can be configured are listed below.

- **simplequery models**: this model will simply run an Elasticsearch query and tag all the matching events as outliers. No additional statistical analysis is done. Example use case: tag all the events that contain the string "mimikatz" as outliers.

- **metrics models**: the metrics model looks for outliers based on a calculated metric of a specific field of events. These metrics include the length of a field, its entropy, and more. Example use case: tag all events that represent Windows processes that were launched using a high number of base64 encoded parameters in order to detect obfuscated fileless malware.

- **terms models**:  the terms model looks for outliers by calculting rare combinations of a certain field(s) in combination with other field(s). Example use case: tag all events that represent Windows network processes that are rarely observed across all reporting endpoints in order to detect C2 phone home activity.

- **word2vec models (BETA)**: the word2vec model is the first Machine Learning model defined in ee-outliers. It allows the analyst to train a model based on a set of features that are expected to appear in the same context. After initial training, the model is then able to spot anomalies in unexected combinations of the trained features. Exampleuse case: train a model to learn which usernames, workstations and user roles are expected to appear together in order to alert on breached Windows accounts that are used to laterally move in the network.


## General model parameters
**es_query_filter**

Each model starts with an Elasticsearch query which selects which events the model should consider for analysis.
The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.

**trigger_on**

possible values: ``low``, ``high``.
This parameter defines if the outliers model should trigger whenever the calculated model value of the event is lower or higher than the decision boundary. For example, a model that should trigger on users that log int
o a statistically high number of workstations should trigger on ``high`` values, whereas a model that should detect processes that rarely communicate on the network should trigger on ``low`` values.

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
 - ``coeff_of_variation``: Coefficient of variation. ``trigger_sensitivity`` defines the comparison value (the value of each document is not taking into account). Value with a range from ``0-Inf.``.

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

--------------------

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


<p align="right"><a href="WHITELIST.md">Whitelist system &#8594;</a></p>