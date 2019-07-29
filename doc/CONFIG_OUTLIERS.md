# Configuring outlier detection models

- [Existing detection models](#existing-detection-models)
- [General model parameters](#general-model-parameters)


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





