<p align="left"><a href="GETTING_STARTED.md">&#8592; Getting started</a></p>

# Building detection use cases

**Table of contents**
- [Existing detection models](#existing-detection-models)
- [General model parameters](#general-model-parameters)
- [Arbitrary parameters](#arbitrary-parameters)
- [simplequery models](#simplequery-models)
- [metrics models](#metrics-models)
- [terms models](#terms-models)
- [sudden_appearance models](#sudden-appearance-models)
- [word2vec models](#word2vec-models)
- [Derived fields](#derived-fields)


## Existing detection models
In this section we discuss all the different detection mechanisms that are available, and the options they provide to the analyst.

The different types of detection models that can be configured are listed below.

- **simplequery models**: this model will simply run an Elasticsearch query and tag all the matching events as outliers. No additional statistical analysis is done. Example use case: tag all the events that contain the string "mimikatz" as outliers.

- **metrics models**: the metrics model looks for outliers based on a calculated metric of a specific field of events. These metrics include the length of a field, its entropy, and more. Example use case: tag all events that represent Windows processes that were launched using a high number of base64 encoded parameters in order to detect obfuscated fileless malware.

- **terms models**:  the terms model looks for outliers by calculating rare combinations of a certain field(s) in combination with other field(s). Example use case: tag all events that represent Windows network processes that are rarely observed across all reporting endpoints in order to detect C2 phone home activity.

- **sudden_appearance models**: the sudden_appearance model looks for outliers by finding te sudden appearance of a 
certain field(s).

- **word2vec models (BETA)**: the word2vec model is the first Machine Learning model defined in ee-outliers. It allows 
the analyst to train a model based on a set of features that are expected to appear in the same context. After initial 
training, the model is then able to spot anomalies in unexpected combinations of the trained features. 
Example use case: train a model to spot usernames that doesn't respect the convention of your enterprise.

The different use cases are defined in configuration files.
Note that one or multiple different detection use cases can be specified in one configuration file.
## General model parameters
**es_query_filter**

Each model starts with an Elasticsearch query which selects which events the model should consider for analysis.
The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.

**es_dsl_filter**

Specify an DSL filter on each Elasticsearch query

**timestamp_field**

Override the general settings "timestamp_field" that allow to specified the field name representing the event timestamp in Elasticsearch

**history_window_days**

Specify how many days back in time to process events and search for outliers. This value is combine with "history_window_hours" which specified the number of hours.

**history_window_hours**

See description "history_window_days".

**should_notify**

Switch to enable / disable notifications for the model

**use_derived_fields**

Enable or not the utilisation of derived field

**es_index**

Possibility of override the `es_index_pattern` parameter

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

### Usual model parameters
**trigger_on**

Possible values: ``low``, ``high``.
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
 - ``coeff_of_variation``: Coefficient of variation. ``trigger_sensitivity`` defines the comparison value (the value of each document is not taking into account). Value with a range from ``0-Inf.``.

**process_documents_chronologically**
Force Elasticsearch to give result in chronological order or not.

**target**
The document field that will be used to do the computation (based on the `trigger_method` selected).

**aggregator**
One or multiple document fields that will be used to group documents.


## Arbitrary parameters
It is also possible to add arbitrary parameters that will simply be copied into the outlier information. Note that these parameters will be taken into account when evaluating the [whitelist](WHITELIST.md).  Also note that placeholders are not supported here.   
These arbitrary parameters could not start with prefix `whitelist_` (which will be used to process [per model whitelist](WHITELIST.md#whitelist-per-model)).

<details>
<summary>Example</summary>

```ini
##############################
# SIMPLEQUERY - NETWORK TROJAN DETECTED
##############################h
[simplequery_suricata_network_trojan_detected]
es_query_filter = _exists_:smoky_filter_name AND smoky_filter_name.keyword:SuricataFilter  AND  SuricataFilter.event_type.keyword:alert AND SuricataFilter.alert.category.keyword:"A Network Trojan was detected"

outlier_type = IDS
outlier_reason = network trojan detected
outlier_summary = {SuricataFilter.alert.signature}
test_arbitrary_key=arbitrary_value

run_model = 1
test_model = 0
```
should then result in an event:

```json
{
    "outliers": {
          "test_arbitrary_key": "arbitrary_value"
     }
}
```

</details>

--------------------

## simplequery models

This model will simply run an Elasticsearch query and tag all the matching events as outliers. No additional statistical analysis is done. Example use case: tag all the events that represent hidden powershell processes as outliers.

Each metrics model section in the configuration file should be prefixed by ``simplequery_``.

**Example model**
```ini
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

All required options are visible in the example. All possible options are listed [here](CONFIG_PARAMETERS.md#common-analyzers-parameters).

## metrics models

The metrics model looks for outliers based on a calculated metric of a specific field of events. These metrics include the length of a field, its entropy, and more. Example use case: tag all events that represent Windows processes that were launched using a high number of base64 encoded parameters in order to detect obfuscated fileless malware.

Each metrics model section in the configuration file should be prefixed by ``metrics_``.

**Example model**

```ini
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

All required options are visible in the example. All possible options are listed [here](CONFIG_PARAMETERS.md#analyzers-parameters).

**How it works**

The metrics model looks for outliers based on a calculated metric of a specific field of events. These metrics include the following:
- ``numerical_value``: use the numerical value of the target field as metric. Example: numerical_value("2") => 2
- ``length``: use the target field length as metric. Example: length("outliers") => 8
- ``entropy``: use the entropy of the field as metric. Example: entropy("houston") => 2.5216406363433186
- ``hex_encoded_length``: calculate total length of hexadecimal encoded substrings in the target and use this as metric.
- ``base64_encoded_length``: calculate total length of base64 encoded substrings in the target and use this as metric. Example: base64_encoded_length("houston we have a cHJvYmxlbQ==") => base64_decoded_string: problem, base64_encoded_length: 7
- ``url_length``: extract all URLs from the target value and use this as metric. Example: url_length("why don't we go http://www.dance.com") => extracted_urls_length: 20, extracted_urls: http://www.dance.com
- ``relative_english_entropy``: compute Kullback Leibler entropy.

The metrics model works as following:

 1. The model starts by taking into account all the events defined in the ``es_query_filter``
 parameter. This should be a valid Elasticsearch query. The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.
 
 2. The model then calculates the selected metric (``url_length`` in the example) for each encountered value of the ``target`` field (``OsqueryFilter.cmdline`` in the example). These values are the checked for outliers in buckets defined by the values of the ``aggregator``field (``OsqueryFilter.name`` in the example). Sensitivity for deciding if an event is an outlier is done based on the ``trigger_method`` (MAD or Mean Average Deviation in this case) and the ``trigger_sensitivity`` (in this case 3 standard deviations).

3. Outlier events are tagged with a range of new fields, all prefixed with ``outliers.<outlier_field_name>``. 

## terms models

The terms model looks for outliers by calculting rare combinations of a certain field(s) in combination with other field(s). Example use case: tag all events that represent Windows network processes that are rarely observed across all reporting endpoints in order to detect C2 phone home activity.

Each terms model section in the configuration file should be prefixed by ``terms_``.

**Example model**
```ini
##############################
# TERMS - RARE PROCESSES WITH OUTBOUND CONNECTIVITY
##############################
[terms_rarely_seen_outbound_connections]
es_query_filter=tags:endpoint AND meta.command.name:"get_outbound_conns" AND -OsqueryFilter.remote_port.keyword:0 AND -OsqueryFilter.remote_address.keyword:127.0.0.1 AND -OsqueryFilter.remote_address.keyword:"::1"

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

All required options are visible in the example. All possible options are listed [here](CONFIG_PARAMETERS.md#analyzers-parameters).

**How it works**

The terms model looks for outliers by calculating rare combinations of a certain field(s) in combination with other field(s). It works as following:

1. The model starts by taking into account all the events defined in the ``es_query_filter`` parameter. This should be a valid Elasticsearch query. The best way of testing if the query is valid is by copy-pasting it from a working Kibana query.
 
2. The model will then count all unique instances of the ``target`` field, for each individual ``aggregator``field. In the example above, the ``OsqueryFilter.name`` field represents the process name. The target field ``meta.hostname`` represents the total number of hosts that are observed for that specific aggregator (meaning: how many hosts are observed to be running that process name which is communicating with the outside world?). Events where the communicating process is observed on less than 5 percent of all the observed hosts that contain communicating processes will be flagged as being an outlier.

3. Outlier events are tagged with a range of new fields, all prefixed with ``outliers.<outlier_field_name>``. 

The ``target_count_method`` parameter can be used to define if the analysis should be performed across all values of the aggregator at the same time, or for each value of the aggregator separately. 

**Special case**

If `trigger_method` is set on `coeff_of_variation`, the process is not completely the same. Indeed, the coefficient of variation is compute like other metrics, based on the number of document for a specific `target` and `aggregator`. But this coefficient of variation is then compare to the `trigger_sensitivity`. Based on `trigger_on`, all the group is mark as outlier or not.

This method could be used for detecting an occurrence in events. Example use case: look for signs of a piece of malware sending out beacons to a Command & Control server at fixed time intervals each minute, hour or day.

**Example model**

```ini
##############################
# DERIVED FIELDS
##############################
[derivedfields]
# These fields will be extracted from all processed events, and added as new fields in case an outlier event is found.
# The format for the new field will be: outlier.<field_name>, for example: outliers.initials
# The format to use is GROK. These fields are extracted BEFORE the analysis happens, which means that these fields can also be used as for example aggregators or targets in use cases.
timestamp=%{YEAR:timestamp_year}-%{MONTHNUM:timestamp_month}-%{MONTHDAY:timestamp_day}[T ]%{HOUR:timestamp_hour}:?%{MINUTE:timestamp_minute}(?::?%{SECOND:timestamp_second})?%{ISO8601_TIMEZONE:timestamp_timezone}?

##############################
# TERMS - DETECT OUTBOUND SSL TERMS - TLS
##############################
[terms_ssl_outbound]
es_query_filter=BroFilter.event_type:"ssl.log" AND _exists_:BroFilter.server_name

aggregator=BroFilter.server_name,BroFilter.id_orig_h,timestamp_day
target=timestamp_hour
target_count_method=within_aggregator
trigger_on=low
trigger_method=coeff_of_variation
trigger_sensitivity=0.1

outlier_type=suspicious connection
outlier_reason=terms TLS connection
outlier_summary=terms TLS connection to {BroFilter.server_name}

run_model=1
test_model=0
```

## Sudden Appearance models
The sudden_appearance model looks for outliers by finding te sudden appearance of a certain field(s).
Example use case: tag sudden appearance of a new TLD DNS. 

Each sudden_appearance model section in the configuration file should be prefixed by `sudden_appearance_`.

**Example model**
```ini
##############################
# SUDDEN APPEARANCE - NEW DNS DOMAIN
##############################
[sudden_appearance_new_dns_domain]
es_query_filter=tags:network AND type:dns AND direction:outbound

aggregator=meta.deployment_name
target= dns_request 

slide_window_days=1
slide_window_hours=0
slide_window_mins=0

slide_jump_days=0
slide_jump_hours=12
slide_jump_mins=0

trigger_slide_window_proportion = 0.8

outlier_type=first observation
outlier_reason=sudden appearance of DNS domain in logs
outlier_summary=sudden appearance of DNS domain {dns_request} in logs

run_model=1
test_model=0
```

**Parameters**

All required options are visible in the example. All possible options are listed [here](CONFIG_PARAMETERS.md#analyzers-parameters).

**How it works**

The sudden_appearance model looks for outliers by finding te sudden appearance of a certain field(s).

Let's define:
- The **global window** determined by the parameters `history_window_days` and `history_window_hours`.
- The **slide window** where the size is determined by the parameters `slide_window_days`, `slide_window_hours` and 
`slide_window_mins`. It has to be smaller than the **global window**.
- The **slide jump** where the size is determined by the parameters `slide_jump_days`, `slide_jump_hours` and 
`slide_jump_mins`. It represents the jump in time, the **slide window** will slide within the **global window**.

The sudden_appearance model works as following:
1. The **slide window** is first placed at the beginning of the **global window**.
2. An analysis of the sudden appearance of a certain field(s) is done in the **slide window**.
More especially, it will take the first occurrence of a certain field(s) in each group of aggregation. It then compute 
in what proportion of the **slide window** it this first occurrence appears. If it appears at a proportion bigger than 
`trigger_slide_window_proportion`, the event will be considered as an outlier.
3. After, the **slide window** slide/jump further in the **global window**, by a time distance defined by 
**slide jump**. 
4. The operation 2. and 3. are repeated until the **slide window** gets through all the **global window**.


## Word2vec models
The word2vec model looks for outliers by analysing weird syntactic and semantic arrangement in an event text field(s).
More precisely, in each event, it will take the text of a certain field(s). 
Then, these texts will be separated into tokens/words which will be used as input to train a Skip-Gram word2vec model.
During evaluation time, word2vec will output for each word, a score that is dependent to his neighborhood words. 
More the word score is low and more the word or his neighborhood words can be seen as anomalies.
Word2vec can also return a general score around the entire text.

Example use case: spot processes that are running in an unusual directory.

Each word2vec model section in the configuration file should be prefixed by ``word2vec_``.

**Example model**
```ini
##############################
# WORD2VEC - SUSPICIOUS PROCESS DIRECTORY
##############################
[word2vec_suspicious_process_directory]
es_query_filter=_exists_:Image
target=Image
aggregator=User

word2vec_batch_eval_size = 10000
min_target_buckets = 3000

use_prob_model=0
seed=43

separators="\\"
size_window=2

print_score_table=1

trigger_focus=word
trigger_score=center
trigger_on=low
trigger_method=stdev
trigger_sensitivity=6

outlier_type=process execution
outlier_reason=suspicious process directory
outlier_summary=suspicious process directory: {Image}

run_model=1
test_model=0
```

**Parameters**

All required options are visible in the example. All possible options are listed [here](CONFIG_PARAMETERS.md#analyzers-parameters).

**How it works**

The word2vec model looks for outliers by analysing weird syntax arrangement of a certain field(s). It works as following:

1. The model starts by taking into account all the events defined in the ``es_query_filter`` parameter. This should be a
 valid Elasticsearch query. The best way of testing if the query is valid is by copy-pasting it from a working Kibana 
 query.

2. The model will then take all instances of the ``target`` field and group them into aggregation defined by the 
parameter ``aggregator``. Each aggregation will create an independent word2vec model. 
In the example above, the ``Image`` field  is the name of the process executed including the full directory path to the 
executable when the ``User`` field let you know who created the process in question.
It will therefore, take the instances of ``Image`` and group them by ``User``.

3. Afterward, each instance of the ``target`` field grouped into aggregation will be tokenized.
More exactly, it will split the **text** of the ``target`` field into **words** by the occurrence of the 
``separator`` field.
If we look at the example above, we know that each instance/text of ``target=Image``  will look like this: 
    
    ``C:\Windows\dir\sub_dir\program.exe``

    Therefore, with ``separator="\\\\"``, it will split the text as follow:

    | text x |   |
    |--------|---|
    | word 1 | C:  |
    | word 2 | Windows |
    | word 3 | dir |
    | word 4 | sub_dir |
    | word 5 |  program.exe |

4. Then it will train a word2vec neural network to do the following; given a specific **center word** in a 
text, try to guess which **context word** will appear in his neighborhood. The **context words** of a **center word** are 
 words contained inside the **center word** window defined by the parameter ``size_window``. 
In inference, given as input a certain **center word** and **context word**, the word2vec network will output a
probability (``output_prob=1``) or a raw value (``output_prob=0``). 

    Given the text example above and a size window of 1, the outputs of the word2vec neural network will give us the 
    following results:
    
    | Center word | Context word | Probability | Raw Value |
    |-------------|--------------|-------------|-----------|
    | C:          | Windows      | P1          | RV1
    | Windows     | C:           | P2          | RV2
    | Windows     | dir          | P3          | RV3
    | dir         | Windows      | P4          | RV4
    | dir         | sub_dir      | P5          | RV5
    | sub_dir     | dir          | P6          | RV6
    | sub_dir     | program.exe  | P7          | RV7
    | program.exe | sub_dir      | P8          | RV8
    
    Note, that it is also possible to use the true probability of ``P(context_word|center_word)`` by setting the 
    parameter ``use_prob_model`` to ``1``. This algorithm has less computational complexity than Word2Vec but gives
    sometimes results with more False Positive. It could be due to the fact that true probabilities will have a good 
    estimation of the syntactic rules between words but not in of the semantic. It will not understand the similarity
    between the meaning of two words.
    
5. From those probabilities/raw values, multiple scores are developed by words or texts. These scores can then be
evaluated with for example ``trigger_method=stdev``, to classify text instances as outlier or not.

    The following table resume all type of scores available:

    |                       | C:            | Windows           | dir           | sub_dir           | program.exe           | TOTAL              |
    |-----------------------|---------------|-------------------|---------------|-------------------|-----------------------|--------------------|
    | Word batch occurrence | 6000          | 400               | 30            | 10                | 300                   |                    |
    | <--Center score-->    |  C:_cntr_scr  | Windows_cntr_scr  | dir_cntr_scr  | sub_dir_cntr_scr  | program.exe_cntr_scr  | text_cntr_ttl_scr  |       
    | -->Context score<--   |  C:_cntxt_scr | Windows_cntxt_scr | dir_cntxt_scr | sub_dir_cntxt_scr | program.exe_cntxt_scr | text_cntxt_ttl_scr |
    |    Total score        |  C:_ttl_scr   | Windows_ttl_scr   | dir_ttl_scr   | sub_dir_ttl_scr   | program.exe_ttl_scr   | text_ttl_ttl_scr   |
    | MEAN                  |               |                   |               |                   |                       | text_mean_ttl_scr  |

    The type of scores are:
    - Center word score:
    
         If word2vec outputs probabilities, the **center word score** is the geometric mean of all the probability corresponding
         to one **center word** in one specific text. If it outputs the raw values, it uses arithmetic mean instead of 
         geometric mean.
         Following the example above with window size of 1, we have for example:
         - If output probabilities: <code>dir_cntr_scr = (P4 \* P5)<sup>1/2</sup></code>
         - If output raw values: <code>dir_cntr_scr = (RV4 + RV5)/2</code>
         
        If the score is **high**/**low**, it means that this word **see**/**don't see** often by the current context
        words.
    - Context word score:
    
         If word2vec outputs the probabilities, the **context word score** is the geometric mean of all the probability corresponding
         to one **context word** in one specific text. If it outputs the raw values, it uses arithmetic mean instead of 
         geometric mean.
         Following the example above with window size of 2, we have for example:
         - If output probabilities: <code>dir_cntxt_scr = (P3 \* P6)<sup>1/2</sup></code>
         - If output raw values: <code>dir_cntxt_scr = (RV3 + RV6)/2</code>
          
       If the score is **high**/**low**, it means that this word is **seen**/**not seen** often by the current context
       words.
    - Total word score:
    
        If word2vec outputs the probabilities, the **total word score** is the geometric mean of the 
        **center word score** and the **context word score** of one specific word. If it outputs the raw values, it uses 
        arithmetic mean instead of geometric mean. Following the examples above, we have:
        - If output probabilities: <code>dir_ttl_scr = (dir_cntr_scr * dir_cntxt_scr)<sup>1/2</sup></code>
        - If output raw values: <code>dir_ttl_scr = (dir_cntr_scr + dir_cntxt_scr)/2 </code>
        
        It expresses the combination of the both scores **center & context word scores**. Therefore, if a word score is 
        low, it means that this word **don't see** or/and **is not seen** often by the context words.
    - Center text score:
    
        If word2vec outputs probabilities, the **center text score** is the geometric mean of all the 
        **center word scores** for one specific text. If it outputs the raw values, it uses arithmetic mean instead of 
        geometric mean. Following the examples above, we have:
        - If output probabilities: <code> text_cntr_ttl_scr = (C:_cntr_scr  * Windows_cntr_scr  * dir_cntr_scr  * sub_dir_cntr_scr  * program.exe_cntr_scr)<sup>1/5</sup></code>
        - If output raw values: <code> text_cntr_ttl_scr = (C:_cntr_scr  + Windows_cntr_scr  + dir_cntr_scr  + sub_dir_cntr_scr  + program.exe_cntr_scr)/5</code>
    - Context test score:
    
        If word2vec outputs probabilities, the **context text score** is the geometric mean of all the 
        **context word scores** for one specific text. If it outputs the raw values, it uses arithmetic mean instead of 
        geometric mean. Following the examples above, we have:
        - If output probabilities: <code> text_cntxt_ttl_scr = (C:_cntxt_scr * Windows_cntxt_scr * dir_cntxt_scr * sub_dir_cntxt_scr * program.exe_cntxt_scr)<sup>1/5</sup></code>
        - If output raw values: <code> text_cntxt_ttl_scr = (C:_cntxt_scr + Windows_cntxt_scr + dir_cntxt_scr + sub_dir_cntxt_scr + program.exe_cntxt_scr)/5</code>
    - Total text score:
    
        If word2vec outputs probabilities, the **total text score** is the geometric mean of all the 
        **total word scores** for one specific text. If it outputs the raw values, it uses arithmetic mean instead of 
        geometric mean. Following the examples above, we have:
        - If output probabilities: <code> text_ttl_ttl_scr = (C:_ttl_scr * Windows_ttl_scr * dir_ttl_scr * sub_dir_ttl_scr * program.exe_ttl_scr)<sup>1/5</sup></code>
        - If output raw values: <code> text_ttl_ttl_scr = (C:_ttl_scr + Windows_ttl_scr + dir_ttl_scr + sub_dir_ttl_scr + program.exe_ttl_scr)/5</code>
      
    - Mean text score:
    
        If word2vec outputs probabilities, the **mean text score** is the geometric mean of all the 
        word2vec outputs for one specific text. If it outputs the raw values, it uses arithmetic mean instead of 
        geometric mean. Following the examples above, we have:
        - If output probabilities: <code> text_mean_ttl_scr = (P1 * P2 * P3 * P4 * P5 * P6 * P7 * P8)<sup>1/8</sup></code>
        - If output raw values: <code> text_mean_ttl_scr = (P1 + P2 + P3 + P4 + P5 + P6 + P7 + P8)/8</code>
    
    Note that, by experience, all this scores are able to find outliers but gives better F-score while it outputs 
    probabilities (vs raw values) are used combined with word scores (vs text scores). We still give for analyst the 
    possibility to use the alternatives because they could be benefic in other data distribution.
    
6. If in texts, semantic and syntactic rules are respected, we will expect for each unique word a similar score in each 
of the text it will appear. At he opposite, if in one text, semantic and syntactic rules are not respected, the score 
should be lower and very different compared to the score of that word in other texts. Taking this assumption, you can 
spot outliers by simply using basic statistic/trigger methods like the Standard Deviation or the Median Average Deviation which
should spot a word that return an abnormal score. This is the parameters ``trigger_on``, ``trigger_method``, and 
``trigger_sensitivity`` that will be in charge of that action.
 
7. As a last step, outlier events are tagged with a range of new fields, all prefixed with 
``outliers.<outlier_field_name>``.



**Remarks**

- It is recommended to put ``num_epoch`` between ``1`` and ``3`` not higher. 
- The default value of ``learning_rate=0.001`` gives generaly good results.
- If you want to analyse outliers directly on the standard output, you 
can put the parameter ``print_score_table`` to ``1``. It will print all outlier scores on a table and highlight in red
word scores that or out of their normal distribution.
```
+-----------------------+----------+---------------+-------------+--------------------+------------+-----------------+--------------+----------+
|                       | C:       | ProgramData   | Microsoft   | Windows Defender   | Platform   | 4.18.1908.7-0   | NisSrv.exe   | TOTAL    |
+=======================+==========+===============+=============+====================+============+=================+==============+==========+
| Word batch occurrence | 10000    | 1045          | 1189        | 1044               | 1044       | 971             | 3            |          |
+-----------------------+----------+---------------+-------------+--------------------+------------+-----------------+--------------+----------+
| <--Center score-->    | 5.76e-02 | 3.21e-01      | 2.17e-01    | 2.24e-01           | 3.68e-02   | 1.67e-02        | 1.35e-03     | 4.97e-02 |
+-----------------------+----------+---------------+-------------+--------------------+------------+-----------------+--------------+----------+
| -->Context score<--   | 2.83e-01 | 1.66e-01      | 1.96e-01    | 2.43e-01           | 7.32e-02   | 3.10e-02        | 7.72e-05     | 4.53e-02 |
+-----------------------+----------+---------------+-------------+--------------------+------------+-----------------+--------------+----------+
| Total score           | 1.28e-01 | 2.31e-01      | 2.06e-01    | 2.33e-01           | 5.19e-02   | 2.28e-02        | 3.22e-04     | 4.74e-02 |
+-----------------------+----------+---------------+-------------+--------------------+------------+-----------------+--------------+----------+
| MEAN                  |          |               |             |                    |            |                 |              | 6.57e-02 |
+-----------------------+----------+---------------+-------------+--------------------+------------+-----------------+--------------+----------+

```
- For development purpose, it is possible to use Elasticsearch labeled data and then print on the standard output a 
confusion matrix along with Precision, recall and F-score metrics. 
To do so, you will have to create a special field ``label`` for each event where each outliers are set to ``1``. 
The parameter ``print_confusion_matrix`` has to be also set to ``1``.


## Derived fields

Some fields contains multiple information, like the timestamp that could be split between year, month, etc..
Data extracted with this method could be used into models parameters.

For this example, the following configuration allow to extract timestamp information:

```ini
##############################
# DERIVED FIELDS
##############################
[derivedfields]
# These fields will be extracted from all processed events, and added as new fields in case an outlier event is found.
# The format for the new field will be: outlier.<field_name>, for example: outliers.initials
# The format to use is GROK. These fields are extracted BEFORE the analysis happens, which means that these fields can also be used as for example aggregators or targets in use cases.
timestamp=%{YEAR:timestamp_year}-%{MONTHNUM:timestamp_month}-%{MONTHDAY:timestamp_day}[T ]%{HOUR:timestamp_hour}:?%{MINUTE:timestamp_minute}(?::?%{SECOND:timestamp_second})?%{ISO8601_TIMEZONE:timestamp_timezone}?
```


<p align="right"><a href="WHITELIST.md">Whitelist system &#8594;</a></p>
