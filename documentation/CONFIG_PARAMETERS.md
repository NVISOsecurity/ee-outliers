# All parameters in configurations

**Table of contents**
- [Environment variables](#environment-variable)
- [General configuration](#general-configuration)
    - [General](#general)
    - [Assets](#assets)
    - [Notifier](#notifier)
    - [Daemon](#daemon)
    - [Simple query](#simple-query)
    - [Terms](#terms)
    - [Metrics](#metrics)
    - [Word2vec](#word2vec)
    - [Whitelist literals](#whitelist-literals)
    - [Whitelist regexps](#whitelist-regexps)
- [Analyzers parameters](#analyzers-parameters)
  - [Common analyzers parameters](#common-analyzers-parameters)
  - [Usual model parameters](#usual-model-parameters)
  - [Simple query parameters](#simple-query-parameters)
  - [Metrics parameters](#metrics-parameters)
  - [Terms parameters](#terms-parameters)
  - [Sudden Appearance parameters](#sudden-appearance-parameters)
  - [Word2vec parameters](#word2vec-parameters)

## Environment variables

The environment variables are mainly used to store sensitive information like credentials or other TLS parameters. All these environment variables are optional.

<table class="tg">
  <tr>
    <th class="tg-0pky">Variable</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Default</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_username</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky"><code>""</code></td>
    <td class="tg-0pky">Username to connect to Elasticsearch</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_password</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky"><code>""</code></td>
    <td class="tg-0pky">Password to connect to Elasticsearch</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>verify_certs</code></td>
    <td class="tg-0pky"><code>Boolean</code></td>
    <td class="tg-0pky"><code>True</code></td>
    <td class="tg-0pky">Whether the Elasticsearch certificate must be validated or not</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>ca_certs</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky"><code>None</code></td>
    <td class="tg-0pky">A path to a valid CA to validate the Elasticsearch server certificate</td>
  </tr>
</table>

## General configuration

ee-outliers makes use of a single configuration file containing all required parameters such as connectivity with your 
Elasticsearch cluster, logging, etc.

A default configuration file with all required configuration sections and parameters, along with an explanation, 
 be found in [`defaults/outliers.conf`](../defaults/outliers.conf).

### General
<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters (<small>*Mandatory</small>)</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_url</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">URL to connect to Elasticsearch. It supports https schema for TLS</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_index_pattern</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">The name of the Elasticsearch index. Can be a glob pattern such as <code>my_indexes*</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_scan_size</code>*</td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">size of the batch used by Elasticsearch for each search request.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_scroll_time</code>*</td>
    <td class="tg-0pky">[Integer][letter] where letter represents a duration (Hours, Minutes, Seconds)</td>
    <td class="tg-0pky">Specify how long a consistent view of the index should be maintained for scrolled Elasticsearch search.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_timeout</code>*</td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Explicit timeout in seconds for each Elasticsearch request.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>timestamp_field</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">The field name representing the event timestamp in Elasticsearch. 
    Default value: <code>timestamp</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_save_results</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, save outlier detection results to Elasticsearch.
    If set to <code>0</code>, do nothing.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>print_outliers_to_console</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, print outlier matches to the console.
    If set to <code>0</code>, do nothing. Default value: <code>0</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>history_window_days</code>*</td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Specify how many days back in time to process events and search for outliers. 
    This value is combine with <code>history_window_hours</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>history_window_hours</code>*</td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Specify how many hours back in time to process events and search for outliers. 
    This value is combine with <code>history_window_days</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_wipe_all_existing_outliers</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, wipe all existing outliers that fall in the history window upon first run.
    If set to <code>0</code>, do nothing.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_wipe_all_whitelisted_outliers</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, existing outliers are checked and wiped if they match with the whitelisting.
    If set to <code>0</code>, do nothing.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>run_models</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, run all use cases with key parameter <code>run_model</code> set to 1.
    If set to <code>0</code>, do nothing.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>test_models</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, run all use cases with key parameter <code>test_model</code> set to 1.
    If set to <code>0</code>, do nothing.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>log_verbosity</code>*</td>
    <td class="tg-0pky"><code>0-5+</code></td>
    <td class="tg-0pky"><code>0</code> for no progress info,  <code>1</code>-<code>4</code> for progressively more 
    outputs, <code>5+</code> for all the log output.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>log_level</code>*</td>
    <td class="tg-0pky"><code>CRITICAL</code>, <code>ERROR</code>, <code>WARNING</code>, <code>INFO</code>, 
    <code>DEBUG</code></td>
    <td class="tg-0pky">Sets the threshold for the logger. Logging messages which are less severe than level will be 
    ignored.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>log_file</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">File path where the log messages will be saved</td>
  </tr>
</table>

### Assets

It allows to extract additional information within the outliers and save them in the dictionary field 
<code>outliers.assets</code>.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters (<small>*Mandatory</small>)</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky">Any existing field name</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Example: <code>timestamp=time</code> will extract the value inside the field 
    <code>timestamp</code> and add it to in the dictionary field <code>outliers.assets</code> at the key <code>time</code>.</td>
  </tr>
</table>

### Notifier

To have more information about the notification system, visit the page [Notifications](NOTIFICATIONS.md).
<table class="tg">
  <tr>
   <th colspan="3">Notifier</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters (<small>*Mandatory</small>)</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>email_notifier</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, enable the notification system and the other key parameters from the 
    section <code>[notifier]</code>, except <code>max_cache_ignore</code>, become mandatory.
    If set to <code>0</code>, do nothing.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>notification_email</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Email where the information needs to be sent.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>smtp_user</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">SMTP username.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>smtp_pass</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">SMTP password</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>smtp_server</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">SMTP server address.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>smtp_port</code></td>
    <td class="tg-0pky"><code>int</code></td>
    <td class="tg-0pky">SMTP port.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>max_cache_ignore</code></td>
    <td class="tg-0pky"><code>int</code></td>
    <td class="tg-0pky">Number of element keep in memory to avoid twice alerts for same notification.
    Default value: <code>1000</code>.</td>
  </tr>
</table>

### Daemon

Used when ee-outliers is running on Daemon mode.
In daemon mode, ee-outliers will continuously run based on a cron schedule
which is defined by the following <code>schedule</code> parameter.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>schedule</code></td>
    <td class="tg-0pky">Standard cron format</td>
    <td class="tg-0pky">Only used when running ee-outliers in daemon mode. 
    Example: <code>schedule=10 0 * * *</code> will run ee-outliers at 00:10 each night.</td>
  </tr>
</table>

### Simple query

Global parameters for all use cases of type simplequery.

The only global parameter for simplequery use cases is <code>highlight_match</code>. 
If set to <code>1</code>, ee-outliers will use the Elasticsearch highlight mechanism to find the fields and values
that matched the search query. The matched fields and values are respectively added to new dictionary fields 
<code>outliers.matched_fields</code> and <code>outliers.matched_values</code>.

Example: If the search query is <code>es_query_filter=CurrentDirectory : sysmon AND Image: System32 AND Image: cmd.exe</code> and the log
event contains the fields:
```
CurrentDirectory: C:\sysmon\
Image: C:\Windows\System32\cmd.exe
```
It will add the fields:
```
outliers.matched_fields: {"CurrentDirectory": ["C:\\<value>sysmon</value>\\"],
                         "Image": ["C:\\Windows\\<value>System32</value>\\<value>cmd.exe</value>"]}
outliers.matched_values: {'CurrentDirectory': ['sysmon'], 'Image': ['System32', 'cmd.exe']}
```
Note that in the field <code>outliers.matched_fields</code>, the values that match the search query has been tagged as
follow: `<value>MACHTED_VALUE</value>`.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>highlight_match</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, it will use the Elasticsearch highlight mechanism to find the fields
    and values that matched the search query. The matched fields and values are respectively added to new 
    dictionary fields <code>outliers.matched_fields</code> and <code>outliers.matched_values</code>.
    If set to <code>0</code>, do nothing. Default: <code>0</code>.</td>
  </tr>
</table>

### Terms

Global parameters for all use cases of type terms.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>terms_batch_eval_size</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Define how many events should be processed at the same time, before looking for outliers.
    Bigger batch means better results, but increase the memory usage.</td>
  </tr>
</table>

### Metrics

Global parameters for all use cases of type metrics.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>metric_batch_eval_size</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Define how many events should be processed at the same time, before looking for outliers.
    Bigger batch means better results, but increase the memory usage.</td>
  </tr>
</table>

### Sudden Appearance

Global parameters for all use cases of type sudden_appearance.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>max_num_aggregators</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Maximum number of estimated aggregation. 
    If the number of aggregation defined in <code>aggregator</code> is bigger than <code>max_num_aggregators</code>, the returned results will not be accurate.
     Default: <code>100000</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>max_num_targets</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Maximum number of estimated targets.
    If the number of terms defined in <code>target</code> is bigger than <code>max_num_targets</code>, the 
    returned results will not be accurate. Default: <code>100000</code>.</td>
  </tr>
</table>

### Word2vec

Global parameters for all use cases of type word2vec.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>word2vec_batch_eval_size</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Define how many events should be processed at the same time, before looking for outliers.
    Bigger batch means better results, but increase the memory usage.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>min_target_buckets</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Minimum number of events required within an aggregation before processing word2vec analyzer.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>drop_duplicates</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, drops duplicate <code>target</code> elements within each 
    aggregation.
    If set to <code>0</code>, do nothing. Set to <code>0</code> by default.
    Note that when activated, <code>dorp_duplicates</code> can increases the memory size. The reason is that it generally 
    increase the size of the vocabulary and therefore the size of the word2vec model.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>use_prob_model</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, use a probabilistic model instead of word2vec.
    If set to <code>0</code>, use word2vec model. 
    Used mainly to evaluate the performance of word2vec. 
    The probabilistic model will compute the true probability that a context word to appear given a certain center word.
    <code>P(context_word|center_word) = (num. of time the pair context_word-center_word appears)/(num. of time center_word appears)</code>.
    Set to <code>0</code> by default.
    </td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>output_prob</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, the models output the probability that a context word appears, 
    given a certain center word.
    If set to <code>0</code>, and <code>use_prob_model=0</code> it outputs the raw value of word2vec 
    (layer before the softmax).
    If set to <code>0</code>, and <code>use_prob_model=1</code> it outputs the logarithmic of the probabilities.
    Set to <code>1</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>separators</code></td>
    <td class="tg-0pky">regex format <b>between quotes</b></td>
    <td class="tg-0pky">Will split <code>target</code> elements by the occurrence of the regex pattern. 
    Example: If <code>separators="\.| "</code> and <code>target</code> of one event is <code>"Our website is nviso.eu"
    </code>  the output tokens will became <code>["Our", "website", "is", "nviso", "eu"]</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>size_window</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Size of the context window.
    Note that as you increase the size window, the number of center word - context word combination will increase. 
    It will then result in a augmentation of memory size and computation time.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>min_uniq_word_occurrence</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">If a word appears less than <code>min_uniq_word_occurrence</code> times, it will be replaced by 
    the 'UNKNOWN' word. Set to <code>1</code> by default.
    Note that as it reduces the vocabulary size of the model, it reduces the memory size.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>num_epoch</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Number of times word2vec model trains on all events within one aggregation.
    Set to <code>1</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>learning_rate</code></td>
    <td class="tg-0pky"><code>Float</code></td>
    <td class="tg-0pky">The learning rate of the word2vec model.
    Set to <code>0.001</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>embedding_size</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Embedding size of the word2vec model.
    Set to <code>40</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>seed</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">The random seed to make word2vec deterministic. 
    If set to <code>0</code> it make word2vec non deterministic.
    If deterministic, it will also read documents chronologically and therefore reduce Elasticsearch scanning performance.
    Set to <code>0</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>print_score_table</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Print all outlier scores on a table. Set to <code>0</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>print_confusion_matrix</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Print confusion matrix and precision, recall and F-Measure metrics.
    Work only if the field "label" (equal to <code>0</code> or <code>1</code>) exist in Elasticsearch events.
    Set to <code>0</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>trigger_focus</code></td>
    <td class="tg-0pky"><code>word</code>, <code>text</code></td>
    <td class="tg-0pky">If set to <code>text</code>, it triggers events based on global text score.
    If set to <code>word</code>, it triggers events based on word score. 
    Set to <code>word</code> by default.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>trigger_score</code></td>
    <td class="tg-0pky"><code>center</code>, <code>context</code>, <code>total</code>, <code>mean</code></td>
    <td class="tg-0pky">Type of score the events are triggered on. Mean compatible only with <code>trigger_focus=text</code></td>
  </tr>
</table>

### Derived fields

Some fields contains multiple information, like <code>timestamp</code> that can be split between sub fields year, month, etc.. 

It requires any existing field name (e.g. <code>timestamp</code>) as key parameter and using the [GROK](https://www.elastic.co/guide/en/logstash/current/plugins-filters-grok.html#_grok_basics)
 format as value to extract the sub information.
The sub information will be extracted from all processed events, and added as new fields in case an outlier event is found.
The format for the new field will be: outlier.derived_<field_name> (e.g. outliers.derived_timestamp_year).


Note that, these fields are extracted **BEFORE** the analysis happens and with their original field_name 
(e.g. timestamp_year), which means that these fields can also be used as for example with aggregators or targets in use 
cases.

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky">Any existing field name</td>
    <td class="tg-0pky"><code>GROK format</code></td>
    <td class="tg-0pky">Example: <code>timestamp=%{YEAR:timestamp_year}-%{MONTHNUM:timestamp_month}-%{MONTHDAY:timestamp_day}[T ]%{HOUR:timestamp_hour}:?%{MINUTE:timestamp_minute}(?::?%{SECOND:timestamp_second})?%{ISO8601_TIMEZONE:timestamp_timezone}?</code>
     will creates from the field <code>timestamp</code> the fields <code>derived_timestamp_year</code>, 
     <code>derived_timestamp_month</code>, etc..</td>
  </tr>
</table>

### Whitelist literals
By whitelisting an outlier, you prevent them from being tagged and stored in Elasticsearch.
For events that have already been enriched and that match a whitelist later, the 
<code>es_wipe_all_whitelisted_outliers</code> flag can be used in order to remove them.

To have more information about literals whitelist, visit the page [Whitelisting outliers](WHITELIST.md#whitelisting-outliers).

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky">Any existing field name</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">This whitelist will only hit for outlier events that contain an exact whitelisted string as one 
    of its event field values. The whitelist is checked against all the event fields, not only the outlier fields!
    Example: <code>slack_connection=rare outbound connection: Slack.exe</code>.</td>
  </tr>
</table>

### Whitelist regexps
By whitelisting an outlier, you prevent them from being tagged and stored in Elasticsearch.
For events that have already been enriched and that match a whitelist later, the 
<code>es_wipe_all_whitelisted_outliers</code> flag can be used in order to remove them.


To have more information about literals whitelist, visit the page [Whitelisting outliers](WHITELIST.md#whitelisting-outliers).

<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky">Any existing field name</td>
    <td class="tg-0pky">regex format</td>
    <td class="tg-0pky">This whitelist will hit for all outlier events that contain a regular expression match against 
    one of its event field values. The whitelist is checked against all the event fields, not only the outlier fields. 
    Example: <code>autorun_user_specific=^.*rare autorun:.*-.*-.*-.*-.*$</code>.</td>
  </tr>
</table>

## Analyzers parameters

To have more information about the configuration of one analyzer, visit the page [Building detection use cases
](CONFIG_OUTLIERS.md).

### Common analyzers parameters
<table>
  <tr>
    <th colspan="3">All analyzers</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters (<small>*Mandatory</small>)</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_query_filter</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-c3ow">Any valid Elasticsearch query.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_dsl_filter</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">DSL filter on Elasticsearch query.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>timestamp_field</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Can be any document field. 
    It will override the general settings <code>timestamp_field</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>history_window_days</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>history_window_days</code> parameter in general settings.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>history_window_hours</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>history_window_hours</code> parameter in general settings.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>should_notify</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, 
    notify the use case via the notifier if <code>email_notifier</code> is set to <code>1</code>.
    If set to <code>0</code>, do nothing.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>use_derived_fields</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Enable or not the utilisation of derived fields.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>es_index</code></td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Override the <code>es_index_pattern</code> parameter in general settings</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>outlier_type</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Freetext field which will be added to the outlier event as new field named <code>outliers.outlier_type</code>. </td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>outlier_reason</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Freetext field which will be added to the outlier event as new field named <code>outliers.reason</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>outlier_summary</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Freetext field which will be added to the outlier event as new field named <code>outliers.summary</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>run_model</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, model run if <code>run_models</code> 
    parameter in general settings is set to <code>1</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>test_model</code>*</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, model run if <code>test_models</code> 
    parameter in general settings is set to <code>1</code>.</td>
  </tr>
</table>

### Usual model parameters

The following parameters could be used for analyzers `terms`, `metrics` and `word2vec`.
More information available [here](CONFIG_OUTLIERS.md#usual-model-parameters).
<table>
  <tr>
    <th colspan="3">Usual model parameters (Terms, Metrics)</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters <small>(*Mandatory)</small></th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>trigger_on</code>*</td>
    <td class="tg-0pky"><code>low</code>, <code>high</code></td>
    <td class="tg-0pky">If set to <code>low</code>, triggers events with model computed value lower than the decision boundary.
    If set to <code>high</code>, triggers events with model computed value higher than the detection boundary.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>trigger_method</code>*</td>
    <td class="tg-0pky">-<code>percentile</code></td>
    <td class="tg-0pky">Percentile. <code>trigger_sensitivity</code> ranges from <code>0</code>-<code>100</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>pct_of_max_value</code></td>
    <td class="tg-0pky">Percentage of maximum value. <code>trigger_sensitivity</code> ranges from <code>0</code>-<code>100</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>pct_of_median_value</code></td>
    <td class="tg-0pky">Percentage of median value. <code>trigger_sensitivity</code> ranges from <code>0</code>-<code>100</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>pct_of_avg_value</code></td>
    <td class="tg-0pky">Percentage of average value. <code>trigger_sensitivity</code> ranges from <code>0</code>-<code>100</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>mad</code>
    <td class="tg-0pky">Median Average Deviation. 
    <code>trigger_sensitivity</code> defines the total number of deviations and ranges from <code>0</code>-<code>Inf.</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>madpos</code></td>
    <td class="tg-0pky">Same as <code>mad</code> but the trigger value will always be positive. 
    In case mad is negative, it will result <code>0</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>stdev</code></td>
    <td class="tg-0pky">Standard Deviation.
    <code>trigger_sensitivity</code> defines the total number of deviations and ranges from <code>0</code>-<code>Inf.</code>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>float</code></td>
    <td class="tg-0pky"> Fixed value to trigger on. <code>trigger_sensitivity</code> defines the trigger value.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>coeff_of_variation</code></td>
    <td class="tg-0pky">Coefficient of variation.
    <code>trigger_sensitivity</code> defines the total number of coefficient of variation and ranges from <code>0</code>-<code>Inf.</code>.</td>
  </tr>
  
  <tr>
    <td class="tg-0pky"><code>trigger_sensitivity</code>*</td>
    <td class="tg-0pky"><code>0-100</code>, <code>0-Inf.</code></td>
    <td class="tg-0pky">Value of the sensitivity linked to the <code>trigger_method</code></td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>process_documents_chronologically</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">If set to <code>1</code>, process documents chronologically when analysing the model. 
    Set by default to <code>0</code> as it has high impact on Elasticsearch scanning performance.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>target</code>*</td>
    <td class="tg-0pky"><code>String</code></td>
    <td class="tg-0pky">Document field that will be used to do the computation 
    (based on the <code>trigger_method</code> selected).</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>aggregator</code>*</td>
    <td class="tg-0pky"><code>Strings</code> separated by a <code>,</code></td>
    <td class="tg-0pky">One or multiple document fields that will be used to group documents.</td>
  </tr>
</table>


#### Arbitrary parameters
Any other parameters that are not used by the model will be automatically copied to the outlier parameter. 
More information available [here](CONFIG_OUTLIERS.md#arbitrary-parameters).

### Simple query parameters

<table>
  <tr>
    <th colspan="3">Simple query</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky">highlight_match</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Override <code>highlight_match</code> parameter in general simplequery settings.
  </tr>
</table>


### Metrics parameters

<table>
  <tr>
    <th colspan="3">Metrics</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters <small>(*Mandatory)</small></th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>metric</code>*</td>
    <td class="tg-0pky">-<code>numerical_value</code></td>
    <td class="tg-0pky">Use the numerical value of the target field as metric. Example: numerical_value("2") => 2.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>length</code></td>
    <td class="tg-0pky">Use the target field length as metric. Example: length("outliers") => 8.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>entropy</code></td>
    <td class="tg-0pky">Use the entropy of the field as metric. Example: entropy("houston") => 2.5216406363433186.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>hex_encoded_length</code></td>
    <td class="tg-0pky">Calculate total length of hexadecimal encoded substrings in the target and use this as metric.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"><code>base64_encoded_length</code></td>
    <td class="tg-0pky">Calculate total length of base64 encoded substrings in the target and use this as metric. 
    Example: base64_encoded_length("houston we have a cHJvYmxlbQ==") => base64_decoded_string: problem, base64_encoded_length: 7.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>url_length</code></td>
    <td class="tg-0pky">Extract all URLs from the target value and use this as metric. 
    Example: url_length("why don't we go http://www.dance.com") => extracted_urls: http://www.dance.com, extracted_urls_length: 20.</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">-<code>relative_english_entropy</code></td>
    <td class="tg-0pky">Compute Kullback Leibler entropy.</td>
  </tr>
</table>


### Terms parameters

<table>
  <tr>
    <th colspan="3">Terms</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters  <small>(*Mandatory)</small></th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>target_count_method</code>*</td>
    <td class="tg-0pky"><code>within_aggregator</code>, <code>across_aggregators</code></td>
    <td class="tg-0pky">If set to <code>across_aggregator</code> the analysis will be performed across all values of the 
    aggregator at the same time. If set to <code>within_aggregator</code>, will be performed for each value of the aggregator separately.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>min_target_buckets</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Minimum number of events within an aggregation before processing terms analyzer. 
    Only with the <code>target_count_method</code> set on <code>within_aggregator</code>.
    </td>
  </tr>
</table>

### Sudden Appearance parameters

<table>
  <tr>
    <th colspan="3">Sudden Appearance</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters  <small>(*Mandatory)</small></th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>target</code>*</td>
    <td class="tg-0pky"><code>String</code> separated by <code>,</code></td>
    <td class="tg-0pky">One or multiple document fields that will be analyzed for sudden appearance in group documents.
    </td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>aggregator</code>*</td>
    <td class="tg-0pky"><code>String</code> separated by <code>,</code></td>
    <td class="tg-0pky">One or multiple document fields that will be used to group documents.
    Each document that contains the same combination of field values will be assembled in the same group. 
    </td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>history_window_days</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>history_window_days</code> parameter in general settings.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>history_window_hours</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>history_window_hours</code> parameter in general settings.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>sliding_window_size</code>*</td>
    <td class="tg-0pky"><code>DDD</code>:<code>HH</code>:<code>MM</code></td>
    <td class="tg-0pky">Size of the sliding window where <code>DDD</code> define the number of days, 
    <code>HH</code> the number of hours and <code>MM</code> the number of minutes.
    Example: <code>20</code>:<code>13</code>:<code>20</code> will correspond to a sliding window of size 20 days, 13
    hours and 20 minutes.
    </td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>sliding_window_step_size</code>*</td>
    <td class="tg-0pky"><code>DDD</code>:<code>HH</code>:<code>MM</code></td>
    <td class="tg-0pky">Size of the sliding step where <code>DDD</code> define the number of days, 
    <code>HH</code> the number of hours and <code>MM</code> the number of minutes. The sliding step represents the
    jump step in time, the sliding window will slide withing the global window.
    Example: <code>10</code>:<code>01</code>:<code>02</code> will correspond to a sliding step of size 10 days, 1
    hours and 2 minutes.
    </td>
  </tr>
</table>

### Word2vec parameters

<table>
  <tr>
    <th colspan="3">Word2vec</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameters</th>
    <th class="tg-0pky">Values</th>
    <th class="tg-0pky">Notes</th>
  </tr>
  <tr>
    <td class="tg-0pky"><code>word2vec_batch_eval_size</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>word2vec_batch_eval_size</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>min_target_buckets</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>min_target_buckets</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>drop_duplicates</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Override <code>drop_duplicates</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>use_prob_model</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Override <code>use_prob_model</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>output_prob</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Override <code>output_prob</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>separators</code></td>
    <td class="tg-0pky">regex format <b>between quotes</b></td>
    <td class="tg-0pky">Override <code>drop_duplicates</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>size_window</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>size_window</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>min_uniq_word_occurrence</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>min_uniq_word_occurrence</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>num_epoch</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>num_epoch</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>learning_rate</code></td>
    <td class="tg-0pky"><code>Float</code></td>
    <td class="tg-0pky">Override <code>learning_rate</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>embedding_size</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>embedding_size</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>seed</code></td>
    <td class="tg-0pky"><code>Int</code></td>
    <td class="tg-0pky">Override <code>seed</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>print_score_table</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Override <code>print_score_table</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>print_confusion_matrix</code></td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Override <code>print_confusion_matrix</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>trigger_focus</code></td>
    <td class="tg-0pky"><code>word</code>, <code>text</code></td>
    <td class="tg-0pky">Override <code>trigger_focus</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
  <tr>
    <td class="tg-0pky"><code>trigger_score</code></td>
    <td class="tg-0pky"><code>center</code>, <code>context</code>, <code>total</code>, <code>mean</code></td>
    <td class="tg-0pky">Override <code>trigger_score</code> parameter in 
    <a href=#word2vec>word2vec general configuration</a>.</td>
  </tr>
</table>
