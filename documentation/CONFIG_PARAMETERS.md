# All parameters in configurations

**Table of contents**
- [General configuration](#general-configuration)
- [Notifier configuration](#notifier-configuration)
- [Derived fields](CONFIG_OUTLIERS.md#derived-fields)
- [Analyzers parameters](#analyzers-parameters)
  - [Common analyzers parameters](#common-analyzers-parameters)
  - [Usual model parameters](#usual-model-parameters)
  - [Metrics parameters](#metrics-parameters)
  - [Terms parameters](#terms-parameters)


## General configuration
<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Note</th>
  </tr>
  <tr>
    <td class="tg-0pky">log_verbosity</td>
    <td class="tg-0pky"><code>0-5</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">log_level</td>
    <td class="tg-0pky"><code>CRITICAL</code>, <code>ERROR</code>, <code>WARNING</code>, <code>INFO</code>, <code>DEBUG</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">log_file</td>
    <td class="tg-0pky">Path to file</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_wipe_all_existing_outliers</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_wipe_all_whitelisted_outliers</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">print_outliers_to_console</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_url</td>
    <td class="tg-0pky">URL to connect to ES</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_timeout</td>
    <td class="tg-0pky">Integer</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">timestamp_field</td>
    <td class="tg-0pky">Any document field</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">history_window_days</td>
    <td class="tg-0pky">Integer</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">history_window_hours</td>
    <td class="tg-0pky">Integer</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_scan_size</td>
    <td class="tg-0pky">Integer</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_scroll_time</td>
    <td class="tg-0pky">Time (format [integer][letter] where letter represent a duration (Hours, Minutes, Seconds))</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_index_pattern</td>
    <td class="tg-0pky">String</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_save_results</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">run_models</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">test_models</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">train_models</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
</table>


## Notifier configuration
TODO: write about notifier


## Analyzers parameters

To have more information about the configuration of one analyzer, visit the page [Building detection use cases
](CONFIG_OUTLIERS.md).

### Common analyzers parameters
<table>
  <tr>
    <th colspan="3">All analyzers</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Note</th>
  </tr>
  <tr>
    <td class="tg-0pky">es_query_filter</td>
    <td class="tg-0pky">Any valid Elasticsearch query</td>
    <td class="tg-c3ow">Mandatory</td>
  </tr>
  <tr>
    <td class="tg-0pky">es_dsl_filter</td>
    <td class="tg-0pky">Any valid filter</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">timestamp_field</td>
    <td class="tg-0pky">Any document key</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">history_window_days</td>
    <td class="tg-0pky"><code>integer</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">history_window_hours</td>
    <td class="tg-0pky"><code>integer</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">should_notify</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">use_derived_fields</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">es_index</td>
    <td class="tg-0pky">Any string</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">outlier_type</td>
    <td class="tg-0pky">Any string</td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
  <tr>
    <td class="tg-0pky">outlier_reason</td>
    <td class="tg-0pky">Any string</td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
  <tr>
    <td class="tg-0pky">outlier_summary</td>
    <td class="tg-0pky">Any string</td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
  <tr>
    <td class="tg-0pky">run_model</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
  <tr>
    <td class="tg-0pky">test_model</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
</table>

#### Arbitrary parameters
Any other parameters that are not used by the model will be automatically copy to the outlier parameter. More information [here](CONFIG_OUTLIERS.md#arbitrary-parameters).


### Usual model paramters

The following parameters could be used for analyzers `terms` and `metrics`.
<table>
  <tr>
    <th colspan="3">Usual model parameters (Terms, Metrics)</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Note</th>
  </tr>
  <tr>
    <td class="tg-0pky">trigger_on</td>
    <td class="tg-0pky"><code>low</code>, <code>high</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">trigger_method</td>
    <td class="tg-0pky"><code>percentile</code>, <code>pct_of_max_value</code>, <code>pct_of_median_value</code>, <code>pct_of_avg_value</code>, <code>mad</code>, <code>madpos</code>, <code>stdev</code>, <code>float</code>, <code>coeff_of_variation</code></td>
    <td class="tg-0pky"><code>coeff_of_variation</code> is only adapt for <code>Terms</code> with <code>target_count_method</code> set on <code>within_aggregator</code></td>
  </tr>
  <tr>
    <td class="tg-0pky">trigger_sensitivity</td>
    <td class="tg-0pky"><code>0-100</code>, <code>0-Inf.</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">process_documents_chronologically</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Default: <code>0</code></td>
  </tr>
  <tr>
    <td class="tg-0pky">target</td>
    <td class="tg-0pky">Any document field</td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
  <tr>
    <td class="tg-0pky">aggregator</td>
    <td class="tg-0pky">List of any document field(s)</td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
</table>


### Metrics parameters

<table>
  <tr>
    <th colspan="3">Metrics</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Note</th>
  </tr>
  <tr>
    <td class="tg-0pky">metric</td>
    <td class="tg-0pky"><code>length</code>, <code>numerical_value</code>, <code>entropy</code>, <code>base64_encoded_length</code>, <code>hex_encoded_length</code>, <code>url_length</code></td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
</table>


### Terms parameters

<table>
  <tr>
    <th colspan="3">Terms</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Note</th>
  </tr>
  <tr>
    <td class="tg-0pky">target_count_method</td>
    <td class="tg-0pky"><code>within_aggregator</code>, <code>across_aggregators</code></td>
    <td class="tg-0pky">Mandatory</td>
  </tr>
  <tr>
    <td class="tg-0pky">min_target_buckets</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Only with the <code>target_count_method</code> set on <code>within_aggregator</code></td>
  </tr>
</table>

*machine_learning*
tensorflow_log_level