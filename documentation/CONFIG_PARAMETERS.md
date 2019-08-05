<table class="tg">
  <tr>
   <th colspan="3">General</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Mandatory</th>
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

<table>
  <tr>
    <th colspan="3">All analyzers</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Mandatory</th>
  </tr>
  <tr>
    <td class="tg-0pky">es_query_filter</td>
    <td class="tg-0pky">Any valid ElasticSearch query</td>
    <td class="tg-c3ow">Yes</td>
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
    <td class="tg-0pky">Yes</td>
  </tr>
  <tr>
    <td class="tg-0pky">outlier_reason</td>
    <td class="tg-0pky">Any string</td>
    <td class="tg-0pky">Yes</td>
  </tr>
  <tr>
    <td class="tg-0pky">outlier_summary</td>
    <td class="tg-0pky">Any string</td>
    <td class="tg-0pky">Yes</td>
  </tr>
  <tr>
    <td class="tg-0pky">run_model</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Yes</td>
  </tr>
  <tr>
    <td class="tg-0pky">test_model</td>
    <td class="tg-0pky"><code>0</code>, <code>1</code></td>
    <td class="tg-0pky">Yes</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="3">Usual model parameters (Terms, Metrics...)</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Mandatory</th>
  </tr>
  <tr>
    <td class="tg-0pky">trigger_on</td>
    <td class="tg-0pky"><code>low</code>, <code>high</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">trigger_method</td>
    <td class="tg-0pky"><code>percentile</code>, <code>pct_of_max_value</code>, <code>pct_of_median_value</code>, <code>pct_of_avg_value</code>, <code>mad</code>, <code>madpos</code>, <code>stdev</code>, <code>float</code>, <code>coeff_of_variation</code></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">trigger_sensitivity</td>
    <td class="tg-0pky"><code>0-100</code>, <code>0-Inf.</code></td>
    <td class="tg-0pky"></td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="3">Metrics</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Mandatory</th>
  </tr>
  <tr>
    <td colspan="3">TODO</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="3">Terms</th>
  </tr>
  <tr>
    <th class="tg-0pky">Key parameter</th>
    <th class="tg-0pky">Possible values</th>
    <th class="tg-0pky">Mandatory</th>
  </tr>
  <tr>
    <td colspan="3">TODO</td>
  </tr>
</table>


*machine_learning*
tensorflow_log_level