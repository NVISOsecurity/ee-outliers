<p align="left"><a href="CONFIG_OUTLIERS.md">&#8592; Configuring outliers</a></p>

# Whitelisting outliers

**Table of contents**
- [Literals whitelist](#literals-whitelist)
- [Regular expression whitelist](#regular-expression-whitelist)
- [Whitelist per model](#whitelist-per-model)

ee-outliers provides support for whitelisting of certain outliers. By whitelisting an outlier, you prevent them from being tagged and stored in Elasticsearch.

For events that have already been enriched and that match a whitelist later, the ``es_wipe_all_whitelisted_outliers`` flag can be used in order to remove them.  If a new entry is added to the whitelist, the housekeeping process will search and remove events that are now whitelisted.

Two different whitelists are defined in the configuration file:

## Literals whitelist

This whitelist will only hit for outlier events that contain an exact whitelisted string as one of its event field values.
The whitelist is checked against all the event fields, not only the outlier fields!

Example:
```
[whitelist_literals]
slack_connection=rare outbound connection: Slack.exe
```


## Regular expression whitelist

This whitelist will hit for all outlier events that contain a regular expression match against one of its event field values.
The whitelist is checked against all the event fields, not only the outlier fields!

Example:
```
[whitelist_regexps]
scheduled_task_user_specific_2=^.*rare scheduled task:.*-.*-.*-.*-.*$
autorun_user_specific=^.*rare autorun:.*-.*-.*-.*-.*$
```


## Whitelist per model

It is possible to define one whitelist per model.  This option is possible with literal or regex whitelist.

To define a whitelist for a model, a new section must be defined.  This section must have the following format: `whitelist_literals_<section_name>`
Notice that `literals` can be replace by `regexps`.

For instance, the following configuration define a whitelist for the simplequery model:
```
##############################
# SIMPLEQUERY - DUMMY TEST
##############################
[simplequery_dummy_test]
es_query_filter=es_valid_query

outlier_type=dummy type
outlier_reason=dummy reason
outlier_summary=dummy summary

run_model=1
test_model=1

[whitelist_literals_simplequery_dummy_test]
hostname_whitelist=HOSTNAME-WHITELISTED
```


<p align="right"><a href="NOTIFICATIONS.md">Notification system &#8594;</a></p>