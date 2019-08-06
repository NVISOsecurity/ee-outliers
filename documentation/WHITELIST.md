<p align="left"><a href="CONFIG_OUTLIERS.md">&#8592; Configuring outliers</a></p>

# Whitelisting outliers

**Table of contents**
- [Literals whitelist](#literals-whitelist)
- [Regular expression whitelist](#regular-expression-whitelist)

ee-outliers provides support for whitelisting of certain outliers. By whitelisting an outlier, you prevent them from being tagged and stored in Elasticsearch.

For events that have already been enriched and that match a whitelist later, the ``es_wipe_all_whitelisted_outliers`` flag can be used in order to remove them.
The whitelist will then be checked for hits periodically as part of the housekeeping work, as defined in the parameter ``housekeeping_interval_seconds``.

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

<p align="right"><a href="DEVELOPMENT.md">Information for developers &#8594;</a></p>