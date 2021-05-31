# Changelog

## [Version 0.2.17](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.17)(August 10, 2020)
### New features
- Enable secured connection to Elasticsearch: TLS encryption + credentials authentication

## [Version 0.2.16](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.16) (July 29, 2020)
### Bug fixes
- Fix issue provoked by an unnecessary debug log message while running _scan
## [Version 0.2.15](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.15) (July 29, 2020)
### New features
- New Sudden Appearance model: the sudden_appearance model looks for outliers by finding the sudden appearance of a 
certain value of a specific field, or multiple fields
### Minor changes
- Security improvements in Dockerfile
- When ee-outliers is unable to write on an index, it returns an error log message instead of stopping the program
- Set "@timestamp" instead of "timestamp" as default timestamps field
### Bug fixes
- Fix issue where generated Elasticsearch DSL queries were not compatible with Elasticsearch version 7.7 and above

## [Version 0.2.14](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.14) (June 24, 2020)
### New features
- Provide the option to authenticate to Elasticsearch using username/password in the configuration file
- Add the option to highlight the part that matched the use case for simplequery models
- Support for multiple use-cases in one configuration file
- Add new version of word2vec model
### Minor changes
- Documentation updated
- Make `whitelist_regexp` and `whitelist_literals` sections non-required in the configuration file
- `timestamp_field` default parameter set to "@timestamp"
- Changed default behavior around the use of derived fields (through grok fields). The derived fields are now by default
 not added as new fields in case an outlier event is found. To activate it, you have to set the `use_derived_fields` use 
 case parameter to `1`
- Process documents non-chronologically by default in simplerequest use cases
### Bug fixes
- In case an outlier event is found, avoid creating fields outside the outlier dictionary when `use_derived_fields` is 
activated

## [Version 0.2.13](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.13) (April 8, 2020)
- Improved documentation (source code and user documentation)
- Fixes an issue where DSL queries in use case configuration files would not be correctly parsed (issue #455)
- Make parsing of configuration use case files more robust
- Improved logging
- PEP8 improvements
- Minor other bug fixes

## [Version 0.2.12](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.12) (March 31, 2020)
- Fixed a bug where a malformed regular expression in the whitelist configuration 
file caused all events to match (fixes issues #462)
- Update Elasticsearch library to 7.5.1
- Changed logging level on some warnings
- Outliers version printed at startup is now read from VERSION file 
instead of being hardcoded
- Major updates to documentation, including new example models
- Minor other bug fixes

## [Version 0.2.11](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.11) (February 6, 2020)
- Detection use cases are now stored in individual config files, rather than all in the main config file

## [Version 0.2.10 hotfix](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.10) (January 6, 2020)
- Flush elasticsearch bulk actions after every model

## [Version 0.2.9](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.9) (November 18, 2019)
- Changed FileHandler to WatchedFilterHandler to support copytruncate log rotation
- Removed flush clause in finally statement in perform_analysis, which could cause outliers to crash in case Elasticsearch is down

## [Version 0.2.5](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.5) (August 26, 2019)
- Add arbitrary config
- Enhance settings performance: store value when the configuration is parse
- Duplicate key/section in configuration doesn't produce a crash
- Add tests

## [Version 0.2.4](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.2.4) (August 22, 2019)
- Fix Jenkins bug
- Return appropriate result if unit tests fail

## Version 0.2.3 (August 21, 2019)
- Add a new metric: `relative_english_entropy`
- Add docstring in main
- Enhance connection to Elasticsearch (in case they are some problem)

## Version 0.2.2 (August 20, 2019)
- Bug fix in Jenkings
- Fix bug with watcher that create bug when settings were modified
- DummyDocumentGenerate now work with dictionary (and not a lot of parameters)
- Add test for Notifier
- Fix [#48](https://github.com/NVISO-BE/ee-outliers/issues/48)

## Version 0.2.1 (August 16, 2019)
- Add documentation about Notifier ([here](documentation/NOTIFICATIONS.md))
- Integrate UML correctly in the documentation ([here](documentation/UML.md))
- Add docstring for a lot of methods
- Add `non_outlier_values` in `Metrics`
- Fix windows that was all the time recompute (changes between "count" and "scan")
- Add and enhance tests

## Version 0.2.0 (August 2, 2019)
- Move Docker image from "debian" to "python 3.6"
- Update Elasticsearch library to 6.4.0 (and also sentry to the version 0.10.2)
- Update Documentation (and add UML schema)
- Remove Beaconing (use metrics)
- Respect PEP8 (including max line size)
- Refactor analyzer (rename varaible, optimisation, split methods)
- Delete "add_term_to_batch" of Analyzer (go back to terms and rename "_add_document_to_batch")
- Document is now include in Outlier
- Housekeeping check file modification instead of using a clock
- Addapt logging to display correctly error (using "exc_info")
- Update the whitelist system (prepare the check of whitelist for batch)
- Using `if __name__ == '__main__'` in `outliers.py` to avoid problem
- Enhance and add tests (all models are tested)
- Create class to generate easily dummy document (see `DummyDocumentsGenerate`)

## Version 0.1.6 (June 8, 2019)
- Optimisation: first check that the number of document is upper than zero before to run the scan
- Small bug enhance (condition simplification)
- Move `add_term_to_batch` into Analyzer
- Move `print_analysis_intro` into Analyzer
- "timestamp_field", "history_window_days" and "history_window_hours" could be different for each model. If not define, take global value (see [#157](https://github.com/NVISO-BE/ee-outliers/issues/157))
- Add and enhance tests of Utils
- Test static methods of Analyzer and Outlier
- Create a class to test easily methods, without having ES connection (see `test_stub_es` file)
- Adapt result given by "count_document", depending of the ElasticSearch version
- Remove import done in the code
- Create `FileModificationWatcher`, which was done before in Utils.
- Bug fix [#154](https://github.com/NVISO-BE/ee-outliers/issues/154), [#156](https://github.com/NVISO-BE/ee-outliers/issues/156)

## Version 0.1.2 (November 13, 2018)

- "assets" is now a required global configuration section
- "outlier_assets" are no longer used on per-use-case basis
- "outlier_value" is no longer used

Please check the defaults/outliers.conf file for an example of this new format

