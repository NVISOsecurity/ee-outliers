# Changelog

## Version 0.2.3 (August 21, 2019)
TODO

## Version 0.2.2 (August 20, 2019)
- Bug fix in Jenkings
- Fix bug with watcher that create bug when settings were modified
- DummyDocumentGenerate now work with dictionary (and not a lot of parameters)
- Add test for Notifier
- Fix #48

## Version 0.2.1 (August 16, 2019)
- Add documentation about Notifier
- Integrate UML correctly in the documentation
- Add docstring for a lot of methods
- Add `non_outlier_values` in `Metrics`
- Fix windows that was all the time recompute (changes between "count" and "scan")
- Add and enhance tests

## Version 0.2.0 (August 2, 2019)
- Move Docker image from "debian" to "python 3.6"
- Update ElasticSearch library to 6.4.0 (and also sentry to the version 0.10.2)
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

## [Version 0.1.2](https://github.com/NVISO-BE/ee-outliers/releases/tag/0.1.2) (November 13, 2018)

- "assets" is now a required global configuration section
- "outlier_assets" are no longer used on per-use-case basis
- "outlier_value" is no longer used

Please check the defaults/outliers.conf file for an example of this new format

