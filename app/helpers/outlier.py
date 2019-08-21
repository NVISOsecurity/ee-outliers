import helpers.singletons
import re
import helpers.utils
import textwrap


class Outlier:
    def __init__(self, outlier_type, outlier_reason, outlier_summary, doc):
        self.outlier_dict = dict()
        self.outlier_dict["type"] = outlier_type  # can be multiple types, for example: malware, powershell
        self.outlier_dict["reason"] = outlier_reason  # can be multiple reasons, for example: DNS tunneling, IDS alert
        # hard-wrap the length of a summary line to 150 characters to make it easier to visualize
        self.outlier_dict["summary"] = textwrap.fill(outlier_summary, width=150)
        self.doc = doc
        self._is_whitelisted = None

    # Each whitelist item can contain multiple values to match across fields, separated with ",". So we need to
    # support this too.
    # Example: "dns_tunneling_fp = rule_updates.et.com, intel_server" -> should match both values across the entire
    # event (rule_updates.et.com and intel_server);
    def is_whitelisted(self):
        """
        Check if the current outlier is whitelisted or not. To do this computation, the document content is used.
        Notice that the computation is done one and then just given when this method is again call.

        :return: True if whitelisted, False otherwise
        """
        if self._is_whitelisted is None:
            # Create dictionary that contain all stuff
            self._is_whitelisted = Outlier.is_whitelisted_doc({'outlier_dict': self.outlier_dict,
                                                               'additional_dict_to_check': self.doc})
        return self._is_whitelisted

    def get_outlier_dict_of_arrays(self):
        """
        Transform outliers information into a dictionary where all value are list

        :return: a dictionary of list
        """
        outlier_dict_of_arrays = dict()

        for k, v in self.outlier_dict.items():
            if isinstance(v, list):
                outlier_dict_of_arrays[k] = v
            else:
                outlier_dict_of_arrays[k] = [v]

        return outlier_dict_of_arrays

    def __str__(self):
        _str = "\n"
        _str += "=======\n"
        _str += "outlier\n"
        _str += "=======\n"

        for key, value in self.outlier_dict.items():
            _str += (str(key) + "\t -> " + str(value) + "\n")

        return _str

    @staticmethod
    def is_whitelisted_doc(dict_to_check=None):
        """
        Allow to know if values of a dictionary is whitelisted or not

        :param dict_to_check: dictionary that must be check
        :return: True if whitelisted, False otherwise
        """
        dict_values_to_check = set()

        for dict_val in helpers.utils.nested_dict_values(dict_to_check):
            if isinstance(dict_val, list):
                for dict_val_item in dict_val:
                    # force to be a string in case the nested element is a dictionary
                    dict_values_to_check.add(str(dict_val_item).strip())
            else:
                dict_values_to_check.add(str(dict_val).strip())

        # Check if value is whitelisted as literal
        for (_, each_whitelist_configuration_file_value) in \
                helpers.singletons.settings.whitelist_literals_config:
            whitelist_values_to_check = set(map(str.strip, each_whitelist_configuration_file_value.split(',')))

            # If all whitelisted value are in the dict to check
            if whitelist_values_to_check.issubset(dict_values_to_check):
                return True

        # Check if value is whitelisted as regexps
        for (_, each_whitelist_configuration_file_value) in \
                helpers.singletons.settings.whitelist_regexps_config:
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            total_whitelisted_fields_to_match = len(whitelist_values_to_check)
            total_whitelisted_fields_matched = 0

            for whitelist_val_to_check in whitelist_values_to_check:  # For each regex value

                try:
                    p = re.compile(whitelist_val_to_check.strip(), re.IGNORECASE)
                except Exception:
                    # something went wrong compiling the regular expression, probably because of a user error such as
                    # unbalanced escape characters. We should just ignore the regular expression and continue (and let
                    # the user know in the beginning that some could not be compiled).  Even if we check for errors
                    # in the beginning of running outliers, we might still run into issues when the configuration
                    # changes during running of ee-outlies. So this should catch any remaining errors in the
                    # whitelist that could occur with regexps.
                    break

                # If regex match with the outlier
                if Outlier.dictionary_matches_specific_whitelist_item_regexp(p, dict_values_to_check):
                    # Increment the number of match
                    total_whitelisted_fields_matched += 1
                else:
                    # Break the loop
                    break

            if total_whitelisted_fields_to_match == total_whitelisted_fields_matched:
                return True

        # If we reach this point, then there is no whitelist match
        return False

    @staticmethod
    def dictionary_matches_specific_whitelist_item_regexp(regex, set_of_values_to_check):
        """
        Check if a dictionary match with a regex

        :param regex: regex use to made the test
        :param set_of_values_to_check: set of value that must be checked
        :return: True if one value of the set match the regex
        """
        # "any" operator will not create all the list to made the computation. The current code allow to "stop" the
        # "loop" when one element match
        return any(regex.match(value_to_check) for value_to_check in set_of_values_to_check)
