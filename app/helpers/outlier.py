import helpers.singletons
import re
import helpers.utils
import textwrap
import copy

from typing import Optional, Dict


class Outlier:
    def __init__(self, outlier_type, outlier_reason, outlier_summary):
        self.outlier_dict = dict()
        self.outlier_dict["type"] = outlier_type  # can be multiple types, for example: malware, powershell
        self.outlier_dict["reason"] = outlier_reason  # can be multiple reasons, for example: DNS tunneling, IDS alert
        # hard-wrap the length of a summary line to 150 characters to make it easier to visualize
        self.outlier_dict["summary"] = textwrap.fill(outlier_summary, width=150)

    # Each whitelist item can contain multiple values to match across fields, separated with ",". So we need to
    # support this too.
    # Example: "dns_tunneling_fp = rule_updates.et.com, intel_server" -> should match both values across the entire
    # event (rule_updates.et.com and intel_server);
    def is_whitelisted(self, additional_dict_values_to_check=None):
        if additional_dict_values_to_check is not None:
            additional_dict_values = copy.deepcopy(additional_dict_values_to_check)
        else:
            additional_dict_values = dict()
        additional_dict_values["__outlier_dict"] = self.outlier_dict
        return Outlier.is_whitelisted_doc(additional_dict_values)

    def get_outlier_dict_of_arrays(self):
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

    def __eq__(self, other):
        return isinstance(other, Outlier) and self.outlier_dict == other.outlier_dict

    @staticmethod
    def is_whitelisted_doc(dict_values_to_check=None):
        # Check if value is whitelisted as literal
        for (_, each_whitelist_configuration_file_value) in \
                helpers.singletons.settings.config.items("whitelist_literals"):
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            total_whitelisted_fields_to_match = len(whitelist_values_to_check)
            total_whitelisted_fields_matched = 0

            for whitelist_val_to_check in whitelist_values_to_check:
                if Outlier.dictionary_matches_specific_whitelist_item_literally(whitelist_val_to_check,
                                                                                dict_values_to_check):
                    total_whitelisted_fields_matched += 1

            if total_whitelisted_fields_to_match == total_whitelisted_fields_matched:
                return True

        # Check if value is whitelisted as regexp
        for (_, each_whitelist_configuration_file_value) in \
                helpers.singletons.settings.config.items("whitelist_regexps"):
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            total_whitelisted_fields_to_match = len(whitelist_values_to_check)
            total_whitelisted_fields_matched = 0

            for whitelist_val_to_check in whitelist_values_to_check:
                p = re.compile(whitelist_val_to_check.strip(), re.IGNORECASE)
                if Outlier.dictionary_matches_specific_whitelist_item_regexp(p, dict_values_to_check):
                    total_whitelisted_fields_matched += 1

            if total_whitelisted_fields_to_match == total_whitelisted_fields_matched:
                return True

        # If we reach this point, then there is no whitelist match
        return False

    @staticmethod
    def dictionary_matches_specific_whitelist_item_literally(whitelist_value, dictionary):
        for dict_val in list(helpers.utils.nested_dict_values(dictionary)):
            if isinstance(dict_val, list):
                for dict_val_item in dict_val:
                    if str(dict_val_item).strip() == whitelist_value.strip():
                        return True
            elif isinstance(dict_val, str):
                if dict_val.strip() == whitelist_value.strip():
                    return True
        return False

    @staticmethod
    def dictionary_matches_specific_whitelist_item_regexp(regex, dictionary):
        for dict_val in list(helpers.utils.nested_dict_values(dictionary)):
            if isinstance(dict_val, list):
                for dict_val_item in dict_val:
                    if regex.match(str(dict_val_item).strip()):
                        return True
            elif isinstance(dict_val, str):
                if regex.match(str(dict_val.strip())):
                    return True
        return False
