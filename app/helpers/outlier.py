from helpers.singletons import settings
import re
import helpers.utils
import textwrap

from typing import Dict, Any, List, Optional, Union


class Outlier:
    def __init__(self, outlier_type: Union[str, List[str]], outlier_reason: Union[str, List[str]], 
                 outlier_summary: str) -> None:
        self.outlier_dict: Dict[str, Any] = dict()
        self.outlier_dict["type"] = outlier_type  # can be multiple types, for example: malware, powershell
        self.outlier_dict["reason"] = outlier_reason  # can be multiple reasons, for example: DNS tunneling, IDS alert
        # hard-wrap the length of a summary line to 150 characters to make it easier to visualize
        self.outlier_dict["summary"] = textwrap.fill(outlier_summary, width=150)

    # Each whitelist item can contain multiple values to match across fields, separated with ",". So we need to
    # support this too.
    # Example: "dns_tunneling_fp = rule_updates.et.com, intel_server" -> should match both values across the entire
    # event (rule_updates.et.com and intel_server);
    def is_whitelisted(self, additional_dict_values_to_check: Optional[Dict]=None) -> bool:
        whitelist_values_to_check: List[str]
        total_whitelisted_fields_to_match: int
        total_whitelisted_fields_matched: int

        # Check if value is whitelisted as literal
        for (_, each_whitelist_configuration_file_value) in settings.config.items("whitelist_literals"):
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            total_whitelisted_fields_to_match = len(whitelist_values_to_check)
            total_whitelisted_fields_matched = 0

            for whitelist_val_to_check in whitelist_values_to_check:
                if self.matches_specific_whitelist_item_literally(whitelist_val_to_check,
                                                                  additional_dict_values_to_check):
                    total_whitelisted_fields_matched += 1

            if total_whitelisted_fields_to_match == total_whitelisted_fields_matched:
                return True

        # Check if value is whitelisted as regexp
        for (_, each_whitelist_configuration_file_value) in settings.config.items("whitelist_regexps"):
            whitelist_values_to_check = each_whitelist_configuration_file_value.split(",")

            total_whitelisted_fields_to_match = len(whitelist_values_to_check)
            total_whitelisted_fields_matched = 0

            for whitelist_val_to_check in whitelist_values_to_check:
                if self.matches_specific_whitelist_item_regexp(whitelist_val_to_check,
                                                               additional_dict_values_to_check):
                    total_whitelisted_fields_matched += 1

            if total_whitelisted_fields_to_match == total_whitelisted_fields_matched:
                return True

        # If we reach this point, then there is no whitelist match
        return False

    def matches_specific_whitelist_item_literally(self, whitelist_value: str,
                                                  additional_dict_values_to_check: Optional[Dict] = None) -> bool:
        if self.outlier_dict["summary"] == whitelist_value.strip():
            return True

        if additional_dict_values_to_check:
            for dict_val in list(helpers.utils.nested_dict_values(additional_dict_values_to_check)):
                if isinstance(dict_val, list):
                    for dict_val_item in dict_val:
                        if str(dict_val_item).strip() == whitelist_value.strip():
                            return True
                elif isinstance(dict_val, str):
                    if dict_val.strip() == whitelist_value.strip():
                        return True
        return False

    def matches_specific_whitelist_item_regexp(self, whitelist_value: str,
                                               additional_dict_values_to_check: Optional[Dict] = None) -> bool:
        p = re.compile(whitelist_value.strip(), re.IGNORECASE)

        if p.match(str(self.outlier_dict["summary"])):
            return True

        # If there is an additional dict to check, check all of its values against the whitelist
        if additional_dict_values_to_check:
            for dict_val in list(helpers.utils.nested_dict_values(additional_dict_values_to_check)):
                if isinstance(dict_val, list):
                    for dict_val_item in dict_val:
                        if p.match(str(dict_val_item).strip()):
                            return True
                elif isinstance(dict_val, str):
                    if p.match(str(dict_val.strip())):
                        return True
        return False

    def get_outlier_dict_of_arrays(self) -> Dict[str, List[str]]:
        outlier_dict_of_arrays: Dict[str, List[str]] = dict()

        for k, v in self.outlier_dict.items():
            if isinstance(v, list):
                outlier_dict_of_arrays[k] = v
            else:
                outlier_dict_of_arrays[k] = [v]

        return outlier_dict_of_arrays

    def __str__(self) -> str:
        _str: str = "\n"
        _str += "=======\n"
        _str += "outlier\n"
        _str += "=======\n"

        for key, value in self.outlier_dict.items():
            _str += (str(key) + "\t -> " + str(value) + "\n")

        return _str
