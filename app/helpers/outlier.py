import helpers.singletons
import re
import helpers.utils
import textwrap

from typing import Dict, Set, Any, List, Tuple, Pattern, Optional, Union, cast


class Outlier:
    def __init__(self, outlier_type: Union[str, List[str]], outlier_reason: Union[str, List[str]],
                 outlier_summary: str, doc: Dict) -> None:
        self.outlier_dict: Dict[str, Any] = dict()
        self.outlier_dict["type"] = outlier_type  # can be multiple types, for example: malware, powershell
        self.outlier_dict["reason"] = outlier_reason  # can be multiple reasons, for example: DNS tunneling, IDS alert
        # hard-wrap the length of a summary line to 150 characters to make it easier to visualize
        self.outlier_dict["summary"] = textwrap.fill(outlier_summary, width=150)
        self.doc: Dict = doc
        self._is_whitelisted: Optional[bool] = None

    # Each whitelist item can contain multiple values to match across fields, separated with ",". So we need to
    # support this too.
    # Example: "dns_tunneling_fp = rule_updates.et.com, intel_server" -> should match both values across the entire
    # event (rule_updates.et.com and intel_server);
    def is_whitelisted(self) -> bool:
        if self._is_whitelisted is None:
            # Create dictionary that contain all stuff
            self._is_whitelisted = Outlier.is_whitelisted_doc({'outlier_dict': self.outlier_dict,
                                                               'additional_dict_to_check': self.doc})
        return self._is_whitelisted

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

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Outlier) and self.outlier_dict == other.outlier_dict

    @staticmethod
    def is_whitelisted_doc(dict_to_check: Dict) -> bool:
        dict_values_to_check: Set[str] = set()

        for dict_val in helpers.utils.nested_dict_values(dict_to_check):
            if isinstance(dict_val, list):
                for dict_val_item in dict_val:
                    # force to be a string in case the nested element is a dictionary
                    dict_values_to_check.add(str(dict_val_item).strip())
            else:
                dict_values_to_check.add(str(dict_val).strip())

        # Check if value is whitelisted as literal
        for (_, each_whitelist_configuration_file_value) in \
                cast(List[Tuple[str, str]], helpers.singletons.settings.whitelist_literals_config):

            whitelist_values_to_check: Set[str] = set(map(str.strip,
                                                          each_whitelist_configuration_file_value.split(',' '')))

            # If all whitelisted value are in the dict to check
            if whitelist_values_to_check.issubset(dict_values_to_check):
                return True

        # Check if value is whitelisted as regexps
        for (_, each_whitelist_configuration_file_value) in \
                cast(List[Tuple[str, str]], helpers.singletons.settings.whitelist_regexps_config):
            whitelist_list_values_to_check: List[str] = each_whitelist_configuration_file_value.split(",")

            total_whitelisted_fields_to_match = len(whitelist_list_values_to_check)
            total_whitelisted_fields_matched = 0

            for whitelist_val_to_check in whitelist_list_values_to_check:  # For each regex value

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
    def dictionary_matches_specific_whitelist_item_regexp(regex: Pattern, set_of_values_to_check: Set[str]) -> bool:
        # "any" operator will not create all the list to made the computation. The current code allow to "stop" the
        # "loop" when one element match
        return any(regex.match(value_to_check) for value_to_check in set_of_values_to_check)
