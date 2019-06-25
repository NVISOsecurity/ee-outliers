from helpers.singletons import settings
import re
import helpers.utils
import textwrap

from typing import Dict, Any, List, Optional


class Outlier:
    def __init__(self, type, reason, summary) -> None:
        self.outlier_dict: Dict[str, Any] = dict()
        self.outlier_dict["type"] = type.split(",")  # can be multiple types, for example: malware, powershell
        self.outlier_dict["reason"] = reason
        self.outlier_dict["summary"] = textwrap.fill(summary, width=150)  # hard-wrap the length of a summary line to
        # 300 characters to make it easier to visualize

    def is_whitelisted(self, additional_dict_values_to_check: Optional[Dict]=None) -> bool:
        # Check if value is whitelisted as literal
        for (_, each_whitelist_val) in settings.config.items("whitelist_literals"):
            if self.matches_specific_whitelist_item(each_whitelist_val, "literal", additional_dict_values_to_check):
                return True

        # Check if value is whitelisted as regexp
        for (_, each_whitelist_val) in settings.config.items("whitelist_regexps"):
            if self.matches_specific_whitelist_item(each_whitelist_val, "regexp", additional_dict_values_to_check):
                return True

        # If we reach this point, then there is no whitelist match
        return False

    def matches_specific_whitelist_item(self,whitelist_value: str, match_type: str,
                                        additional_dict_values_to_check: Optional[Dict]=None) -> bool:
        if match_type == "literal":
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

        elif match_type == "regexp":
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
        else:
            raise ValueError("whitelist match type must be either literal or regexp")

        # If nothing of this matches, the item does not match
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
        _str = "\n"
        _str += "=======\n"
        _str += "outlier\n"
        _str += "=======\n"

        for key, value in self.outlier_dict.items():
            _str += (str(key) + "\t -> " + str(value) + "\n")

        return _str
