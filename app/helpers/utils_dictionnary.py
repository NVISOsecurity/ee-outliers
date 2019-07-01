from typing import Dict

def dict_contains_dotkey(dict_value: Dict, key_name, case_sensitive: bool=True) -> bool:
    try:
        get_dotkey_value(dict_value, key_name, case_sensitive)
        return True
    except KeyError:
        return False


def get_dotkey_value(dict_value: Dict, key_name, case_sensitive: bool=True) -> Dict:
    """
    Get value by dot key in dictionary
    By default, the dotkey is matched case sensitive; for example, key "OsqueryFilter.process_name" will only match if
    the event contains a nested dictionary with keys "OsqueryFilter" and "process_name".
    By changing the case_sensitive parameter to "False", all elements of the dot key will be matched case insensitive.
    For example, key "OsqueryFilter.process_name" will also match a nested dictionary with keys "osqueryfilter" and
    "prOcEss_nAme".
    """
    keys = key_name.split(".")

    for k in keys:
        if not case_sensitive:
            dict_keys = list(dict_value.keys())
            lowercase_keys = list(map(str.lower, dict_keys))
            lowercase_key_to_match = k.lower()
            if lowercase_key_to_match in lowercase_keys:
                matched_index = lowercase_keys.index(lowercase_key_to_match)
                dict_value = dict_value[dict_keys[matched_index]]
            else:
                raise KeyError
        else:
            dict_value = dict_value[k]

    return dict_value