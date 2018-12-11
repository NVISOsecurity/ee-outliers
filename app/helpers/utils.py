import math

import collections
import datetime
import iso8601
import dateparser
import netaddr
import numpy as np
import base64
import re
from statistics import mean, median
from tld import get_tld
import os
import validators


class FileModificationWatcher:
    _previous_mtimes = {}

    def __init__(self, files=[]):
        self.add_files(files)

    def add_files(self, files):
        for f in files:
            self._previous_mtimes[f] = os.path.getmtime(f)

    def files_changed(self):
        changed = []
        for f in self._previous_mtimes:
            if self._previous_mtimes[f] != os.path.getmtime(f):
                self._previous_mtimes[f] = os.path.getmtime(f)
                changed.append(self._previous_mtimes[f])
        return changed


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_contains_dotkey(dict_value, key_name):
    try:
        get_dotkey_value(dict_value, key_name)
        return True
    except KeyError as e:
        return False


def get_safe_dotkey_value(dict_value, key_name):
    try:
        v = get_dotkey_value(dict_value, key_name)
        return v
    except KeyError:
        return None


def get_dotkey_value(dict_value, key_name):
    """Get value by dot key in dictionary"""
    keys = key_name.split(".")
    for k in keys:
        dict_value = dict_value[k]

    return dict_value


class DictQuery(dict):
    def get(self, path, default = None):
        keys = path.split(".")
        val = None

        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(self, key, default)

            if not val:
                break

        return val


def parse_datestring(datestr=None, _format="auto"):
    if _format == "auto":
        # First try ISO parsing. If that fails go auto
        try:
            date_obj = iso8601.parse_date(datestr)
        except Exception:
            date_obj = dateparser.parse(datestr, settings={'RETURN_AS_TIMEZONE_AWARE': True})
    else:
        date_obj = datetime.datetime.strptime(datestr, _format)

    return date_obj


def time_to_string(_datetime):
    # INPUT : a datetime object (e.g. '2017-07-24T12:46:11.000Z')
    # OUTPUT : the string corresponding to the time parameter (e.g. 12:46:11)
    return _datetime.strftime('%H:%M:%S')


def match_ip_ranges(source_ip, ip_cidr):
    return False if len(netaddr.all_matching_cidrs(source_ip, ip_cidr)) <= 0 else True


def is_weekend(_datetime):
    return True if _datetime.weekday() >= 5 else False


def datetime_to_date_string(_datetime):
    return _datetime.strftime('%d-%m-%Y')


def day_to_datetime(_datetime):
    return _datetime.strptime(_datetime, '%d-%m-%Y')


def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(chr(x))) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy


# Don't care about div by zero
def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


def incremental_range(start, stop, step, inc):
    value = start
    while value <= stop:
        yield value
        value += step
        step += inc


def is_in_top_x(array, el, top_n):
    top_n_indices = (array[np.argsort(array)[-top_n:]])

    if el in top_n_indices:
        return True
    else:
        return False


def extract_outlier_asset_information(fields, settings):
    outlier_assets = list()
    for (asset_field_name, asset_field_type) in settings.config.items("assets"):

        if dict_contains_dotkey(fields, asset_field_name):
            outlier_assets.append(replace_placeholder_fields_with_values(asset_field_type + ": {" + asset_field_name + "}", fields))

    return outlier_assets


# Convert a sentence value into a flat string, if possible
# If not, just return None
def flatten_sentence(sentence=None):
    if sentence is None:
        return None

    if type(sentence) is list:
        # Make sure the list does not contain nested lists, but only strings. If it's a nested list, we give up and return None
        if any(isinstance(i, list) or isinstance(i, dict) for i in sentence):
            return None
        else:
            # We convert a list value such as [1,2,3] into a single string, so the model can use it: 1-2-3
            field_value = " - ".join(str(x) for x in sentence)
            return field_value
    elif type(sentence) is dict:
        return None
    else:
        # We just cast to string in all other cases
        field_value = str(sentence)
        return field_value


# Convert a sentence format and a fields dictionary into a list of sentences.
# Example:
# sentence_format: hostname, username
# fields: {hostname: [WIN-DRA, WIN-EVB], draman}
# output: [[WIN-DRA, draman], [WIN-EVB, draman]]
def flatten_fields_into_sentences(fields=None, sentence_format=None):
    sentences = [[]]

    for i, field_name in enumerate(sentence_format):
        new_sentences = []
        if type(get_dotkey_value(fields, field_name)) is list:
            for field_value in get_dotkey_value(fields, field_name):
                for sentence in sentences:
                    sentence_copy = sentence.copy()
                    sentence_copy.append(flatten_sentence(field_value))
                    new_sentences.append(sentence_copy)
        else:
            for sentence in sentences:
                sentence.append(flatten_sentence(get_dotkey_value(fields, field_name)))
                new_sentences.append(sentence)

        sentences = new_sentences.copy()

    # Remove all sentences that contain fields that could not be parsed, and that have been flattened to "None".
    # We can't reasonably work with these, so we just ignore them.
    sentences = [sentence for sentence in sentences if None not in sentence]

    return sentences


# if _objis a list of strings, convert it into a single list of strings. if it's a string, just return it
def flatten_into_list_of_strings(_obj):
    if isinstance(_obj, list):
        _lst = list()
        for value in _obj:
            _lst.append(value)
        return _obj
    else:
        return str(_obj)


def replace_placeholder_fields_with_values(placeholder, fields):
    # Replace fields from fieldmappings in summary
    regex = re.compile(r'\{([^\}]*)\}')
    field_name_list = regex.findall(placeholder)  # ['source_ip','destination_ip'] for example

    for field_name in field_name_list:
        if dict_contains_dotkey(fields, field_name):
            if type(get_dotkey_value(fields, field_name)) is list:
                try:
                    field_value = ", ".join(get_dotkey_value(fields, field_name))
                except TypeError:
                    field_value = "complex field " + field_name
            else:
                field_value = str(get_dotkey_value(fields, field_name))

            placeholder = placeholder.replace('{' + field_name + '}', field_value)
        else:
            placeholder = placeholder.replace('{' + field_name + '}', "{field " + field_name + " not found in event}")

    return placeholder


def is_base64_encoded(_str):
    try:
        decoded_bytes = base64.b64decode(_str)
        if base64.b64encode(decoded_bytes) == _str.encode("ascii"):
            return decoded_bytes.decode("ascii")
    except Exception as e:
        return False


def is_hex_encoded(_str):
    try:
        decoded = int(_str, 16)
        return str(decoded)
    except Exception as e:
        return False


def is_url(_str):
    try:
        if validators.url(_str):
            return True
    except Exception as e:
        return False


def transform_value(transformation, value):
    if transformation == "extract_tld":
        try:
            transformed_value = get_tld(value, fix_protocol=True)
        except:
            transformed_value = None
    else:
        transformed_value = None

    return transformed_value


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def get_decision_frontier(trigger_method, values_array, trigger_sensitivity, trigger_on=None):
    if trigger_method == "percentile":
        return get_percentile_decision_frontier(values_array, trigger_sensitivity)

    elif trigger_method == "pct_of_max_value":
        max_value = max(values_array)
        return np.float64(max_value * (trigger_sensitivity / 100))

    elif trigger_method == "pct_of_median_value":
        median_value = median(values_array)
        return np.float64(median_value * (trigger_sensitivity / 100))

    elif trigger_method == "pct_of_avg_value":
        avg_value = mean(values_array)
        return np.float64(avg_value * (trigger_sensitivity / 100))

    elif trigger_method == "mad" or trigger_method == "madpos":
        decision_frontier = get_mad_decision_frontier(values_array, trigger_sensitivity, trigger_on)

        # special case - if MAD is zero, then we use stdev instead of MAD, since more than half of all values are equal
        if decision_frontier == np.nanmedian(values_array):
            decision_frontier = get_stdev_decision_frontier(values_array, 1, trigger_on)

        # special case - if MADPOS is being used, we never want to return a negative MAD, so cap it at 0
        if trigger_method == "madpos":
            decision_frontier = np.float64(max([decision_frontier, 0]))

        return decision_frontier

    elif trigger_method == "stdev":
        return get_stdev_decision_frontier(values_array, trigger_sensitivity, trigger_on)
    elif trigger_method == "float":
        return np.float64(trigger_sensitivity)
    else:
        raise ValueError("Unexpected trigger method " + trigger_method + ", could not calculate decision frontier")


# Calculate percentile decision frontier
# Example: values array is [0 5 10 20 30 2 5 5]
# trigger_sensitivity is 10 (meaning: 10th percentile)
def get_percentile_decision_frontier(values_array, percentile):
    res = np.percentile(list(set(values_array)), percentile)
    return res


def get_stdev_decision_frontier(values_array, trigger_sensitivity, trigger_on):
    stdev = np.std(values_array)

    if trigger_on == "high":
        decision_frontier = np.nanmean(values_array) + trigger_sensitivity * stdev

    elif trigger_on == "low":
        decision_frontier = np.nanmean(values_array) - trigger_sensitivity * stdev
    else:
        raise ValueError("Unexpected trigger condition " + trigger_on + ", could not calculate decision frontier")

    return decision_frontier


def get_mad_decision_frontier(values_array, trigger_sensitivity, trigger_on):
    mad = np.nanmedian(np.absolute(values_array - np.nanmedian(values_array, 0)), 0)  # median absolute deviation

    if trigger_on == "high":
        decision_frontier = np.nanmedian(values_array) + trigger_sensitivity * mad

    elif trigger_on == "low":
        decision_frontier = np.nanmedian(values_array) - trigger_sensitivity * mad
    else:
        raise ValueError("Unexpected trigger condition " + trigger_on + ", could not calculate decision frontier")

    return decision_frontier


def is_outlier(term_value_count, decision_frontier, trigger_on):
    if trigger_on == "high":
        if term_value_count > decision_frontier:
            return True
        else:
            return False
    elif trigger_on == "low":
        if term_value_count < decision_frontier:
            return decision_frontier
        else:
            return False
    else:
        raise ValueError("Unexpected outlier trigger condition " + trigger_on)


def nested_dict_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v
