import math
import collections
import netaddr
import numpy as np
import base64
import re
from statistics import mean, median
import validators
from string import Formatter

import helpers.singletons


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dict_contains_dotkey(dict_value, key_name, case_sensitive=True):
    try:
        get_dotkey_value(dict_value, key_name, case_sensitive)
        return True
    except KeyError:
        return False


def get_dotkey_value(dict_value, key_name, case_sensitive=True):
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


def match_ip_ranges(source_ip, ip_cidr):
    return False if len(netaddr.all_matching_cidrs(source_ip, ip_cidr)) <= 0 else True


def shannon_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(chr(x))) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy


def extract_outlier_asset_information(fields, settings):
    """
    :param fields: the dictionary containing all the event information
    :param settings: the settings object which also includes the configuration file that is used
    :return:
    """
    outlier_assets = list()
    for (asset_field_name, asset_field_type) in settings.config.items("assets"):
        try:
            # Could raise an error if key does't exist
            dict_value = get_dotkey_value(fields, asset_field_name, case_sensitive=False)

            sentences = _flatten_one_field_into_sentences(dict_value)
            asset_field_values_including_empty = [sentence for sentence in sentences if None not in sentence]

            # also remove all empty asset strings
            asset_field_values = [sentence[0] for sentence in asset_field_values_including_empty if "" not in sentence]

            # make sure we don't process empty process information, for example an empty user field
            for asset_field_value in asset_field_values:
                outlier_assets.append(asset_field_type + ": " + asset_field_value)

        except KeyError:
            pass  # If error, do nothing

    return outlier_assets


# Convert a sentence value into a flat string, if possible
# If not, just return None
def flatten_sentence(sentence=None):
    if sentence is None:
        return None

    if type(sentence) is list:
        # Make sure the list does not contain nested lists, but only strings. If it's a nested list, we give up and
        # return None
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
        dict_value = get_dotkey_value(fields, field_name, case_sensitive=False)
        new_sentences = _flatten_one_field_into_sentences(sentences=sentences, dict_value=dict_value)
        sentences = new_sentences.copy()

    # Remove all sentences that contain fields that could not be parsed, and that have been flattened to "None".
    # We can't reasonably work with these, so we just ignore them.
    sentences = [sentence for sentence in sentences if None not in sentence]

    return sentences


def _flatten_one_field_into_sentences(dict_value, sentences=list(list())):
    new_sentences = []
    if type(dict_value) is list:
        for field_value in dict_value:
            for sentence in sentences:
                sentence_copy = sentence.copy()
                sentence_copy.append(flatten_sentence(field_value))
                new_sentences.append(sentence_copy)
    else:
        for sentence in sentences:
            sentence.append(flatten_sentence(dict_value))
            new_sentences.append(sentence)

    return new_sentences


def replace_placeholder_fields_with_values(placeholder, fields):
    # Replace fields from fieldmappings in summary
    regex = re.compile(r'\{([^\}]*)\}')
    field_name_list = regex.findall(placeholder)  # ['source_ip','destination_ip'] for example

    for field_name in field_name_list:
        try:
            dict_value = get_dotkey_value(fields, field_name, case_sensitive=False)

            if type(dict_value) is list:
                try:
                    field_value = ", ".join(dict_value)
                except TypeError:
                    field_value = "complex field " + field_name
            else:
                field_value = str(dict_value)

            placeholder = placeholder.replace('{' + field_name + '}', field_value)

        except KeyError:
            placeholder = placeholder.replace('{' + field_name + '}', "{field " + field_name + " not found in event}")

    return placeholder


def is_base64_encoded(_str):
    try:
        decoded_bytes = base64.b64decode(_str)
        if base64.b64encode(decoded_bytes) == _str.encode("ascii"):
            return decoded_bytes.decode("ascii")
    except Exception:
        return False


def is_hex_encoded(_str):
    try:
        decoded = int(_str, 16)
        return str(decoded)
    except Exception:
        return False


def is_url(_str):
    try:
        if validators.url(_str):
            return True
    except Exception:
        return False


def get_decision_frontier(trigger_method, values_array, trigger_sensitivity, trigger_on=None):
    if trigger_method == "percentile":
        decision_frontier = get_percentile_decision_frontier(values_array, trigger_sensitivity)

    elif trigger_method == "pct_of_max_value":
        max_value = max(values_array)
        decision_frontier = np.float64(max_value * (trigger_sensitivity / 100))

    elif trigger_method == "pct_of_median_value":
        median_value = median(values_array)
        decision_frontier = np.float64(median_value * (trigger_sensitivity / 100))

    elif trigger_method == "pct_of_avg_value":
        avg_value = mean(values_array)
        decision_frontier = np.float64(avg_value * (trigger_sensitivity / 100))

    elif trigger_method == "mad" or trigger_method == "madpos":
        decision_frontier = get_mad_decision_frontier(values_array, trigger_sensitivity, trigger_on)

        # special case - if MAD is zero, then we use stdev instead of MAD, since more than half of all values are equal
        if decision_frontier == np.nanmedian(values_array):
            decision_frontier = get_stdev_decision_frontier(values_array, 1, trigger_on)

        # special case - if MADPOS is being used, we never want to return a negative MAD, so cap it at 0
        if trigger_method == "madpos":
            decision_frontier = np.float64(max([decision_frontier, 0]))

    elif trigger_method == "stdev":
        decision_frontier = get_stdev_decision_frontier(values_array, trigger_sensitivity, trigger_on)
    elif trigger_method == "float":
        decision_frontier = np.float64(trigger_sensitivity)
    elif trigger_method == "coeff_of_variation":
        decision_frontier = np.std(values_array) / np.mean(values_array)
    else:
        raise ValueError("Unexpected trigger method " + trigger_method + ", could not calculate decision frontier")

    if decision_frontier < 0:
        # Could not do "from helpers.singletons import logging" due to circle import
        helpers.singletons.logging.logger.debug("negative decision frontier %.2f, this will not generate any outliers",
                                                decision_frontier)

    return decision_frontier


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
        raise ValueError("Unexpected trigger condition " + str(trigger_on) + ", could not calculate decision frontier")

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
        return term_value_count > decision_frontier
    elif trigger_on == "low":
        return term_value_count < decision_frontier
    else:
        raise ValueError("Unexpected outlier trigger condition " + trigger_on)


def nested_dict_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


def seconds_to_pretty_str(seconds):
    return strfdelta(tdelta=seconds, inputtype="seconds", fmt='{D}d {H}h {M}m {S}s')


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)
