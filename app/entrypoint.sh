#!/usr/bin/env bash

# rsync default configuration file in case it's not configured yet
if [[ ! -f /mappedvolumes/config/outliers.conf ]]; then
    echo "installing default configuration file - outliers.conf"
    cp -v -n /defaults/outliers.conf /mappedvolumes/config/outliers.conf
fi

# Jump into python
exec "$@"
