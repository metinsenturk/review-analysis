import os
import logging
from configparser import SafeConfigParser, ConfigParser

logger = logging.getLogger(__name__)

def create_config():
    """ creates a config file based on app requirements if needed. """

    path = os.path.join(os.curdir, 'config.ini')
    config = ConfigParser()
    yelp_api = 'YelpApi'
    config.add_section(yelp_api)
    config.set(yelp_api, 'api_key1', '')
    config.set(yelp_api, 'api_key2', '')
    config.set(yelp_api, 'api_key3', '')

    yelp_api = 'TwitterApi'
    config.add_section(yelp_api)
    config.set(yelp_api, 'consumer_key', '')
    config.set(yelp_api, 'consumer_secret', '')
    config.set(yelp_api, 'access_token_key', '')
    config.set(yelp_api, 'access_token_secret', '')
    with open(path, 'w') as f:
        config.write(f)


def get_config(path):
    """ get config file. """

    if not os.path.exists(path):
        create_config()

    config = ConfigParser()
    config.read(path)

    return config

def add_setting(path, section, setting, value):
    """ gets a value from config.ini file based on section and setting. """

    config = get_config(path)
    config.set(section, setting, value)

    with open(path, 'w') as f:
        config.write(f)
    
    return True

def get_setting(path, section, setting):
    """ gets a value from config.ini file based on section and setting. """

    config = get_config(path)
    value = config.get(section, setting)

    return value


def update_setting(path, section, setting, value=None):
    """ updates a setting with the new value parameter. """

    config = get_config(path)
    config.set(section, setting, value)

    try:
        with open(path, 'w') as f:
            config.write(f)
        return True
    except Exception as ex:
        logger.warning('could not updated config file.', exc_info=ex)
        return False
