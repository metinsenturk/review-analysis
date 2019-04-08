import json
import os
import logging
from ast import literal_eval

logger = logging.getLogger(__name__)

def get_credidentials():  
    """ load credentials from json file. """
    json_data = open("{}/{}".format(os.getcwd(), "parameters.json")).read() 
    data = json.loads(json_data)
    return(data['data'])

def fix_token_columns(df):
    """ fix column type error for pandas dataframes. """
    fix_columns_list = ['sent_tokens', 'word_tokens_doc', 'norm_tokens_doc', 'word_tokens', 'norm_tokens']
    for column in fix_columns_list:
        try:
            df[column] = df[column].apply(lambda x: literal_eval(x))
        except Exception as ex:
            logger.warning(f"fix_token_columns failed for {column}.", exc_info=ex)
    
    return df

def fix_token_columns2(df):
    """ fix column type error for pandas dataframes. """
    try:
        df = df.apply(lambda x: literal_eval(x))
    except Exception as ex:
        logger.warning(f"fix_token_columns failed.", exc_info=ex)
    
    return df
