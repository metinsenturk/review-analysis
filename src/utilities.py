import json
import os
from ast import literal_eval

def get_credidentials():  
    """ load credentials from json file. """
    json_data = open("{}/{}".format(os.getcwd(), "parameters.json")).read() 
    data = json.loads(json_data)
    return(data['data'])

def fix_token_columns(df):
    """ fix column type error for pandas dataframes. """
    fix_columns_list = ['sent_tokens', 'word_tokens_doc', 'norm_tokens_doc', 'word_tokens', 'norm_tokens']
    for column in fix_columns_list:
        df[column] = df[column].apply(lambda x: literal_eval(x))
    
    return df
