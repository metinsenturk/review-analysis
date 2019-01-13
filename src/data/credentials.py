import os
import sys
import json

__dir_path = os.path.dirname(os.path.realpath(__file__))

def get_credidentials():  
    json_data = open("{}/{}".format(__dir_path, "parameters.json")).read() 
    data = json.loads(json_data)
    return(data['data'])
