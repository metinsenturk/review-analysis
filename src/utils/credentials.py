import json
import os

print(f'module name: {__name__},path: {os.getcwd()}')

def get_credidentials():  
    json_data = open("{}/{}".format(os.getcwd(), "parameters.json")).read() 
    data = json.loads(json_data)
    return(data['data'])
