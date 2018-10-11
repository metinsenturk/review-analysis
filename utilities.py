import json
import os

def get_credidentials():  
    json_data = open("{}/{}".format(os.getcwd(), "sanalysis/parameters.json")).read() 
    data = json.loads(json_data)
    return(data['data'])

if __name__ == '__main__':
    get_credidentials()