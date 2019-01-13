import json
import pandas as pd

def create_dataset(from_path, to_path):
    """
    Creates a comma seperated value (CSV) file from JSON file. Handles reviews and status during conversion.
    
    from_path: The path to get JSON file for conversion.
    to_path: The path to place the CSV file.
    """
    with open(from_path) as f:
        json_data = json.loads(f.read())
    dataset = pd.io.json.json_normalize(json_data['reviews'])

    dataset_new = pd.DataFrame()
    dataset_new['status'] = dataset.iloc[:, 3].apply(lambda x: 1 if x > 3 else 0)
    dataset_new['reviews'] = dataset.iloc[:,2].values

    dataset_new.to_csv(to_path, index=False)
    dataset_new.head()
