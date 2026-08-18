import requests
import json
import time
import urllib.parse
import pandas as pd

print("here2")

def get_compound_by_name(compound_name):
    # URL encode the compound name to handle spaces and special characters
    encoded_name = urllib.parse.quote(compound_name)
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    url = f"{base_url}/{encoded_name}/property/XlogP/JSON"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error for '{compound_name}': {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception for '{compound_name}': {e}")
        return None
    
def get_compound_properties(data, compound_name):
    try:
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            properties = data['PropertyTable']['Properties'][0]
            return {
                'NAME': compound_name,
                'CID': properties.get('CID'),
                'XlogP': properties.get('XLogP'), #must be L to match with PubChem API
            }
        else:
            return {
                'NAME': compound_name,
                'CID': None,
                'XlogP': None
            }
    except Exception as e:
        print(f"Error extracting SMILES for {compound_name}: {e}")
        return {
            'NAME': compound_name,
            'CID': None,
            'XlogP': None
        }


compound_names = ["aspirin"]

results = []

print("here")

with open("11_FDAnonsalts_names.txt", 'r') as file:
    for line in file:
        name = line.strip()
        print(f"Searching for: {name}")
        data = get_compound_by_name(name)
        if data:
            results.append(get_compound_properties(data, name))
        else:
            results.append({
                'NAME': name,
                'CID': None,
                'XlogP': None
            }) #this is to add an entry also for drugs with errors
        time.sleep(0.2)  # Be respectful to the API

#for name in compound_names:
    
#to save results to a CSV file
df = pd.DataFrame(results)
df.to_csv('11_FDAnonsalts_XlogP.csv', index=False)

print(results)