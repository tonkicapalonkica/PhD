import requests
import json
import time
import urllib.parse
import pandas as pd

print("testimports") #to see if the above lines work

def get_compound_by_name(compound_name):
    # URL encode the compound name to handle spaces and special characters
    encoded_name = urllib.parse.quote(compound_name)
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    url = f"{base_url}/{encoded_name}/property/CanonicalSMILES/JSON"
    
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
                'SMILES': properties.get('ConnectivitySMILES'),
            }
        else:
            return {
                'NAME': compound_name,
                'CID': None,
                'SMILES': None
            }
    except Exception as e:
        print(f"Error extracting SMILES for {compound_name}: {e}")
        return {
            'NAME': compound_name,
            'CID': None,
            'SMILES': None
        }


compound_names = ["aspirin"]

results = []

fails = []

print("test2") #to see if the above lines work

with open("01_APIs_names.txt", 'r') as file: #this is to open the file
    lines = file.readlines()
    lines = [item for line in lines for item in line.split(";")]  # Replace semicolons with new lines
    lines = [line.strip() for line in lines if line.strip()]
    lines = list(set(lines))  # Remove duplicates
    for line in lines:
        name = line.strip()
        print(f"Searching for: {name}")
        data = get_compound_by_name(name)
        if data:
            results.append(get_compound_properties(data, name))
        else:
            fails.append(name)
            results.append({
                'NAME': name,
                'CID': None,
                'SMILES': None
            })  
        time.sleep(0.2)  # Be respectful to the API

#save results in a CSV file:

df = pd.DataFrame(results)
df.to_csv('01_APIs_smiles.csv', index=False)

# Save failed searches to a separate CSV file
if fails:
    fails_df = pd.DataFrame(fails)
    fails_df.to_csv('01_APIs_smilesFailed.csv', index=False)

print(fails)