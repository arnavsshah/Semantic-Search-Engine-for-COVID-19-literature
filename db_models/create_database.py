import pandas as pd
import sqlite3 as sql
import json
import os
from pprint import pprint

path_to_cord = '../../../../../media/arnav/Arnav/ArnavCode/projects/cord-19'

df = pd.read_csv(path_to_cord + '/metadata.csv')

df.drop(['sha', 'source_x', 'pmcid', 'mag_id', 'who_covidence_id', 'arxiv_id', 'pmc_json_files', 's2_id'], 
        axis=1, 
        inplace=True)

def prepare_sliced_df(start, end) :
    df_slice = df[start:end]
    df_slice = df_slice.dropna()
    
    #get pdf_json_file location
    pdf_json_file = df_slice.iloc[:, 9]
    
    text = []
    for i, location in enumerate(pdf_json_file) :
        
        path = path_to_cord + '/' + str(location)
        
        if os.path.isfile(path) :

            f = open(path)
            data = json.load(f)
            text_temp = ''
            count_text = 0
            
            for body in data['body_text'] :
                text_temp += body['text']
                count_text += 1
                if count_text > 3 :
                    break

            text.append(text_temp)

            f.close()
            
        else : text.append('')
            
    df_slice['body_text'] = text
    return df_slice


df_slice = prepare_sliced_df(0, 5000)

conn = sql.connect('cord.db')
df_slice.to_sql('cord19', conn, if_exists='replace', index=False)
