import pandas as pd
import sqlite3 as sql
import json
import os
from pprint import pprint
from dotenv import load_dotenv
import os


load_dotenv()
PATH_TO_CORD = os.getenv("PATH_TO_CORD")



# #delete tables
# con = sqlite3.connect('../cord.db')
# cur = con.cursor()

# # Create table
# cur.execute("DROP TABLE relevant_factors")

# # Insert a row of data
# cur.execute("DROP TABLE symptoms")

# # Save (commit) the changes
# con.commit()

# # We can also close the connection if we are done with it.
# # Just be sure any changes have been committed or they will be lost.
# con.close()


def prepare_sliced_df_for_metadata(df, start, end) :
    df_slice = df[start:end]
    df_slice = df_slice.dropna()
    
    #get pdf_json_file location
    pdf_json_file = df_slice.iloc[:, 9]
    
    text = []
    for i, location in enumerate(pdf_json_file) :
        
        path = PATH_TO_CORD + '/' + str(location)
        
        if os.path.isfile(path) :

            f = open(path)

            # returns JSON object as a dictionary
            data = json.load(f)

            text_temp = ''

            count_text = 0
            for body in data['body_text'] :
                text_temp += body['text']
                count_text += 1
                if count_text > 3 :
                    break

            text.append(text_temp)

            # Closing file
            f.close()
            
        else : text.append('')
            
    df_slice['body_text'] = text
    return df_slice


def make_table(path_to_file, to_remove, table_name) :
    df = pd.read_csv(path_to_file)

    df.drop(to_remove,  axis=1, inplace=True, errors='ignore')
    
    df_slice = prepare_sliced_df_for_metadata(df, 0, 5000)
    
    conn = sqlite3.connect('../cord.db')
    df_slice.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    return df_slice


def append_to_table(path_to_file, to_remove, table_name) :
    df = pd.read_csv(path_to_file)

    df.drop(to_remove, axis=1, inplace=True, errors='ignore')
    df.drop_duplicates(subset=['Study'], inplace=True)
    conn = sqlite3.connect('../cord.db')
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    
    return df


to_remove = ['sha', 'source_x', 'pmcid', 'mag_id', 'who_covidence_id', 'arxiv_id', 'pmc_json_files', 's2_id']
table_name = 'cord19'
path_to_file = PATH_TO_CORD + '/metadata.csv'
df = make_table(path_to_file, to_remove, table_name)

to_remove = ['Date', 'Study Type', 'Influential', 'Infuential', 'Influential (Y/N)', 'Date Published', 'Factors Described']
path_to_folder = '/Kaggle/target_tables/2_relevant_factors/'
for file_name in os.listdir(PATH_TO_CORD + path_to_folder) :
    path_to_file = PATH_TO_CORD + path_to_folder + file_name
    
    table_name = 'relevant_factors'
    df = append_to_table(path_to_file, to_remove, table_name)

to_remove = ['Range (Days)', 'Days', 'Date', 'Study Type', 'Aymptomatic', 'Manifestation', 'Frequency of Symptoms', 'Sample Size', 'Specific Sampled Viral load correlated to postive test', 'Age', 'Asymptomatic', 'Sample Obtained', 'Sample obtained', 'Asymptomatic Transmission', 'Characteristic Related to Question 2']
path_to_folder = '/Kaggle/target_tables/3_patient_descriptions/'
for file_name in os.listdir(PATH_TO_CORD + path_to_folder) :
    path_to_file = PATH_TO_CORD + path_to_folder + file_name
    
    table_name = 'symptoms'
    df = append_to_table(path_to_file, to_remove, table_name)

