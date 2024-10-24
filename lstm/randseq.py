import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.lib import xla_bridge
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
from utils import split_and_discard_sparse,normalize_sequence

class RandomSequenceLoader():
    def __init__(self, data_directory):
        self.directory = data_directory

    def get_all_columns(self,csv_file):
        try:
            with open(csv_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)  # Get the header row
                data = [row for row in reader]

            # Transpose the data to get an array of arrays, where each array is a column
            data = np.array(data, dtype=float).T

            # Create a dictionary to store the columns with their corresponding names
            all_columns = {header[i]: data[i] for i in range(len(header))}
            return all_columns
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def getAllRawData(self,search_term):
        # List files that contain "NoDownsampling" in their name
        matching_files = [filename for filename in os.listdir(self.directory) if search_term in filename]
        raw_data=[] #list of dictionaries from all files
        print(matching_files)
        # Print the matching file names
        for filename in matching_files:
            # Extract all columns as an array of arrays
            cols = self.get_all_columns(os.path.join(self.directory, filename))
            # Print all columns
            raw_data.append(cols)
        return raw_data
