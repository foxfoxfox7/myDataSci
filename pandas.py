# TURNING LISTS INTO A DATAFRAME

# list_keys is a list and will be cols, list_values is a list of lists
zipped = list(zip(list_keys, list_values))
# Build a dictionary with the zipped list: data
data = dict(zipped)
# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
