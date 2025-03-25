import pandas as pd

# loading the original csv
df = pd.read_csv("data/SYB67_313_202411_Threatened Species.csv")

# Using the first row as the actual header
df.columns = df.iloc[0]
df = df[1:].copy()

# Identifying unnamed country column (2nd column)
unnamed_country_col = df.columns[1]

# Renaming columns for clarity
df = df.rename(columns={
    'Region/Country/Area': 'Region_Code',
    unnamed_country_col: 'Country',
    'Year': 'Year',
    'Series': 'Species_Type',
    'Value': 'Count'
})

# Keeping relevant columns
df_cleaned = df[['Country', 'Year', 'Species_Type', 'Count']]

# Dropping rows with missing values in key columns
df_cleaned = df_cleaned.dropna(subset=['Country', 'Year', 'Species_Type', 'Count'])

# Converting 'Year' and 'Count' to numeric
df_cleaned['Year'] = pd.to_numeric(df_cleaned['Year'], errors='coerce')
df_cleaned['Count'] = pd.to_numeric(df_cleaned['Count'], errors='coerce')

# Dropping rows where numeric conversion failed
df_cleaned = df_cleaned.dropna(subset=['Year', 'Count'])

# Resetting index
df_cleaned.reset_index(drop=True, inplace=True)

# Saving the cleaned dataset to a CSV file
df_cleaned.to_csv("data/cleaned_ThreatenedSpecies.csv", index=False)


