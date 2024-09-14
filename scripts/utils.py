import pandas as pd
def missing_data(data):
  
    # Total missing values per column
    missing_data = data.isnull().sum()
    
    # Filter only columns with missing values greater than 0
    missing_data = missing_data[missing_data > 0]
    
    # Calculate the percentage of missing data
    missing_percentage = (missing_data / len(data)) * 100
    
    # Combine the counts and percentages into a DataFrame
    missing_df = pd.DataFrame({
        'Missing Count': missing_data, 
        'Percentage (%)': missing_percentage
    })
    
    # Sort by percentage of missing data
    missing_df = missing_df.sort_values(by='Percentage (%)', ascending=False)
    
    return missing_df


def drop_column(df):

    columns_to_drop = ['NumberOfVehiclesInFleet', 
                'CrossBorder', 
                'CustomValueEstimate', 
                'Converted', 'Rebuilt', 
                'WrittenOff']
    # Ensure the columns to drop are actually in the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Drop the columns in place
    df.drop(columns=columns_to_drop, inplace=True)