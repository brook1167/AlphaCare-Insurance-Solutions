import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def calculate_variability_measures(num_stats):

    # Calculate variability measures
    range_ = num_stats.max() - num_stats.min()
    variance = num_stats.var()
    std_dev = num_stats.std()
    cv = std_dev / num_stats.mean()
    iqr = num_stats.quantile(0.75) - num_stats.quantile(0.25)
    
    # Create a DataFrame to display these measures
    variability_df = pd.DataFrame({
        'Range': [range_],
        'Variance': [variance],
        'Standard Deviation': [std_dev],
        'Coefficient of Variation': [cv],
        'Interquartile Range (IQR)': [iqr]
    })

    # Display the first few rows of the variability DataFrame
    return variability_df


def handle_missing_data(df):
    
    # Specific columns with 15% to 4% of missing
    moderate_cols = ['NewVehicle', 'Bank', 'AccountType']

    # Specific columns with less than 1% type
    low_missing_cols = [
        'Gender', 'MaritalStatus', 'Cylinders', 'cubiccapacity', 
        'kilowatts', 'NumberOfDoors', 'VehicleIntroDate', 'Model', 
        'make', 'VehicleType', 'mmcode', 'bodytype', 'CapitalOutstanding'
    ]
    
    # Handle columns specific to moderate type missing data
    for col in moderate_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                # Impute categorical columns with mode (check if mode exists)
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna('Unknown')  # Default for empty mode
            else:
                # Impute numerical columns with median (check if median exists)
                if not df[col].isnull().all():  # Ensure column has some numeric values
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)  # Default for empty median

    # Handle columns specific to low missing data
    for col in low_missing_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna('Unknown')  # Default for empty mode
            else:
                if not df[col].isnull().all():
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)  # Default for empty median

    return df


# Ploating functions

def plot_univariate_analysis(data: pd.DataFrame, num_cols=None, cat_cols=None):

    # Histograms for Numerical Columns
    for col in num_cols:
        plt.figure(figsize=(8, 3))  # Increased figure size slightly
        sns.histplot(data[col].dropna(), kde=True, bins=30, color='darkcyan', edgecolor='black', alpha=0.8)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    # Bar Charts for Categorical Columns
    for col in cat_cols:
        plt.figure(figsize=(8, 4))  # Increased figure size slightly
        sns.countplot(x=col, data=data, order=data[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

def scatter_plot(data, x_col, y_col,hue_col):
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=data, x=x_col, y=y_col,hue=hue_col)
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.show()

def correlation_matrix(data, cols):

    corr_matrix = data[cols].corr()
    plt.figure(figsize=(8, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()