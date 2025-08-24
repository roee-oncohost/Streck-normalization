"""
methods for dealing with the adat dataframes
    
Author: Roee Orland
date: 2025-8-20
"""
import os
import re
import pandas as pd
from adat_handling import read_adat_file


def get_method_indices_df(method: str, indices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves a DataFrame containing indices for a specific method from the indices DataFrame.
    Args:
        method (str): The name of the method to search for.
        indices_df (pd.DataFrame): DataFrame containing method indices.

    Returns:
        pd.DataFrame: DataFrame containing indices of the specified method.
    """
    if method not in indices_df['material_collected'].values:
        raise ValueError(f"Method '{method}' not found in indices DataFrame.")
    method_df = indices_df[indices_df['material_collected'] == method].copy()
    method_df.drop(columns=['PlateId'], inplace=True)
    return method_df.reset_index(drop=True)


def merge_method_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two DataFrames on 'SampleId' and 'PlateId'.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged_df = pd.merge(df1, df2, on=['sample_name'], how='inner')
    return merged_df.reset_index(drop=True)


def change_column_names(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Changes the column names of a DataFrame by adding a prefix to each column name.

    Args:
        df (pd.DataFrame): The DataFrame whose column names are to be changed.
        prefix (str): The prefix to add to each column name.

    Returns:
        pd.DataFrame: DataFrame with updated column names.
    """
    df.columns = [prefix + col if col != 'sample_name' else col for col in df.columns]
    return df


def filter_file_names(file_paths: list) -> list:
    """
    Filters out non-existent file paths from a list.

    Args:
        file_paths (list): List of file paths to filter.

    Returns:
        list: Filtered list of existing file paths.
    """
    return [file_path for file_path in file_paths if os.path.exists(file_path)]


def get_sample_and_protein_data(file_paths:list, name_pattern:str=r'OH2025_0(\d+)\.adat') -> tuple:
    """
    Reads ADAT files and returns a dict of DataFrames with sample data
    and a DataFrame with protein data.

    Args:
        file_paths (list): List of file paths to ADAT files.

    Returns:
        tuple: A tuple containing a dict of DataFrames with sample data
        and a DataFrame with protein data.
    """
    # df_list = []
    df_dict = {}
    protein_df = None
    for file in file_paths:
        sample_df, protein_df = read_adat_file(file)
        df_dict[file] = sample_df
    new_dict = {}
    for old_key, value in df_dict.items():
        match = re.search(name_pattern, old_key)
        if match:
            new_key = match.group(1)
            new_dict[new_key] = value
    return new_dict, protein_df


def add_plate_number_to_indices(merged_indices: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'plate' column to the merged indices DataFrame by extracting
    the plate number from the 'edta_PlateCode' column.

    Args:
        merged_indices (pd.DataFrame): DataFrame containing merged indices.

    Returns:
        pd.DataFrame: DataFrame with an additional 'plate' column.
    """
    merged_indices['plate'] = merged_indices['edta_PlateCode'].str.split('_0').str[1]
    return merged_indices

def get_sample_values(df: pd.DataFrame, sample_position: str) -> pd.DataFrame:
    """
    Retrieves the values of a specific sample from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing sample data.
        sample_id (str): The ID of the sample to retrieve.

    Returns:
        pd.Series: Series containing the values of the specified sample.
    """
    if sample_position not in df['PlatePosition'].values:
        raise ValueError(f"Sample ID '{sample_position}' not found in DataFrame.")
    columns = [col for col in df.columns if str(col)[0].isdigit()]
    values =  df[df['PlatePosition'] == sample_position][columns]
    return values


if __name__ == "__main__":
    print("This module is not intended to be run directly. Please import it in your script.")
