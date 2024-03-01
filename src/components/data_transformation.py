import os
from logger import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from config.configuration import DataTransformationConfig
from pandas import DataFrame
import re
from datetime import datetime
import numpy as np

class DataTransformation:
    """
    Process incoming data for machine learning
    
    Args:
        config : configuration for data transformation
        df (dataframe): data frame to be preprocessed
    
    """

    def __init__(self,config: DataTransformationConfig):
        self.config = config

    def column_transformation(self,df_path: Path):
        """
        This function is used to lower case all the columns names and textual content
        and replace the " "(space) with "_" in the textural data
        Args:
            df (DataFrame): the dataset location on which transformation is required
        Returns:
            df (DataFrame): dataset after transformation is done 
        
        """
        data = pd.read_csv(self.df_path)
        data.columns = data.columns.str.replace(" ","_").str.lower()
        cat_features = list((data.dtypes[data.dtypes=="object"]).index)
        for col in cat_features:
            data[col] = data[col].str.replace(" ","_").str.lower()
        return data
    
    def get_data_preprocessed(self,df: DataFrame):
        '''
        This function is responsible for data transformation
        Args:
            df (DataFrame) : data frame on which preprocessing is required
        Returns:
            df (DataFrame) : data frame after preprocessing
        '''
        df = self.df
        #taking relevant columns from the dataset
        columns = [
            "host_since",
            "host_is_superhost",
            "host_listings_count",
            "neighbourhood_cleansed", 
            "room_type", 
            "accommodates", 
            "bathrooms_text", 
            "beds", 
            "price", 
            "number_of_reviews", 
            "review_scores_rating",
            "availability_365",
            "minimum_nights"
        ]
        missing_columns = [ col for col in df.columns if col not in columns]

        if missing_columns:                     
                raise ValueError(f"Missing columns in the input dataset: {', '.join(missing_columns)}")
        
        df = df[columns]

        #Removing dollar sign from the price category
        df.dropna(subset=['price'], inplace=True)
        def remove_dollar_sign(text):
            text = str(text)
            numeric_text = text.replace("$","").replace(",","")
            return float(numeric_text)
        df.price =  df["price"].apply(remove_dollar_sign)
        #data seems to have outliers so removing all the rows that have price more than 2000 per night
        df = df[df["price"] <= 2000]

        #Working with "bedrooms_text" column in the dataframe to change it into numeric data and adding additional column to see if bathrooms are private or shared.   
        df["bathrooms_text"].fillna(df["beds"], inplace=True)

        def bathroom_number(text):
            text = str(text)
            if "half" in text:
                return 0.5
            else:
                number = re.findall(r'\d+\.\d+|\d+', text)
            return float(number[0])
        df["bathrooms"] = df["bathrooms_text"].apply(bathroom_number)
        df["private"] = df["bathrooms_text"].apply(lambda x:0 if "shared" in str(x) else 1)
        del df["bathrooms_text"]

        #filling missing super_host with f 
        df.host_is_superhost.fillna("f",inplace = True)
        df["host_is_superhost"] = df["host_is_superhost"].apply(lambda x : 1 if x == "t" else 0)

        # Changing the "host since" column to number of days from today.
        df["host_since"] = pd.to_datetime(df["host_since"])
        df["host_since"] = ((pd.Timestamp(datetime.today().date())) - df["host_since"]).dt.days

        ## Since most of the missing values of the bed are less i.e below 5. From EDA we saw earlier beds are usually half of accommodates so we follow the same pattern while filling the NA values of bed
        df["beds"].fillna(round((df["accommodates"])/2),inplace=True)
        df["beds"] = df["beds"].astype("int64")

        #Since the distribution of minimum_nights has long tale so all the values greater then 30 are set to 31
        df["minimum_nights"] = df["minimum_nights"].apply(lambda x: 31 if x > 30 else x)

        #places having availability more than 365 days are set to 365 i.e. year availability
        df["availability_365"] = df["availability_365"].apply(lambda x: 365 if x > 365 else x)

        ## Applying log to the numeric columns which have skewed distribution:
        numeric_skewed_columns = ["host_since", "price", "number_of_reviews","availability_365"]
        for col in numeric_skewed_columns:
            df[col] = np.log(df[col])
        
        return df
    




    
    

