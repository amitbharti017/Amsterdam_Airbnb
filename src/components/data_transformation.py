import os
from logger import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from config.configuration import DataTransformationConfig
from pandas import DataFrame
import re
from datetime import datetime
import numpy as np
# from sklearn import set_config
# set_config(transform_output = "pandas")
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path

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
        self.df_path = df_path
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
        self.df = df
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
        missing_columns = [ col for col in columns if col not in self.df.columns]

        if missing_columns:                     
                raise ValueError(f"Missing columns in the input dataset: {', '.join(missing_columns)}")
        
        self.df = self.df[columns]

        #Removing dollar sign from the price category
        self.df = self.df.dropna(subset=['price'])
        def remove_dollar_sign(text):
            text = str(text)
            numeric_text = text.replace("$","").replace(",","")
            return float(numeric_text)
        self.df.price =  self.df["price"].apply(remove_dollar_sign)
        #data seems to have outliers so removing all the rows that have price more than 2000 per night
        self.df = self.df[self.df["price"] <= 2000]

        #Working with "bedrooms_text" column in the dataframe to change it into numeric data and adding additional column to see if bathrooms are private or shared.   
        self.df["bathrooms_text"]=self.df["bathrooms_text"].fillna(self.df["beds"])

        def bathroom_number(text):
            text = str(text)
            if "half" in text:
                return 0.5
            else:
                number = re.findall(r'\d+\.\d+|\d+', text)
            return float(number[0])
        self.df["bathrooms"] = self.df["bathrooms_text"].apply(bathroom_number)
        self.df["private"] = self.df["bathrooms_text"].apply(lambda x:0 if "shared" in str(x) else 1)
        del self.df["bathrooms_text"]

        #filling missing super_host with f 
        self.df.host_is_superhost = self.df.host_is_superhost.fillna("f")
        self.df["host_is_superhost"] = self.df["host_is_superhost"].apply(lambda x : 1 if x == "t" else 0)

        # Changing the "host since" column to number of days from today.
        self.df["host_since"] = pd.to_datetime(self.df["host_since"])
        self.df["host_since"] = ((pd.Timestamp(datetime.today().date())) - self.df["host_since"]).dt.days

        ## Since most of the missing values of the bed are less i.e below 5. From EDA we saw earlier beds are usually half of accommodates so we follow the same pattern while filling the NA values of bed
        self.df["beds"] = self.df["beds"].fillna(round((self.df["accommodates"])/2))
        self.df["beds"] = self.df["beds"].astype("int64")

        #Since the distribution of minimum_nights has long tale so all the values greater then 30 are set to 31
        self.df["minimum_nights"] = self.df["minimum_nights"].apply(lambda x: 31 if x > 30 else x)

        #places having availability more than 365 days are set to 365 i.e. year availability
        self.df["availability_365"] = self.df["availability_365"].apply(lambda x: 365 if x > 365 else x)

        ## Applying log to the numeric columns which have skewed distribution:
        numeric_skewed_columns = ["host_since", "price", "number_of_reviews","availability_365"]
        for col in numeric_skewed_columns:
            self.df[col] = np.log1p(self.df[col])
        
        return self.df
    
    def data_transformation(self):
        '''
        This function is responsible for data transformation
        '''
        numerical_columns = ["host_since",
            "host_is_superhost",
            "host_listings_count",
            "accommodates", 
            "private",
            "bathrooms",
            "beds", 
            "number_of_reviews", 
            "review_scores_rating",
            "availability_365",
            "minimum_nights"]
        categorical_columns = ["neighbourhood_cleansed", "room_type"]
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
        cat_pipeline = Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder())
            ]
        )
        data_transformer = ColumnTransformer(
            [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ]
        )
        return data_transformer

    def preprocessing_wrapper(self):
        print(self.config.data_train_path)

        df_train = self.column_transformation(self.config.data_train_path)
        df_val = self.column_transformation(self.config.data_val_path)
        df_test = self.column_transformation(self.config.data_test_path)

        df_train = self.get_data_preprocessed(df_train)
        df_val = self.get_data_preprocessed(df_val)
        df_test = self.get_data_preprocessed(df_test)

        target_column = "price"

        y_train = df_train[target_column]
        y_val = df_val[target_column]
        y_test = df_test[target_column]

        df_train = df_train.drop(columns=[target_column])
        df_val = df_val.drop(columns=[target_column])
        df_test = df_test.drop(columns=[target_column])

        trans_obj = self.data_transformation()
        X_train = trans_obj.fit_transform(df_train)
        X_val = trans_obj.transform(df_val)
        X_test = trans_obj.transform(df_test)
       
        dump(X_train, os.path.join(self.config.root_dir, "X_train.joblib"))
        dump(X_val, os.path.join(self.config.root_dir, "X_val.joblib"))
        dump(X_test, os.path.join(self.config.root_dir, "X_test.joblib"))
        dump(y_train, os.path.join(self.config.root_dir, "y_train.joblib"))
        dump(y_val, os.path.join(self.config.root_dir, "y_val.joblib"))
        dump(y_test, os.path.join(self.config.root_dir, "y_test.joblib"))
        
        dump(trans_obj, self.config.transformer)
        logger.info("Data Transformation done")
       

        












    
    

