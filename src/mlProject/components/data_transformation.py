import os
from src.mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.mlProject.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import MinMaxScaler



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up
        
    def transform_data(self , df):

        try:
            df.drop('id' , axis=1 ,  inplace= True) # drop id column
            df.dropna(inplace=True) # dropping all NA field Columns

            object_columns= df.select_dtypes(include=['object','bool']).columns # selecting only object datatype columns
            df , encoder = self.encode_categorical_columns(df , object_columns) # encoding categorical columns
            df.drop('age' , axis=1 , inplace=True) # dropping age column after removing multicollinearity

            feature_to_scale = ['avg_glucose_level' , 'bmi']
            min_max = MinMaxScaler()
            df[feature_to_scale] = min_max.fit_transform(df[feature_to_scale])
            with open(os.path.join(self.config.root_dir, "MinMaxScalar.pkl"), 'wb') as f:
                pickle.dump(min_max, f)
            


            return df

        except Exception as e:
                raise e


    def train_test_spliting(self):
        raw_data = pd.read_csv(self.config.data_path)

        data = self.transform_data(raw_data)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        

    def encode_categorical_columns(self , df, categorical_columns):
        """
        Encode multiple categorical columns using LabelEncoder and save the encoders.
        
        Parameters:
        df (pandas.DataFrame): The DataFrame containing the categorical columns to be encoded.
        categorical_columns (list): A list of column names containing categorical data.
        
        Returns:
        pandas.DataFrame: DataFrame with categorical columns encoded.
        dict: Dictionary containing encoder objects for each categorical column.
        """
        encoders = {}  # Dictionary to store encoder objects
        
        for col in categorical_columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])  # Add encoded column to DataFrame
            encoders[col] = encoder  # Store encoder object in dictionary
        
        # Save encoders using pickle
        with open(os.path.join(self.config.root_dir, "encoders.pkl"), 'wb') as f:
            pickle.dump(encoders, f)
        
        return df, encoders

