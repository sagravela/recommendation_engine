import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, 
                 train:pd.DataFrame, 
                 validation:pd.DataFrame=None, 
                 target:str=None, 
                 categorical:list=None):
        """
        Initializes the class instance with the specified parameters.

        Parameters:
            train (pd.DataFrame): The training data.
            validation (pd.DataFrame, optional): The validation data. Default is None.
            target (str, optional): The target variable. Default is None.
            categorical (list, optional): The list of categorical columns. Default is None.

        Returns:
            None
        """
        train = train.copy()
        train.columns = train.columns.str.strip().str.lower().str.replace(" ", "_")
        if validation is not None:
            validation = validation.copy()
            validation.columns = validation.columns.str.strip().str.lower().str.replace(" ", "_")
        if categorical is not None:
            train[categorical] = train[categorical]. \
                                            apply(lambda x: pd.Categorical(x,ordered=True))
            if validation is not None:
                validation[categorical] = validation[categorical]. \
                                                    apply(lambda x: pd.Categorical(x,ordered=True))
        
        self.train = train
        self.val = validation
        self.target = target
    
        print("Shape of train set: ", self.train.shape)
        if self.val is not None:
            print("Shape of validation set: ", self.val.shape)
    
    
    def _data(self, df, val:bool) -> pd.DataFrame:
        """
        Returns the input DataFrame if it is given. 
        Otherwise, returns the `val` DataFrame if `val` is True, or the `train` DataFrame if `val` is False.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            val (bool): A boolean value indicating whether to return the `val` DataFrame or the `train` DataFrame if `df` is None.
        
        Returns:
            pd.DataFrame: The DataFrame to be returned.
        """
        if df is None:
            df = self.val.copy() if val and self.val is not None else self.train.copy()
        return df
    
    
    def explore(self, df:pd.DataFrame=None, val:bool=False) -> pd.DataFrame:
        """
        Displays the summary statistics of the dataset.
        
        Parameters:
            df (pd.DataFrame, optional): The input DataFrame. Default is `train` DataFrame.
            val (bool, optional): A boolean value indicating whether to use the `val` DataFrame. Default is False,
            indicating `train` DataFrame is used.
        """
        pd.set_option('display.float_format', '{:.2f}'.format)
        
        df = self._data(df, val)
        df.replace("", np.nan, inplace=True)
        temp = df.describe(include="all").drop(["count", "unique"]).T.fillna("-")
        temp.insert(loc=0, column='Dtype', value=[df[f].dtype for f in df])
        temp.insert(loc=1, column='nunique', value=df.nunique())
        temp.insert(loc=2, column='unique', value=[df[f].unique() for f in df])
        temp.insert(loc=3, column='nulls', value=df.isnull().sum())
        temp.insert(loc=4, column='nulls%', value=df.isnull().mean() * 100)
        return temp
    
    
    def target_distribution(self, feature:str=None, val:bool=False):
        """
        Displays the distribution of the target variable.
        
        Parameters:
            feature (str, optional): The name of the feature to display the distribution. Default is target variable
            defined in constructor, otherwise is asked to provide the feature.
            val (bool, optional): A boolean value indicating whether to use the `val` DataFrame. Default is False,
            indicating `train` DataFrame is used.
        """
        if feature is None:
            feature = self.target if self.target else None
            if feature is None:
                print("Define target")
                return
        df = self._data(df=None, val=val)
        temp = df[feature].value_counts()
        if len(temp.index) < 20:
            return px.pie(values=temp, names=temp.index, title='Target Distribution', width=500, height=500)
        return px.histogram(data_frame=df, x=feature, title='Target Distribution')
    
    
    def plot_distributions(self, df: pd.DataFrame = None, columns_to_drop: list = [], target: str = None, val: bool = False):
        """
        Displays the distribution of the features in the dataset.

        Parameters:
            df (pd.DataFrame, optional): The input DataFrame. Default is `train` DataFrame.
            columns_to_drop (list, optional): The list of columns to drop. Default is an empty list.
            target (str, optional): The name of the target variable. Default is target variable defined in constructor,
            otherwise None.
            val (bool, optional): A boolean value indicating whether to use the `val` DataFrame. Default is False,
            indicating `train` DataFrame is used.
        """
        df = self._data(df, val)
        if target is None:
            target = self.target

        features = df.columns.drop(columns_to_drop)
        rows = len(features) // 3 if len(features) % 3 == 0 else len(features) // 3 + 1

        fig, axes = plt.subplots(rows, 3, figsize=(15, 20))
        for i, feature in enumerate(features):
            ax = axes.ravel()[i]
            sns.histplot(df,
                         x=feature,
                         hue=target if target and df[target].nunique() < 20 else None,
                         multiple="stack",
                         ax=ax, bins=50,
                         kde=True if df[feature].dtype not in ['O', 'category'] else False)
            
            ax.set_ylabel("")
            ax.set_title(f"{feature} distribution")

            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')

        plt.tight_layout()
        plt.show()
    
    
    def correlation(self, df:pd.DataFrame=None, columns_to_drop:list=[], val:bool=False):
        """
        Displays the correlation matrix of the dataset.
        
        Parameters:
            df (pd.DataFrame, optional): The input DataFrame. Default is `train` DataFrame.
            columns_to_drop (list, optional): The list of columns to drop. Default is an empty list.
            val (bool, optional): A boolean value indicating whether to use the `val` DataFrame. Default is False,
            indicating `train` DataFrame is used.
        """
        df = self._data(df, val)
        df = df.drop(columns_to_drop, axis=1)
        not_number = df.select_dtypes(exclude='number').columns
        df[not_number] = df[not_number].apply(lambda x: pd.factorize(x)[0])
        correlation_matrix = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        correlation_matrix = correlation_matrix.mask(mask)
        return px.imshow(correlation_matrix,
                labels=dict(color="Correlation"),
                x=correlation_matrix.index,
                y=correlation_matrix.columns,
                color_continuous_scale='Viridis')
    
    
    def pairplot(self, df:pd.DataFrame=None, 
                 columns_to_drop:list=[], 
                 target:str=None, val:bool=False,
                 height:int=None, width:int=None):
        """
        Displays the pairplot of the dataset.
        
        Parameters:
            df (pd.DataFrame, optional): The input DataFrame. Default is `train` DataFrame.
            columns_to_drop (list, optional): The list of columns to drop. Default is an empty list.
            target (str, optional): The name of the target variable. Default is target variable
            defined in constructor, otherwise is None.
            val (bool, optional): A boolean value indicating whether to use the `val` DataFrame. Default is False,
            indicating `train` DataFrame is used.
        """
        if target is None:
            target = self.target
        df = self._data(df, val)
        df.drop(columns_to_drop, axis=1, inplace=True)
        fig = px.scatter_matrix(df,
                                color=target if target and df[target].nunique() < 20 else None, 
                                opacity=0.5)
        fig.update_traces(showupperhalf=False, diagonal_visible=False)
        fig.update_layout(title="Pairplot between high cardinality features and correlation with target")
        return fig
