import pandas as pd
import numpy as np

######################################################
#Generics
######################################################

# a measure for the sampling bias: avg. difference between the oracle and model performance over all but the first n start_years (bc there is no/small bias at the start)
def measure_bias(oracle_metric, model_metric, last_n_years = 3):
    return sum([om - mm for (om, mm) in zip(oracle_metric[-last_n_years:], model_metric[-last_n_years:])]) / last_n_years

######################################################
#Calculate WOEs
######################################################

class IV_Calc:
    def __init__(self, df, feature, target):
        self.feature = feature
        self.target = target
        self.df = df

    def count_values(self):
        data = pd.DataFrame()
        data['Count'] = self.df[self.feature].value_counts()               # Count instances of each category, create row for each
        data['Bad'] = self.df.groupby([self.feature])[self.target].sum()   # Count y=1 instances of that category
        data['Good'] = data['Count'] - data['Bad']                          # Count y=0 instances of that category
        data = data.sort_values(by=["Count"], ascending=False)
    
        try:
            assert data["Bad"].sum() != 0                               # Check that there are y=1 instances in sample
            assert data["Good"].sum() != 0                              # Check that there are y=0 instances in sample
            assert np.isin(self.df[self.target].unique(), [0, 1]).all()      # Check that target includes only 0,1 or True,False
        except:
          print("Error: Target must include 2 binary classes.")
          raise                                                       # Stop running if one of the above conditions is not satisfied
          
        data = data[data["Count"] != 0]
        
        return data

    def distribution(self):
        data = self.count_values()
        data['Ratio Bad'] = data['Bad'] / data['Count']
        data['Ratio Good'] = data['Good'] / data['Count']
        data["Distribution Bad"] = data["Bad"] / data["Bad"].sum()    # Of all y=0 instances, what percentage are from each category?
        data["Distribution Good"] = data["Good"] / data["Good"].sum() # Of all y=1 instances, what percentage are from each category?
        data = data.sort_values(by=["Count"], ascending=False)
        return data.iloc[:,-2:]
  
    def woe(self):
        data = self.distribution()
        data['WOE'] = np.log(data["Distribution Good"] / data["Distribution Bad"])
        data.replace({"WOE": {np.inf: 0, -np.inf: 0}})  # If no instances are bad, this will replace values of infinity with 0
        data = data.sort_values(by=["WOE"], ascending=False)
        return data.iloc[:,-1]
  
    def woe_adj(self):
        data = self.count_values()
        data["WOE_adj"] = np.log( 
            ((data["Count"] - data["Bad"] + 0.5) / (data["Count"].sum() - data["Bad"].sum())) / 
            ((data["Bad"] + 0.5) / data["Bad"].sum())
            )
        data.replace({"WOE_adj": {np.inf: 0, -np.inf: 0}})
        data = data.sort_values(by=["Count"], ascending=False)
        return data.iloc[:,-1]
  
    def IV_per_cat(self):
        data = self.distribution()
        data['WOE'] = self.woe()
        data["IV"] = data["WOE"]*(data["Distribution Good"] - data["Distribution Bad"])
        data = data.sort_values(by=["IV"], ascending=False)
        return data.iloc[:,-1]
  
    def full_summary(self):
        data = self.count_values()
        data['Ratio Bad'] = data['Bad'] / data['Count']
        data['Ratio Good'] = data['Good'] / data['Count']
        data["Distribution Bad"] = data["Bad"] / data["Bad"].sum()
        data["Distribution Good"] = data["Good"] / data["Good"].sum()
        data['WOE'] = self.woe()
        data["WOE_adj"] = self.woe_adj()
        data["IV"] = self.IV_per_cat()
        data = data.sort_values(by=["Count"], ascending=False)
        return data

    def final_assessment(self):
        data = self.full_summary()
        iv = data["IV"].sum() # final IV value
        if iv < .02:
            print("The variable " + self.feature + " is not predictive with an IV of: {}".format(iv))
        elif iv < .1:
            print("The variable " + self.feature + " is weakly predictive with an IV of:{}".format(iv))
        elif iv < .3:
            print("The variable " + self.feature + " is moderately predictive with an IV of:{}".format(iv))
        else :
            print("The variable " + self.feature + " is highly predictive with an IV of: {}".format(iv))
        return iv

######################################################
#Feature Scorer
######################################################

class filter_binary_target:
    def __init__(self, df, target):
        self.target = target
        self.data_head = df.head()
        self.df = df.copy()

    def auto_filter_binary_target(self):
        #print('Data must be in a clean pandas DataFrame. Categorical variables must be of data type bool or category. Continuous variables must be int64 or float64.')
        data_no_target = self.df.drop(columns=self.target)
        columns = ['Data Type', 'Metric', 'Score']
        index = data_no_target.columns
        result = pd.DataFrame(index=index, columns=columns)

        for col in data_no_target:
            if data_no_target.dtypes[col] == 'bool' or data_no_target.dtypes[col].name == 'category':
                result.loc[col, 'Data Type'] = "discrete"
                result.loc[col, 'Metric'] = "IV"
                result.loc[col, 'Score'] = self.IV_binary_target(feature=col)

            if data_no_target.dtypes[col] == 'int64' or data_no_target.dtypes[col] == 'float64':
                result.loc[col, 'Data Type'] = "continuous"
                result.loc[col, 'Metric'] = "Fisher"
                result.loc[col, 'Score'] = self.fisher_binary_target(feature=col)

        return result

    def IV_binary_target(self, feature):  # same code as used above
        data = pd.DataFrame()
    
        data['Count'] = self.df[feature].value_counts()
        data['Bad'] = self.df.groupby([feature])[self.target].sum()
        data['Good'] = data['Count'] - data['Bad']
    
        data["Distribution Bad"] = data["Bad"] / data["Bad"].sum()
        data["Distribution Good"] = data["Good"] / data["Good"].sum()
    
        data['WOE'] = np.log(data["Distribution Good"] / data["Distribution Bad"])
        data.replace({"WOE": {np.inf: 0, -np.inf: 0}})

        data["IV"] = data["WOE"] * (data["Distribution Good"] - data["Distribution Bad"])

        iv = data["IV"].sum()

        return iv

    def fisher_binary_target(self, feature):
        mu_0 = self.df.groupby(self.df[self.target])[feature].mean()[0]
        mu_1 = self.df.groupby(self.df[self.target])[feature].mean()[1]
        var_0 = self.df.groupby(self.df[self.target])[feature].var()[0]
        var_1 = self.df.groupby(self.df[self.target])[feature].var()[1]

        num = abs(mu_0 - mu_1)
        den = (var_0 + var_1) ** 0.5
        score = num/den
    
        return score

    def pearson(self, feature):  # since our target is binary, we actually don't need this. However, if you would like to expand this class, you can use this code
        mean_feature = self.df[feature].mean()
        mean_target = self.df[self.target].mean()
        num = ((self.df[feature] - mean_feature)*(self.df[self.target] - mean_target)).sum()
        den = (((self.df[feature] - mean_feature)**2).sum() * ((self.df[self.target] - mean_target)**2).sum()) ** .5
        rho = num/den
        return rho

######################################################
# Print with markdown in Jupyter Notebook
######################################################
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))