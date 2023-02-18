#Import Libraries
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import io
from dtreeviz.trees import *
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import pickle
import graphviz
from sklearn.impute import SimpleImputer 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from collections import Counter
import lightgbm as lgb
from IPython.display import HTML
import lime.lime_tabular


# function for random forest importance inside a pipeline
# unsing n_estimor = 100
class RF_Feat_Selector(BaseEstimator, TransformerMixin):

    # class constructor
    # make sure class attributes end with a "_"
    # per scikit-learn convention to avoid errors
    def __init__(self, n_features_=15):
        self.n_features_ = n_features_
        self.fs_indices_ = None

    # override the fit function
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from numpy import argsort
        model_rfi = RandomForestClassifier(n_estimators=100)
        model_rfi.fit(X, y)
        self.fs_indices_ = argsort(model_rfi.feature_importances_)[::-1][0:self.n_features_]
        return self

    # override the transform function
    def transform(self, X, y=None):
        return X[:, self.fs_indices_]


# custom function to format the search results as a Pandas data frame
def get_search_results(gs):

    def model_result(scores, params):
        scores = {'mean_score': np.mean(scores),
             'std_score': np.std(scores),
             'min_score': np.min(scores),
             'max_score': np.max(scores)}
        return pd.Series({**params,**scores})

    models = []
    scores = []

    for i in range(gs.n_splits_):
        key = f"split{i}_test_score"
        r = gs.cv_results_[key]
        scores.append(r.reshape(-1,1))

    all_scores = np.hstack(scores)
    for p, s in zip(gs.cv_results_['params'], all_scores):
        models.append((model_result(s, p)))

    pipe_results = pd.concat(models, axis=1).T.sort_values(['mean_score'], ascending=False)

    columns_first = ['mean_score', 'std_score', 'max_score', 'min_score']
    columns = columns_first + [c for c in pipe_results.columns if c not in columns_first]

    return pipe_results[columns]

#Input: Dataframe containing user datapoint

class ExplainableDecisonTree:

    def __init__(self,model=None,train=None,column_names=[],X_train=None,Y_train=None):
        self.Model = model
        self.Traindata = train
        self.column_names = column_names
        self.X_train=X_train
        self.y_train=Y_train
        





    def Gettraindata(self):
        # Username of your GitHub account
        username = 'Mattjben'
        # Personal Access Token (PAO) from your GitHub account
        token = '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'
        # Creates a re-usable session object with your creds in-built
        github_session = requests.Session()
        github_session.auth = (username, token)
        # Downloading the csv file from your GitHub
        url = "https://raw.githubusercontent.com/IwVr/CSIDS-Finance/main/Datasets/heloc_dataset_v1.csv" # Make sure the url is the raw version of the file on GitHub
        download = github_session.get(url).content
        # Reading the downloaded content and making it a pandas dataframe
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))
        
    
        self.Traindata = df

    def getfeaturenames(self):
        df = self.Traindata
        
        df['RiskPerformance'] = pd.get_dummies(df['RiskPerformance'], drop_first=True, dtype=np.int64)
        X = df.drop('RiskPerformance',axis=1)
        fs_indices_rfi = [ 0 ,17,  3,  1, 22, 13 , 4, 11 , 7, 14,  2 ,18 , 8 ,19 ,21, 20, 12, 15,  9 ,16, 10,  5,  6]
        a=fs_indices_rfi[0:10]
        self.column_names = X.columns[a].to_list()
        print(self.column_names)
        X_train = df.drop("RiskPerformance", axis = 1).values
        y_train = df["RiskPerformance"].values
        X_train_final = X_train[:, np.r_[fs_indices_rfi[0:10]]]
        self.X_train = X_train_final
        self.y_train=y_train 

        return self.column_names
    def model(self):
        df = self.Traindata
        
        df['RiskPerformance'] = pd.get_dummies(df['RiskPerformance'], drop_first=True, dtype=np.int64)
        X = df.drop("RiskPerformance", axis = 1)
        # Dropping the target variable
        X = X.values    # Changing into numpy array
        y = df["RiskPerformance"]   # storing target variable "RiskPerformance"
        y = y.values
        num_features = 23 # 24 minus 1 for the target Variable
        model_rf = RandomForestClassifier(n_estimators=100)
        model_rf.fit(X, y)
        fs_indices_rfi = np.argsort(model_rf.feature_importances_)[::-1][0:num_features]
        print(fs_indices_rfi[0:num_features])
        X_train = df.drop("RiskPerformance", axis = 1).values
        y_train = df["RiskPerformance"].values
        X_train_final = X_train[:, np.r_[fs_indices_rfi[0:10]]]
        self.X_train = X_train_final
        self.y_train=y_train 
        clf = DecisionTreeClassifier(random_state=999,max_depth=5,min_samples_split=2)
        print("------Fitting model--------")
        model = clf.fit(X_train_final, y_train)
        print("------Fitted--------")
        X = df.drop('RiskPerformance',axis=1)
        a=fs_indices_rfi[0:10]
        self.column_names = X.columns[a].to_list()
        self.Model =model
        print(self.column_names)
        pickle.dump(model, open('model.pkl','wb'))



    def frontendpred(self,datapoint,model,user):
        
        text = explain_prediction_path(model, datapoint,
                                    feature_names=self.column_names,
                                    explanation_type="plain_english")
        text=text.strip()
        text=text.split("\n")
        
      
        
        fig=dtreeviz(model, 
               x_data=self.X_train,
               y_data=self.y_train,
               target_name='RiskPerformance',
               feature_names=self.column_names, 
               class_names=['Good', 'Bad'], 
               title="Decision Tree",X=datapoint,show_just_path=True,orientation='LR')
        
        fig.save('Frontend/static/dtreeviz_'+str(user)+'.svg')
        
        return text
    def explainoutput(arrayin):
        
        explanationdict= {
        'ExternalRiskEstimate':	'External Risk Estimate',
    'MSinceOldestTradeOpen':'months since oldest approved credit agreement', 
    'MSinceMostRecentTradeOpen':'months since last approved credit agreement',
    'MSinceMostRecentDelq':'Months since most recent overdue payment',
    'AverageMInFile':'average Months in File',
    'MaxDelq2PublicRecLast12M':'Maximum number of credit agreements with overdue payments or derogatory comments in the last 12 months.',
    'NumSatisfactoryTrades':'number of credit agreements on the customers credit bureau report with on-time payments',
    'PercentTradesNeverDelq':'Percentage of credit agreements on the customers credit bureau report with on-time payments',
    'NumTotalTrades':	'total number of credit agreements the customer' ,
    'PercentInstallTrades':	'percent of installment trades the customer',
    'MSinceMostRecentInqexcl7days':	'months since most recent credit inquiry into the customers credit history (excluding the last 7 days)' ,
    'NetFractionRevolvingBurden':	 'customers revolving burden (portion of credit card spending that goes unpaid at the end of a billing cycle/credit limit)' ,
    'NetFractionInstallBurden':	'customers installment burden (portion of loan that goes unpaid at the end of a billing cycle/monthly instalment to be paid)' ,
    'PercentTradesWBalance'	:'number of trades currently not fully paid off by the customer'}

        line=[]
        for i,value in enumerate(arrayin):
            range = re.search(r'(\d+\.?\d*) (?:<=|<|>=|>) (\w+)\s? (?:<=|<|>=|>) (\d+\.?\d*)',value)
            greater = re.search(r'^(\d+\.?\d*) (?:<=|<) (\w+)\s?$',value)
            lesser = re.search(r'^(\w+)\s? (?:<=|<) (\d+\.?\d*)$',value)
            if range:
                line.append('\n'+'The '+str(explanationdict[range.group(2)])+' was between '+str(range.group(1))+' and '+str(range.group(3)))
            if greater:
                line.append('\n'+'The '+str(explanationdict[greater.group(2)])+' was greater than or equal to '+str(greater.group(1)))
            if lesser:
                line.append('\n'+'The '+str(explanationdict[lesser.group(1)])+' was less than or equal to '+str(lesser.group(2)))
        return line

    def main(self):
    
        self.Gettraindata()
        self.model()
        model = pickle.load(open('model.pkl','rb'))
        print(np.argsort(model.feature_importances_)[::-1][0:23])
        model = self.Model
        #text= self.frontendpred(np.array(datapoint),model,user)
        #value = model.predict(pd.DataFrame([datapoint]))
        #return text,value #pickle.dump(model,open("model.pkl","wb"))

    def run(self,datapoint,user):
        
        
        model = pickle.load(open('model.pkl','rb'))
        self.Gettraindata()
        self.getfeaturenames()
        text= self.frontendpred(np.array(datapoint),model,user)
        value = model.predict(pd.DataFrame([datapoint]))
        return text,value


class Explainablelogisticregression:

    def __init__(self,model=None,train=None,column_names=[],X_train=None,Y_train=None,Imputer=None,Scaler=None,orig=None):
        self.Model = model
        self.Traindata = train
        self.column_names = column_names
        self.X_train=X_train
        self.y_train=Y_train
        self.Imputer = Imputer
        self.Scaler = Scaler 
        self.column_names_orig = orig



    def Gettraindata(self):
        # Username of your GitHub account
        username = 'Mattjben'
        # Personal Access Token (PAO) from your GitHub account
        token = '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'
        # Creates a re-usable session object with your creds in-built
        github_session = requests.Session()
        github_session.auth = (username, token)
        # Downloading the csv file from your GitHub
        url = "https://raw.githubusercontent.com/IwVr/CSIDS-Finance/main/Datasets/heloc_dataset_v1.csv" # Make sure the url is the raw version of the file on GitHub
        download = github_session.get(url).content
        # Reading the downloaded content and making it a pandas dataframe
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))
        
    
        self.Traindata = df
        a= df.drop('RiskPerformance', axis=1)
        self.column_names_orig = a.columns
        return df

    def transformdata(self):
        df=self.Traindata
        X = df.drop('RiskPerformance', axis=1)
        y = df.RiskPerformance

        # Encoding target varible 
        y = pd.get_dummies(y, drop_first=True, dtype=np.int64)

        # Encode special values:
        import warnings
        warnings.filterwarnings("ignore")
        def get_special_dummies(X, col):
            """
            One-hot encode for -7, -8, -9 values in each column
            """
            X[col + '_-7'] = X[col].apply(lambda row:int(row==-7))
            X[col + '_-8'] = X[col].apply(lambda row:int(row==-8))
            X[col + '_-9'] = X[col].apply(lambda row:int(row==-9))

        for col in X.columns.values.tolist():
            get_special_dummies(X, col)

        # Impute special values: 
        X[X < 0] = np.nan

        X_save = X.copy()
        Imputer = SimpleImputer(strategy='mean')
        scaler = preprocessing.MinMaxScaler()
        X = Imputer.fit_transform(X)
        X= scaler.fit_transform(X)
        self.Scaler=scaler
        self.Imputer=Imputer
        X=pd.DataFrame(X,columns=X_save.columns)
        X['RiskPerformance']=y
        self.Traindata=X
        return self.Traindata

    def getfeaturenames(self):
        df = self.Traindata
        
        df['RiskPerformance'] = pd.get_dummies(df['RiskPerformance'], drop_first=True, dtype=np.int64)
        X = df.drop('RiskPerformance',axis=1)
        fs_indices_rfi =[ 3,  4,  0, 13, 17, 14,  1,  7, 19, 22,  8, 18, 15, 11, 66,  2, 65,
       12, 21, 20, 16,  9,  5, 48,  6, 87, 78, 84, 40, 32, 39, 38, 37, 36,
       35, 34, 33, 28, 31, 30, 29, 10, 27, 26, 25, 24, 23, 42, 41, 91, 43,
       77, 69, 70, 71, 72, 73, 74, 75, 76, 79, 44, 80, 81, 82, 83, 85, 86,
       88, 89, 68, 67, 64, 63, 90, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56,
       57, 58, 59, 60, 61, 62, 45]
        a=fs_indices_rfi[0:30]
        self.column_names = X.columns[a].to_list()
    
        X_train = df.drop("RiskPerformance", axis = 1).values
        y_train = df["RiskPerformance"].values
        X_train_final = X_train[:, np.r_[fs_indices_rfi[0:30]]]
        self.X_train = X_train_final
        self.y_train=y_train 

        return self.column_names
    def model(self):
       
        model = LogisticRegression(max_iter=500,C=100)
        print("------Fitting model--------")
        model = model.fit(self.X_train, self.y_train)
        print("------Fitted--------")
        self.Model =model
    
        pickle.dump(model, open('model2.pkl','wb'))



    def frontendpred(self,model,datapoint):
        print('intercept ', model.intercept_[0])
        print('classes', model.classes_)
        res=pd.DataFrame({'coefficient': model.coef_[0]}, 
                    index=self.column_names)


        
        answer=0
        neg_ans =0
        pos_ans=0
        contributions = {}
        
        for i,val in enumerate(datapoint[0]):
            
            contributions[res.iloc[i].name]=float(res.iloc[i].values[0])*float(val)
            
            answer+= float(res.iloc[i].values[0])*float(val)
            if float(res.iloc[i].values[0]) < 0:
                neg_ans+= float(res.iloc[i].values[0])*float(val)
            else:
                pos_ans+= float(res.iloc[i].values[0])*float(val)
        answer1=answer
        answer+= model.intercept_[0]
        
        d = Counter(contributions)
        
        if answer < 0:
            line=[]
            for feature, contribution in d.most_common()[::-1][0:5]:
                print(answer,'hi')
                print(contribution)
                print((contribution/answer)*100)
                line.append('\n'+'The '+feature+' contributed to '+str(round(abs((contribution/neg_ans)*100),2))+' % of the final result')
                
        else: 
            line=[]
            for feature, contribution in d.most_common(5):
                line.append('\n'+'The '+feature+' contributed to '+str(round(abs((contribution/pos_ans)*100),2))+' % of the final result')
                

            
        return res,line
    def transformtest(self,df):
                # Specify X and y
        X = df
       
       

        # Encode special values:
        import warnings
        warnings.filterwarnings("ignore")
        def get_special_dummies(X, col):
            """
            One-hot encode for -7, -8, -9 values in each column
            """
            X[col + '_-7'] = X[col].apply(lambda row:int(row==-7))
            X[col + '_-8'] = X[col].apply(lambda row:int(row==-8))
            X[col + '_-9'] = X[col].apply(lambda row:int(row==-9))

        for col in X.columns.values.tolist():
            get_special_dummies(X, col)

        # Impute special values: 
        X[X < 0] = np.nan
 
        X=self.Imputer.transform(X)

        X=self.Scaler.transform(X)
        fs_indices_rfi = [ 3,  4,  0, 13, 17, 14,  1,  7, 19, 22,  8, 18, 15, 11, 66,  2, 65,
       12, 21, 20, 16,  9,  5, 48,  6, 87, 78, 84, 40, 32, 39, 38, 37, 36,
       35, 34, 33, 28, 31, 30, 29, 10, 27, 26, 25, 24, 23, 42, 41, 91, 43,
       77, 69, 70, 71, 72, 73, 74, 75, 76, 79, 44, 80, 81, 82, 83, 85, 86,
       88, 89, 68, 67, 64, 63, 90, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56,
       57, 58, 59, 60, 61, 62, 45]
        a=fs_indices_rfi[0:30]
        X=X[:, np.r_[a]]

        return X
    def main(self):
    
        self.Gettraindata()
        data=self.transformdata()
        self.getfeaturenames()
        self.model()
        return data.loc[1]
        #text= self.frontendpred(np.array(datapoint),model,user)
        #value = model.predict(pd.DataFrame([datapoint]))
        #return text,value #pickle.dump(model,open("model.pkl","wb"))

    def run(self,datapoint):
        
        
        model = pickle.load(open('model2.pkl','rb'))
        
        self.Gettraindata()
        self.transformdata()
        self.getfeaturenames()
        
        X=self.transformtest(pd.DataFrame([datapoint],columns=self.column_names_orig))
        res,line= self.frontendpred(model,X)
        value = model.predict(pd.DataFrame(X,columns=self.column_names))
        print(value)
        return value,res,line

class Explainablelightgbm:

    def __init__(self,model=None,train=None,column_names=[],X_train=None,Y_train=None,Imputer=None,Scaler=None,orig=None):
        self.Model = model
        self.Traindata = train
        self.column_names = column_names
        self.X_train=X_train
        self.y_train=Y_train
        self.Imputer = Imputer
        self.Scaler = Scaler 
        self.column_names_orig = orig



    def Gettraindata(self):
        # Username of your GitHub account
        username = 'Mattjben'
        # Personal Access Token (PAO) from your GitHub account
        token = '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'
        # Creates a re-usable session object with your creds in-built
        github_session = requests.Session()
        github_session.auth = (username, token)
        # Downloading the csv file from your GitHub
        url = "https://raw.githubusercontent.com/IwVr/CSIDS-Finance/main/Datasets/heloc_dataset_v1.csv" # Make sure the url is the raw version of the file on GitHub
        download = github_session.get(url).content
        # Reading the downloaded content and making it a pandas dataframe
        df = pd.read_csv(io.StringIO(download.decode('utf-8')))
        
    
        self.Traindata = df
        a= df.drop('RiskPerformance', axis=1)
        self.column_names_orig = a.columns
        return df

    def transformdata(self):
        df=self.Traindata
        X = df.drop('RiskPerformance', axis=1)
        y = df.RiskPerformance

        # Encoding target varible 
        y = pd.get_dummies(y, drop_first=True, dtype=np.int64)

        # Encode special values:
        import warnings
        warnings.filterwarnings("ignore")
        def get_special_dummies(X, col):
            """
            One-hot encode for -7, -8, -9 values in each column
            """
            X[col + '_-7'] = X[col].apply(lambda row:int(row==-7))
            X[col + '_-8'] = X[col].apply(lambda row:int(row==-8))
            X[col + '_-9'] = X[col].apply(lambda row:int(row==-9))

        for col in X.columns.values.tolist():
            get_special_dummies(X, col)

        # Impute special values: 
        X[X < 0] = np.nan

        X_save = X.copy()
        Imputer = SimpleImputer(strategy='mean')
        scaler = preprocessing.MinMaxScaler()
        X = Imputer.fit_transform(X)
        X= scaler.fit_transform(X)
        self.Scaler=scaler
        self.Imputer=Imputer
        X=pd.DataFrame(X,columns=X_save.columns)
        X['RiskPerformance']=y
        self.Traindata=X
        return self.Traindata

    def getfeaturenames(self):
        df = self.Traindata
        
        df['RiskPerformance'] = pd.get_dummies(df['RiskPerformance'], drop_first=True, dtype=np.int64)
        X = df.drop('RiskPerformance',axis=1)
        fs_indices_rfi = [ 3,  4,  0, 13, 17, 14,  1,  7, 19, 22,  8, 18, 15, 11, 66,  2, 65,
       12, 21, 20, 16,  9,  5, 48,  6, 87, 78, 84, 40, 32, 39, 38, 37, 36,
       35, 34, 33, 28, 31, 30, 29, 10, 27, 26, 25, 24, 23, 42, 41, 91, 43,
       77, 69, 70, 71, 72, 73, 74, 75, 76, 79, 44, 80, 81, 82, 83, 85, 86,
       88, 89, 68, 67, 64, 63, 90, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56,
       57, 58, 59, 60, 61, 62, 45]
        a=fs_indices_rfi[0:40]
        self.column_names = X.columns[a].to_list()
    
        X_train = df.drop("RiskPerformance", axis = 1).values
        y_train = df["RiskPerformance"].values
        X_train_final = X_train[:, np.r_[fs_indices_rfi[0:40]]]
        self.X_train = X_train_final
        self.y_train=y_train 

        return self.column_names
    def model(self):
       
        model = lgb.LGBMClassifier(n_estimators=200,num_leaves=20,min_data_in_leaf=4,max_depth=5,max_bin=50,bagging_fraction=0.5,
                          bagging_freq=5,feature_fraction=0.24,
                          feature_fraction_seed=9,
                          bagging_seed=9,
                          min_sum_hessian_in_leaf=11,learning_rate= 0.04)
        print("------Fitting model--------")
        model = model.fit(self.X_train, self.y_train)
        print("------Fitted--------")
        self.Model =model
    
        pickle.dump(model, open('model3.pkl','wb'))



    def frontendpred(self,model,datapoint):
        
        
        explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train,feature_names = self.column_names,class_names=['Low Risk','High Risk'],
                                                    kernel_width=3)
        predict_fn = lambda x: model.predict_proba(x).astype(float)
        exp = explainer.explain_instance(np.array(datapoint[0]), predict_fn,num_features=10)
        text = exp.as_list()
        html_data =HTML(exp.as_html())
        line=[]
        
        for i,value in enumerate(text[:6]):
           
            range = re.search(r'(\d+\.?\d*) (?:<=|<|>=|>) ([\w_-]+)\s? (?:<=|<|>=|>) (\d+\.?\d*)',value[0])
            greater = re.search(r'^(\d+\.?\d*) (?:<=|<) ([\w_-]+)\s?$',value[0])
            lesser = re.search(r'^([\w_-]+)\s? (?:<=|<) (\d+\.?\d*)$',value[0])
            if range:
                line.append('\n'+'The '+str(range.group(2))+' was between '+str(range.group(1))+' and '+str(range.group(3))+' which has a LIME weight of '+str(round(value[1],2)))
            if greater:
                line.append('\n'+'The '+str(greater.group(2))+' was greater than or equal to '+str(greater.group(1))+' which has a LIME weight of '+str(round(value[1],2)))
            if lesser:
                line.append('\n'+'The '+str(lesser.group(1))+' was less than or equal to '+str(lesser.group(2))+' which has a LIME weight of '+str(round(value[1],2)))
        
                
        
        return html_data,line
    def transformtest(self,df):
                # Specify X and y
        X = df
       
       

        # Encode special values:
        import warnings
        warnings.filterwarnings("ignore")
        def get_special_dummies(X, col):
            """
            One-hot encode for -7, -8, -9 values in each column
            """
            X[col + '_-7'] = X[col].apply(lambda row:int(row==-7))
            X[col + '_-8'] = X[col].apply(lambda row:int(row==-8))
            X[col + '_-9'] = X[col].apply(lambda row:int(row==-9))

        for col in X.columns.values.tolist():
            get_special_dummies(X, col)

        # Impute special values: 
        X[X < 0] = np.nan
 
        X=self.Imputer.transform(X)

        X=self.Scaler.transform(X)
        fs_indices_rfi = [ 3,  4,  0, 13, 17, 14,  1,  7, 19, 22,  8, 18, 15, 11, 66,  2, 65,
       12, 21, 20, 16,  9,  5, 48,  6, 87, 78, 84, 40, 32, 39, 38, 37, 36,
       35, 34, 33, 28, 31, 30, 29, 10, 27, 26, 25, 24, 23, 42, 41, 91, 43,
       77, 69, 70, 71, 72, 73, 74, 75, 76, 79, 44, 80, 81, 82, 83, 85, 86,
       88, 89, 68, 67, 64, 63, 90, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56,
       57, 58, 59, 60, 61, 62, 45]
        a=fs_indices_rfi[0:40]
        X=X[:, np.r_[a]]

        return X
    def main(self):
    
        self.Gettraindata()
        data=self.transformdata()
        self.getfeaturenames()
        self.model()
        return data.loc[1]
        #text= self.frontendpred(np.array(datapoint),model,user)
        #value = model.predict(pd.DataFrame([datapoint]))
        #return text,value #pickle.dump(model,open("model.pkl","wb"))

    def run(self,datapoint):
        
        
        model = pickle.load(open('model3.pkl','rb'))
        
        self.Gettraindata()
        self.transformdata()
        self.getfeaturenames()
        
        X=self.transformtest(pd.DataFrame([datapoint],columns=self.column_names_orig))
        plot,lines= self.frontendpred(model,X)
        value = model.predict(pd.DataFrame(X,columns=self.column_names))
        
        return value,plot,lines



