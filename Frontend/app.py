

from io import TextIOWrapper
import csv
from flask import Flask, render_template , request, session
import matplotlib.pyplot as plt
from predictfunc import Explainablelogisticregression , Explainablelightgbm
from predictfunc import ExplainableDecisonTree

# Create a Flask Instance
app = Flask(__name__)

# Define secret key to enable session
app.secret_key = 'data janitors'

 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
# Configure upload file path flask


@app.route("/")
def Intro():
    return render_template("Intropage.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route('/home',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        if request.method == 'POST':
            csv_file = request.files['uploaded-file']
            csv_file = TextIOWrapper(csv_file, encoding='utf-8')
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            data = []
            for row in csv_reader:
                data.append(row)
            session['data'] = data
            
        return render_template("index2.html")
@app.route("/lr")
def lr():
    return render_template("indexlr.html")
@app.route('/lr',  methods=("POST", "GET"))
def uploadFilelr():
    if request.method == 'POST':
        if request.method == 'POST':
            csv_file = request.files['uploaded-file']
            csv_file = TextIOWrapper(csv_file, encoding='utf-8')
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            data = []
            for row in csv_reader:
                data.append(row)
            session['data'] = data
            
        return render_template("index2lr.html")


@app.route('/predict',methods=['GET','POST'])
def predict():
    data = session.get('data', None)
    user_data={}
    for line in data:
        name=line[0]
        line=[int(i) for i in line[1:]]
        user_data[name]=line
    user_outputs ={}
    imgs=[]
    prediction={}
    for user in user_data:
   
        
    
        features = user_data[user]
        edt_array,value=ExplainableDecisonTree().run(features,user)
        img='./static/dtreeviz_'+str(user)+'.svg'
        #img = fig
        imgs.append(img)
        if value[0] == 0:
            credit = 'Low Risk'
        else:
            credit = 'High Risk'
        
        

        output=(ExplainableDecisonTree.explainoutput(edt_array))
        user_outputs[user]=output
        prediction[user]=credit
        
    return render_template("predict.html",your_list=user_outputs,pred=prediction,img=imgs,debug=True)


@app.route('/lrpredict',methods=['GET','POST'])
def lrpredict():
    data = session.get('data', None)
    user_data={}
    for line in data:
        name=line[0]
        line=[int(i) for i in line[1:]]
        user_data[name]=line
    
    
    prediction={}
    credit_ = {}
    for user in user_data:
   
        
    
        features = user_data[user]
        print(features)
        value,table,lines=Explainablelogisticregression().run(features)
        
        #img = fig
        #imgs.append(img)
        if value[0] == 0:
            credit = 'Low Risk'
        else:
           credit = 'High Risk'
        
        

        #output=(ExplainableDecisonTree.explainoutput(edt_array))
        
        prediction[user]=lines
        credit_[user]= credit
        
    return render_template("lrpredict.html",prediction=prediction,credit=credit_,table=table.to_html(classes='data'),debug=True)
@app.route("/risk")
def risk():
    return render_template("risk.html")
@app.route("/lgbm")
def lgbm():
    return render_template("indexlgbm.html")
@app.route('/lgbm',  methods=("POST", "GET"))
def uploadFilelgbm():
    if request.method == 'POST':
        if request.method == 'POST':
            csv_file = request.files['uploaded-file']
            csv_file = TextIOWrapper(csv_file, encoding='utf-8')
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            data = []
            for row in csv_reader:
                data.append(row)
            session['data'] = data
            
        return render_template("index2lgbm.html")


@app.route('/predictlgbm',methods=['GET','POST'])
def predictlgbm():
    data = session.get('data', None)
    user_data={}
    for line in data:
        name=line[0]
        line=[int(i) for i in line[1:]]
        user_data[name]=line
    user_outputs ={}
    imgs=[]
    top5 ={}
    prediction={}
    for user in user_data:
   
        
    
        features = user_data[user]
        value,table,lines=Explainablelightgbm().run(features)
        
        
        if value[0] == 0:
            credit = 'Low Risk'
        else:
            credit = 'High Risk'
        
        

        imgs.append(table)
        user_outputs[user]=credit
        top5[user]=lines
        
        
    return render_template("predictlgbm.html",your_list=user_outputs,imgs=imgs,top5=top5,debug=True)



    
    

