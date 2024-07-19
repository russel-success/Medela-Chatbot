import streamlit as st
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

st.write("what is it?")

training = pd.read_csv('Data/Training.csv')
cols= training.columns
#x is all feature names
cols= cols[:-1]
x = training[cols]

#y is the target. the name of disease:prognosis
y = training['prognosis']
y1= y

#take maximium values for features for each prognosis or disease name
reduced_data = training.groupby(training['prognosis']).max()    
#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# 23% test, 77% train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=0)

clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print("DTC accuracy score:",scores.mean())

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
#description from its dictionary
def getDescription():
    global description_list
    with open('Basic_info_data/disease_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

#get severity values from its dictionary
def getSeverityDict():
    global severityDictionary
    with open('Basic_info_data/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction={row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass
        
#get precaution info from its corresponding dictionary
def getprecautionDict():
    global precautionDictionary
    with open('Basic_info_data/disease_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec={row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)
            
  
symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
       
#function to calculate severity of user condition       
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but make sure to take these precautions.")
        
#function to initiate conversation
def getInfo():
    print("\n\nMEDELA HealthCare ChatBot\n")
    print("\nYour Name?----- \n\t\t\t\t\t\t",end="->")
    name=input("")
    if len(name)>=2:
        print("\nHello, ",name,"."," I'm MEDELA, your personal healthcare chatbot")
    else:
        print("pls input a valid name")    
        getInfo()
        
#function to find symptoms that resemble users input returning a list they choose from
def check_pattern(symp_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    #check char by char if input text matches any symptoms in the symp_list
    pred_list=[item for item in symp_list if regexp.search(item)]
    if(len(pred_list)>0):
        #return list of symptoms that resemble input
        return 1,pred_list
    else:
        #no match_noing symptom found
        return 0,[]

#second prediction
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = x
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}

    #initialise input vector with zeros
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    #translate from integer rep to actual disease name
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))


#function to generate understandable rules from tree model
def conv_tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #ensuring consistent formatting of feature names
    #list of symptoms in cols simply extracted by split with delimiter
    symp_pattern=",".join(feature_names).split(",")
    symptoms_present = []
    

    while True:
        print("\n\nTell me the most obvious symptom you are experiencing-----\n\t\t\t\t\t\t",end="->")
        symptom_input = input("")
        conf,cnf_dis= check_pattern(symp_pattern, symptom_input)
        if conf==1:
            print("\n\nHere are searches related to your input: ------")
            #cnf_dis is confirmed list of possible symptoms judged from input
            for match_no, it in enumerate(cnf_dis):
                print(match_no,")",it)
            #last match_no after end of loop will be index of last possible resembling symptom in cnf_dis
            #if there was more than 1  i.e after first loop, 
            #then that index>0 or !=0, so pick corresponding match_no
            if match_no!=0:
                print(f"Select the one you meant (0 - {match_no}):  ", end="\n\t\t\t\t\t\t->")
                conf_inp = int(input(""))
            else:
                conf_inp=0
            #symptom input is disease name at corresponding conf_inp index of possible symptom list
            symptom_input=cnf_dis[conf_inp]
            break
          
        else:
            print("Enter a valid symptom.-------")

    while True:
        try:
            num_days=int(input("\n\nOkay. For what number of days ? : ------\n\t\t\t\t\t\t->"))
            break
        except:
            print("Enter a valid number of days.------",end="\n\t\t\t\t\t->")
     
     
    def recurse(node, depth):    #recursive function to traverse the tree to the prognosis class or leave node   
        indent = "  " * depth
        #check that current node is not a leaf node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            #check if current node or feature name(symptom) matches user input
            if name == symptom_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
            #until we arrive leaf node or disease label    
        else:
            #current node is a disease label- present disease
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            print("\n\nHave you been experiencing: --------") 
            symptoms_exp=[] 
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='\n\t\t\t\t\t\t->')
                while True:
                    inp=input("")
                    print('\n')
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("pls provide proper answers i.e. (yes/no) : --------\n\t\t\t\t\t\t",end="->")
                if(inp=="yes"):symptoms_exp.append(syms)
                #if "no", simply discard. interest is in "yes" response for symptom

            second_prediction=sec_predict(symptoms_exp)
            calc_condition(symptoms_exp,num_days)
            
            #if both predictions are same, do not redundantly display info
            if(present_disease[0]==second_prediction[0]):
                print("\nHere, you may have ", "<<",present_disease[0],">>")
                print("\n\n",description_list[present_disease[0]],"\n")


            else:
                print("\nHere, you may have ", "<<",present_disease[0],">>", "or ", "<<",second_prediction[0],">>\n",)
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precaution_list=precautionDictionary[present_disease[0]]
            print("\n\nTake following measures : ---------\n")
            for  i,j in enumerate(precaution_list):
                print(i+1,")",j)


    recurse(0, 1)
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
conv_tree_to_code(clf,cols)
print("\n\nDone.\nDISCLAIMER: This bot has been built primarily for eductational purposes and may not always be right ----------\n")