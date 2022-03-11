import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


'''
In this file, I want to find the best way to predict whether someone's
income is more than 50k or not.
'''
def run():
	# Read x and y:
	x, x_final, y = import_data()
	x, x_final = clean_data(x, x_final)
	x_train, x_test, y_train, y_test = split_data(x,y)

	# Trying XGBClassifier
	# cm, report =\
				# train_evaluate_xgbclassifier(x_train,x_test,y_train,y_test)



	# Trying Ranfomr forest
	# cm, report =\
				# train_evaluate_randomforest(x_train,x_test,y_train,y_test)

	df_final = predict_test_data(x_final, x_train, y_train)



def import_data():
	df_train = pd.read_csv('census-income-training.csv', index_col='Id')
	# df_train = df_train.iloc[:100, :]
	df_test = pd.read_csv('census-income-test.csv',index_col='Id')
	# df_test = df_test.iloc[:100, :]

	# Define all columns except the income column as x
	x = df_train.drop(['income_morethan_50K'], axis=1)
	x_final = df_test

	# Define income column as y 
	y = df_train.iloc[:,-1]
	# y = df_train.iloc[:100,-1]
	return x, x_final, y




def clean_data(x, x_final):

	# Replace the missing values with the most frequent observations in the column
	imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	imputer.fit(x.iloc[:,19:33])
	x.iloc[:,19:33] = imputer.transform(x.iloc[:,19:33])
	imputer.fit(x_final.iloc[:,19:33])
	x_final.iloc[:,19:33] = imputer.transform(x_final.iloc[:,19:33])

	# Label encode the columns that have labels, these columns include:
	# Column 1, 4, 6 to 15, 19 to 23, 25 to 28, 30 to 34, 36
	col_list = ["ACLSWKR","AHGA","AHSCOL",
				"AMARITL","AMJIND","AMJOCC","ARACE","AREORGN",
				"ASEX","AUNMEM","AUNTYPE","AWKSTAT","FEDTAX",
				"FILESTAT","GRINREG","GRINST","HHDFMX","MIGMTR1",
				"MIGMTR3","MIGMTR4","MIGSAME","PARENT","PEFNTVTY",
				"PEMNTVTY","PENATVTY","PRCITSHP","VETQVA"]
	# Make dummies from categorical columns and drop the first dummy
	x = pd.get_dummies(data=x, columns=col_list, drop_first=True)
	x_final = pd.get_dummies(data=x_final, columns=col_list, drop_first=True)
	'''
	# x and x_final have different columns becauses for PENATVTY which
	is the nationality, they have differnt labels.
	'''
	return x, x_final




def split_data(x,y):
	# Split the data into test and train
	x_train, x_test, y_train, y_test = \
					train_test_split(x,y,test_size=0.2, random_state=0)
	return x_train, x_test, y_train, y_test
	



def train_evaluate_xgbclassifier(x_train,x_test,y_train,y_test):
	

	print("The result for XGBClassifier with default settings:")
	# Train the data with xgbclassifier:
	model = XGBClassifier()
	model.fit(x_train,y_train)
	y_predicted = model.predict(x_test)
	# Check how good the results are
	cm = confusion_matrix(y_test, y_predicted)
	report = classification_report(y_test, y_predicted)
	print(cm, report)

	print("The result for XGBClassifier with max_depth=10:")
	# Train the data with xgbclassifier:
	model = XGBClassifier(max_depth=10)
	model.fit(x_train,y_train)
	y_predicted = model.predict(x_test)
	# Check how good the results are
	cm = confusion_matrix(y_test, y_predicted)
	report = classification_report(y_test, y_predicted)
	print(cm, report)
	
	return cm, report




def train_evaluate_randomforest(x_train,x_test,y_train,y_test):

	# Trying different estimators
	print("The result for Random forest with n_estimators=30")
	model = RandomForestClassifier(n_estimators=30)
	model.fit(x_train, y_train)
	y_predicted = model.predict(x_test)
	cm = confusion_matrix(y_test, y_predicted)
	report = classification_report(y_test, y_predicted)
	print(cm, report)

	print("The result for Random forest with n_estimators=10")
	model = RandomForestClassifier(n_estimators=10)
	model.fit(x_train, y_train)
	y_predicted = model.predict(x_test)
	cm = confusion_matrix(y_test, y_predicted)
	report = classification_report(y_test, y_predicted)
	print(cm, report)

	return cm, report


def predict_test_data(x_final, x, y):

	# Now use the best model to predit the census_income_test.csv:
	# I will use the default setting of xgb classifier
	print("Training the model with XGBClassifier:")
	model = XGBClassifier()
	model.fit(x,y)
	y_predicted = model.predict(x_final)
	df = x_final
	df['income_morethan_50K'] = y_predicted
	df = df.iloc[:,-1]
	print(df)
	df.to_csv('Prediction_20746272.csv')

	return df




if __name__ == "__main__":
	run()