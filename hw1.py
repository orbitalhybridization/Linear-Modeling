from sys import argv
import pandas as pd
import numpy as np
import pdb
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ATTRIBUTE_LABELS = {'symboling':0,'normalized-losses':1,'make':2,'fuel-type':3,
					'aspiration':4,'num-of-door':5,'body-style':6,'drive-wheels':7,
					'engine-location':8,'wheel-base':9,'length':10,'width':11,'height':12,
					'curb-weight':13,'engine-type':14,'num-of-cylinders':15,'engine-size':16,
					'fuel-system':17,'bore':18,'stroke':19,'compression-ratio':20,'horsepower':21,
					'peak-rpm':22,'city-mpg':23,'highway-mpg':24,'price':25}

LABELS_OF_INTEREST = ['wheel-base','length','width','height','curb-weight','engine-size',
					'bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg',
					'highway-mpg']

TARGETS = ['price']

def main(filename):

	print("Running main function")
	print("---------------------------")

	print("Importing Data")
	print("---------------------------")
	data = import_data(filename,unknown_char='?')

	print("Preprocessing")
	print("---------------------------")
	data = preprocess(data)

	print("Plotting Initial Data")
	print("---------------------------")
	plot_vs_target(data,LABELS_OF_INTEREST,TARGETS,subplot_dim=[3,5]) # make a 3x5 subplot

	print("Plotting Pairwise Data")
	print("---------------------------")
	plot_pairwise(data,LABELS_OF_INTEREST,subplot_dim=[4,4])
	
	print("Computing Model 1")
	print("---------------------------")
	#linear_model(data,['wheel-base','engine-size'],'price')


	print("Computing Model 2")
	print("---------------------------")
	data['highway-mpg-inverse'] = 1/data['highway-mpg'] # invert highway-mpg
	#linear_model(data,['wheel-base','curb-weight','highway-mpg-inverse'],'price')

	print("Computing Model 3")
	print("---------------------------") # higher dimensions?
	#inear_model(data,['length','height','horsepower'],'price')
	
	display_plots()

	print("Done.")

def import_data(filename,unknown_char=np.nan):
	df = pd.read_csv(filename,names=ATTRIBUTE_LABELS.keys())
	df = df.replace(unknown_char,np.nan) # replace unknowns with column means
	return df

def preprocess(df):

	#df = df.dropna() # remove empty rows

	# remove features not of interest
	df = df[LABELS_OF_INTEREST+TARGETS]

	df = df.astype(float) # make float

	# remove data points for which target variable is unknown
	for label in LABELS_OF_INTEREST+TARGETS: # go through columns	
		df.fillna(df[label].mean(skipna=True),inplace=True)

	return df


def plot_vs_target(df,variables,targets,subplot_dim=[1,1]):
	index=1
	plt.figure(1)
	for var in variables:
		for target in targets:
			#pdb.set_trace()
			#df.plot.scatter(x=var,y=target)
			plt.subplot(subplot_dim[0],subplot_dim[1],index,title=var+' vs. '+target)
			plt.scatter(list(df[var]),list(df[target]))
			index+=1

def plot_pairwise(df,variables,subplot_dim=[1,1]):
	index = 1
	plt.figure(2)
	for var1 in variables: # get every third
		for var2 in ['compression-ratio']:
			if var1 == var2:
				continue
			plt.subplot(subplot_dim[0],subplot_dim[1],index,title=var1+' vs. '+var2)
			plt.scatter(list(df[var1]),list(df[var2]))
			plt.xlabel(var1)
			plt.ylabel(var2)
			index+=1

def linear_model(df,variables,target):

	#perform any necessary transformations on the data

	# split data
	#x_train,x_test,y_train,y_test = train_test_split(df[variables],df[target],test_size=0.25)

	# perform linear regression
	model = LinearRegression()
	model = model.fit(df[variables],df[target])
	predicted_target = model.predict(df[variables])
	plt.figure()
	plt.scatter(df[target],predicted_target)
	plt.xlabel("Actual "+target)
	plt.ylabel("Predicted "+target)
	# extract weights
	print("Model parameters:")
	print(model.coef_)
	print()
	print("R^2 for this model:")
	print(model.score(df[variables],df[target]))
	print("\n-----------------------------------------------------\n\n")



def display_plots():
	plt.show()


if __name__ == "__main__":

	main(argv[1])