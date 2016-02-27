import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# this file is for Kelsey Hern
# This loads the data from the csv file
data = np.recfromcsv("Adult.csv", delimiter = ",", skip_header=2)
## currently data is an array of the rows, we need to chose what
## columns we want to get out

body_len = [] # element 3
weight = [] # element 4

for r in data:
	body_len.append(r[3])
	weight.append(r[4])
body_len = np.array(body_len)
weight = np.array(weight)

# get regression for body_len and weight
linregress = stats.linregress(weight, body_len)
# linregress is slope, intercept, rvalue .....
#now set up the linear regression line
# x is a range of values from the minimum weight to the max
x = np.arange(int(weight.min()), int(weight.max()))
y_predict = linregress.slope * x + linregress.intercept

# now plot the data
plt.plot(x,y_predict)
plt.scatter(weight,body_len)
plt.show()