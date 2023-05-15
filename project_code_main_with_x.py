from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
import numpy as np
import pandas as pd
import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint

#setting global graphing options
seaborn.set_theme(style='white', palette='deep')
mpl.rcParams['axes.linewidth'] = 0.75
mpl.rcParams['lines.linewidth'] = 1.3
mpl.rcParams['lines.markersize'] = 3.2
plt.rcParams["font.family"] = "serif"
blue = '#1B53A5'
purple = '#901BA5'
orange = '#FB6910'
cyan = '#237D93'



# finding the derivative of a mlp model
def compute_mlp_derivative(x, mlp_model, delta_x=0.000001):
	x = np.longdouble(x)
	delta_x = np.longdouble(delta_x)

	points_df = pd.DataFrame([x,x+delta_x], columns=['x'])
	predictions = mlp_model.predict(points_df.values)

	dydx = np.longdouble(predictions[1] - predictions[0])/delta_x

	return(dydx)


# use the mlp model on a single input
def compute_prediction(input_, mlp_model):
	return ( mlp_model.predict(np.array(input_).reshape(-1,1))[0] )


# use the mlp model on an input row with dydx, yest
def compute_prediction_three_var_dydx_yest_x(row, mlp_model):
	my_array = np.array([row['dydx'], row['yest'], row['x']]).reshape(1, -1)
	return ( mlp_model.predict( my_array )[0] )


# generating the line data:
def data_generator(function, number_of_values, range_start, range_end):

	the_list = []

	for i in range(number_of_values):
		x = (i)/(number_of_values-1) * (range_end-range_start) + range_start
		y = function(x)
		the_list.append([x,y])

	return(the_list)


def x_y_function(x):
	output = np.cos(x)
	return (output)


def x_z_function(x):
	output = -np.sin(x)
	return (output)


def y_z_relation(number_of_values=50):
	output = []

	for i in range(number_of_values):
		x = np.cos(2*np.pi *i/number_of_values)
		y = np.sin(2*np.pi *i/number_of_values)

		output.append([x,y])

	return(output)


# - scaler   = the scaler object (it needs an inverse_transform method)
# - data     = the data to be inverse transformed as a Series, ndarray, ... 
#              (a 1d object you can assign to a df column)
# - ftName   = the name of the column to which the data belongs
# - colNames = all column names of the data on which scaler was fit 
#              (necessary because scaler will only accept a df of the same shape as the one it was fit on)
def invTransform(scaler, data, colName, colNames):
	dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
	dummy[colName] = data
	dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
	return dummy[colName].values






x_y_dataframe = pd.DataFrame( 
	data_generator(
		function=x_y_function, 
		number_of_values=100,
		range_start= -2*np.pi,
		range_end= 2*np.pi
		), 
	columns=['x', 'y']
	)
x_y_dataframe['x_aux'] = x_y_dataframe['x']

x_z_dataframe = pd.DataFrame( 
	data_generator(
		function= x_z_function, 
		number_of_values=50,
		range_start= -2*np.pi,
		range_end= 0
		), 
	columns=['x', 'z']
	)


y_z_dataframe = pd.DataFrame( 
	y_z_relation(),
	columns=['y', 'z']
	)




# here the above dataframes are concatenated to form a unified dataframe with all the data named 'data_frame'
data_frame = pd.merge(left=x_y_dataframe, right=x_z_dataframe, on='x', how='outer')

# scaling prior to training
scaler_A = StandardScaler()
data_frame[['x', 'y','z']] = scaler_A.fit_transform(data_frame[['x', 'y','z']])


# Training a model to go from X -> Y:
x_to_y_mlp_model = MLPRegressor(
	hidden_layer_sizes=(50,50,50), 
	activation='tanh', 
	solver='adam', 
	alpha=0.001, 
	batch_size='auto', 
	learning_rate='adaptive', 
	learning_rate_init=0.002, 
	power_t=0.5, 
	max_iter=10000000, 
	shuffle=True, 
	random_state=None, 
	tol=0.0000001, 
	verbose=False, 
	warm_start=False, 
	momentum=0.5, 
	nesterovs_momentum=True, 
	early_stopping=False, 
	validation_fraction=0.1, 
	beta_1=0.9, 
	beta_2=0.999, 
	epsilon=1e-08, 
	n_iter_no_change=1000, 
	max_fun=15000
	)


x_train = data_frame.dropna(subset=['x', 'y'])['x'].to_frame()
y_train = data_frame.dropna(subset=['x', 'y'])['y']
x_to_y_mlp_model.fit(x_train.values, y_train.values)


# adding a column to the dataframe for the gradient at each point 'x':
data_frame['dydx'] = data_frame.dropna(subset=['x'])['x'].apply(compute_mlp_derivative, mlp_model=x_to_y_mlp_model)
data_frame['yest'] = data_frame.dropna(subset=['x'])['x'].apply(compute_prediction, mlp_model=x_to_y_mlp_model)





# ------------------
# building a second model for y, dydx -> z:



# Training a model to go from y, dydx -> z:
dydx_y_x_to_z_mlp_model = MLPRegressor(
	hidden_layer_sizes=(50,50), 
	activation='tanh', 
	solver='adam', 
	alpha=0.001, 
	batch_size='auto', 
	learning_rate='adaptive', 
	learning_rate_init=0.002, 
	power_t=0.5, 
	max_iter=100000, 
	shuffle=True, 
	random_state=None, 
	tol=0.000001, 
	verbose=False, 
	warm_start=False, 
	momentum=0.3, 
	nesterovs_momentum=True, 
	early_stopping=False, 
	validation_fraction=0.1, 
	beta_1=0.9, 
	beta_2=0.999, 
	epsilon=1e-08, 
	n_iter_no_change=100, 
	max_fun=15000
	)


dydx_y_x_train = data_frame.dropna(subset=['dydx', 'z', 'yest', 'x'])[['dydx', 'yest', 'x']]

# dydx_y_x_train['x'].values[:] = 0
# dydx_y_x_train['dydx'].values[:] = 0

z_train = data_frame.dropna(subset=['dydx', 'z', 'yest', 'x'])['z']

# dydx_y_x_to_z_mlp_model.fit(dydx_y_x_train, z_train)

dydx_y_x_to_z_mlp_model.fit(dydx_y_x_train.values, z_train.values)

data_frame['z_predicted'] = data_frame.dropna(subset=['dydx', 'yest']).apply(compute_prediction_three_var_dydx_yest_x, mlp_model=dydx_y_x_to_z_mlp_model, axis=1)

# making more 'branches':
dydx_y_x_to_z_mlp_model.fit(dydx_y_x_train.values, z_train.values)
data_frame['z_predicted_2'] = data_frame.dropna(subset=['dydx', 'yest']).apply(compute_prediction_three_var_dydx_yest_x, mlp_model=dydx_y_x_to_z_mlp_model, axis=1)

dydx_y_x_to_z_mlp_model.fit(dydx_y_x_train.values, z_train.values)
data_frame['z_predicted_3'] = data_frame.dropna(subset=['dydx', 'yest']).apply(compute_prediction_three_var_dydx_yest_x, mlp_model=dydx_y_x_to_z_mlp_model, axis=1)



# -----------------------------------------------
# scaling everything back

data_frame['x'] = invTransform(scaler=scaler_A, data=data_frame['x'], colName='x', colNames=['x', 'y','z'])
data_frame['y'] = invTransform(scaler=scaler_A, data=data_frame['y'], colName='y', colNames=['x', 'y','z'])
data_frame['z'] = invTransform(scaler=scaler_A, data=data_frame['z'], colName='z', colNames=['x', 'y','z'])
data_frame['yest'] = invTransform(scaler=scaler_A, data=data_frame['yest'], colName='y', colNames=['x', 'y','z'])
data_frame['dydx'] = data_frame['dydx']*(scaler_A.var_[1]/scaler_A.var_[0])**(1/2)
data_frame['z_predicted'] = invTransform(scaler=scaler_A, data=data_frame['z_predicted'], colName='z', colNames=['x', 'y','z'])
data_frame['z_predicted_2'] = invTransform(scaler=scaler_A, data=data_frame['z_predicted_2'], colName='z', colNames=['x', 'y','z'])
data_frame['z_predicted_3'] = invTransform(scaler=scaler_A, data=data_frame['z_predicted_3'], colName='z', colNames=['x', 'y','z'])

# ---------------------------



# metrics
# usage: sklearn.metrics.r2_score(y_true, y_pred)

# v vs vest
r_squared_y = sklearn.metrics.r2_score(data_frame.dropna(subset=['y'])['y'], data_frame.dropna(subset=['y'])['yest'])

# a vs a_est_smooth
r_squared_z_training = sklearn.metrics.r2_score(data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z'], data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z_predicted'])
# a vs a_est_rough
r_squared_z_validation = sklearn.metrics.r2_score(data_frame.dropna(subset=['z_predicted']).loc[data_frame['x'] > 0, 'x'].apply(x_z_function), data_frame.dropna(subset=['z_predicted']).loc[data_frame['x'] > 0, 'z_predicted'])


# a vs a_est_smooth
r_squared_z_training_2 = sklearn.metrics.r2_score(data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z'], data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z_predicted_2'])
# a vs a_est_rough
r_squared_z_validation_2 = sklearn.metrics.r2_score(data_frame.dropna(subset=['z_predicted_2']).loc[data_frame['x'] > 0, 'x'].apply(x_z_function), data_frame.dropna(subset=['z_predicted_2']).loc[data_frame['x'] > 0, 'z_predicted_2'])


# a vs a_est_smooth
r_squared_z_training_3 = sklearn.metrics.r2_score(data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z'], data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z_predicted_3'])
# a vs a_est_rough
r_squared_z_validation_3 = sklearn.metrics.r2_score(data_frame.dropna(subset=['z_predicted_3']).loc[data_frame['x'] > 0, 'x'].apply(x_z_function), data_frame.dropna(subset=['z_predicted_3']).loc[data_frame['x'] > 0, 'z_predicted_3'])


print(r_squared_y, r_squared_z_training, r_squared_z_validation)
print(r_squared_z_training_2, r_squared_z_validation_2)
print(r_squared_z_training_3, r_squared_z_validation_3)




# --------------------------
# plot one: training, validation split

fig_plot_1, ax_plot_1 = plt.subplots()
fig_plot_1.set_figheight(4)
fig_plot_1.set_figwidth(6)

fig_plot_1.tight_layout(pad=3.0)

ax_plot_1.axvspan(0, 7, facecolor='grey', alpha=0.1)

seaborn.lineplot(
	data=data_frame,
	x="x_aux", 
	y="z_predicted",
	ax=ax_plot_1,
	color=orange,
	label='predicted(1)',
	alpha=0.5
	)

seaborn.lineplot(
	data=data_frame,
	x="x_aux", 
	y="z_predicted_2",
	ax=ax_plot_1,
	color='purple',
	label='(2)',
	alpha=0.5
	)

seaborn.lineplot(
	data=data_frame,
	x="x_aux", 
	y="z_predicted_3",
	ax=ax_plot_1,
	color='red',
	label='(3)',
	alpha=0.5
	)

seaborn.scatterplot(
	data=data_frame,
	x="x", 
	y="z",
	ax=ax_plot_1,
	label='training',
	zorder=2
	)

ax_plot_1.set_xlabel("X")
ax_plot_1.set_ylabel("Z")

# ax_plot_1.legend(loc='upper center', ncol=2)
ax_plot_1.get_legend().remove()
fig_plot_1.legend(loc='upper center', ncol=4)

ax_plot_1.set_xlim([-7,7])
ax_plot_1.set_ylim([-1.1,1.1])
ax_plot_1.set_xticks([-6, -4, -2, 0, 2, 4, 6])
ax_plot_1.set_yticks([-1, -0.5, 0, 0.5, 1])

fig_plot_1.savefig('plot_one_with_x.png', dpi=600)

plt.show()















