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
def compute_prediction_two_var_dydx_yest(row, mlp_model):
	my_array = np.array([row['dydx'], row['yest']]).reshape(1, -1)
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



# Generating the idealised dataset
x_y_dataframe = pd.DataFrame( 
	data_generator(
		function=x_y_function, 
		number_of_values=100,
		range_start= -2*np.pi,
		range_end= 2*np.pi
		), 
	columns=['x', 'y']
	)

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


# adding columns to the dataframe for y estimated and the gradient at each point
data_frame['yest'] = data_frame.dropna(subset=['x'])['x'].apply(compute_prediction, mlp_model=x_to_y_mlp_model)
data_frame['dydx'] = data_frame.dropna(subset=['x'])['x'].apply(compute_mlp_derivative, mlp_model=x_to_y_mlp_model)




# Training a model to go from y, dydx -> z:
dydx_y_to_z_mlp_model = MLPRegressor(
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


# setting all values of dydx = 0 if needed by uncommenting the line:
data_frame['dydx'].values[:] = 0

dydx_y_train = data_frame.dropna(subset=['dydx', 'z', 'yest'])[['dydx', 'yest']]
z_train = data_frame.dropna(subset=['dydx', 'z', 'yest'])['z']
dydx_y_to_z_mlp_model.fit(dydx_y_train.values, z_train.values)


# predicting z in a separate column
data_frame['z_predicted'] = data_frame.dropna(subset=['dydx', 'yest']).apply(compute_prediction_two_var_dydx_yest, mlp_model=dydx_y_to_z_mlp_model, axis=1)




# making a separate dataframe to contain the predictions only for graphing (using normalised input X)
graphing_data_frame = pd.DataFrame((np.linspace(-2*np.pi, 2*np.pi, 3000) - scaler_A.mean_[0])/(scaler_A.var_[0])**(1/2), columns=['x'])
graphing_data_frame['yest'] = graphing_data_frame.dropna(subset=['x'])['x'].apply(compute_prediction, mlp_model=x_to_y_mlp_model)
graphing_data_frame['dydx'] = graphing_data_frame.dropna(subset=['x'])['x'].apply(compute_mlp_derivative, mlp_model=x_to_y_mlp_model)
graphing_data_frame['z_predicted'] = graphing_data_frame.dropna(subset=['dydx', 'yest']).apply(compute_prediction_two_var_dydx_yest, mlp_model=dydx_y_to_z_mlp_model, axis=1)
# -----------------------------------------------

# -----------------------------------------------
# scaling everything back

data_frame['x'] = invTransform(scaler=scaler_A, data=data_frame['x'], colName='x', colNames=['x', 'y','z'])
data_frame['y'] = invTransform(scaler=scaler_A, data=data_frame['y'], colName='y', colNames=['x', 'y','z'])
data_frame['z'] = invTransform(scaler=scaler_A, data=data_frame['z'], colName='z', colNames=['x', 'y','z'])
data_frame['yest'] = invTransform(scaler=scaler_A, data=data_frame['yest'], colName='y', colNames=['x', 'y','z'])
data_frame['dydx'] = data_frame['dydx']*(scaler_A.var_[1]/scaler_A.var_[0])**(1/2)
data_frame['z_predicted'] = invTransform(scaler=scaler_A, data=data_frame['z_predicted'], colName='z', colNames=['x', 'y','z'])


graphing_data_frame['x'] = invTransform(scaler=scaler_A, data=graphing_data_frame['x'], colName='x', colNames=['x', 'y','z'])
graphing_data_frame['yest'] = invTransform(scaler=scaler_A, data=graphing_data_frame['yest'], colName='y', colNames=['x', 'y','z'])
graphing_data_frame['dydx'] = graphing_data_frame['dydx']*(scaler_A.var_[1]/scaler_A.var_[0])**(1/2)
graphing_data_frame['z_predicted'] = invTransform(scaler=scaler_A, data=graphing_data_frame['z_predicted'], colName='z', colNames=['x', 'y','z'])
# ---------------------------



# metrics
# usage: sklearn.metrics.r2_score(y_true, y_pred)

# v vs vest
r_squared_y = sklearn.metrics.r2_score(data_frame.dropna(subset=['y'])['y'], data_frame.dropna(subset=['y'])['yest'])
# a vs a_est_smooth
r_squared_z_training = sklearn.metrics.r2_score(data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z'], data_frame.dropna(subset=['z']).loc[data_frame['x'] <= 0, 'z_predicted'])
# a vs a_est_rough
r_squared_z_validation = sklearn.metrics.r2_score(data_frame.dropna(subset=['z_predicted']).loc[data_frame['x'] > 0, 'x'].apply(x_z_function), data_frame.dropna(subset=['z_predicted']).loc[data_frame['x'] > 0, 'z_predicted'])


print(r_squared_y, r_squared_z_training, r_squared_z_validation)






# -----------------------------------------------
# plotting the 3d graph of Y, dYdX vs Z.

# plotting the surface
@np.vectorize
def z_function(x, y, mlp_model):
	x = (x - scaler_A.mean_[1])/(scaler_A.var_[1])**(1/2)
	y = y*(scaler_A.var_[0]/scaler_A.var_[1])**(1/2)
	my_array = np.array([y, x]).reshape(1, -1)
	return ( (mlp_model.predict( my_array )[0]*(scaler_A.var_[2])**(1/2) + scaler_A.mean_[2]) )

N = 50
X1, Y1 = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

Z1 = z_function(x=X1, y=Y1, mlp_model=dydx_y_to_z_mlp_model)

plt.figure(figsize=(6, 6))
plot_axes = plt.axes(projection='3d')

plot_axes.set_xlabel('Y')
plot_axes.set_ylabel('dY/dX')
plot_axes.set_zlabel('Z')
plot_axes.plot_surface(X1, Y1, Z1, cmap='viridis', linewidth=0.1, alpha=0.7)

plot_axes.set_xticks([-1, -0.5, 0, 0.5, 1])
plot_axes.set_yticks([-1, -0.5, 0, 0.5, 1])
plot_axes.set_zticks([-1, -0.5, 0, 0.5, 1])


# plotting the line parametrised by x for predictions
plot_axes.plot(
	graphing_data_frame.dropna(subset=['x', 'yest', 'dydx', 'z_predicted']).loc[graphing_data_frame['x'] >= 0, 'yest'],
	graphing_data_frame.dropna(subset=['x', 'yest', 'dydx', 'z_predicted']).loc[graphing_data_frame['x'] >= 0, 'dydx'],
	graphing_data_frame.dropna(subset=['x', 'yest', 'dydx', 'z_predicted']).loc[graphing_data_frame['x'] >= 0, 'z_predicted'],
	label='Parametric curve',
	color=orange
	)

# plotting the line parametrised by x for training data
plot_axes.plot(
	data_frame.dropna(subset=['x', 'yest', 'dydx', 'z'])['yest'],
	data_frame.dropna(subset=['x', 'yest', 'dydx', 'z'])['dydx'],
	data_frame.dropna(subset=['x', 'yest', 'dydx', 'z'])['z'],
	label='Parametric curve',
	color=blue
	)

plt.savefig('idealised_plot_3D.png', dpi=600)
plt.show()
# -----------------------------------------------





# -----------------------------------------------
# plot one
# plotting the two graphs of input data
fig, ((ax1), (ax2)) = plt.subplots(2, 1)
fig.set_figheight(6)
fig.set_figwidth(6)

fig.tight_layout(pad=3.0)

seaborn.scatterplot(
	data=data_frame,
	x="x", 
	y="y",
	ax=ax1,
	color=blue
)

seaborn.scatterplot(
	data=data_frame,
	x="x", 
	y="z",
	color=blue,
	ax=ax2
)

ax1.set_xlim([-7,7])
ax2.set_xlim([-7,7])
# ax3.axis('equal')

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax2.set_xlabel("X")
ax2.set_ylabel("Z")

plt.savefig('idealised_plot_one.png', dpi=600)

plt.show()
# -----------------------------------------------



# -----------------------------------------------
# plot two
# plotting the two graphs of fitted data
fig, ((ax1), (ax2)) = plt.subplots(2, 1)
fig.set_figheight(6)
fig.set_figwidth(6)

fig.tight_layout(pad=3)

seaborn.lineplot(
	data=graphing_data_frame,
	x="x", 
	y="yest",
	ax=ax1,
	color=orange,
	label='Model A',
)

seaborn.scatterplot(
	data=data_frame,
	x="x", 
	y="y",
	ax=ax1,
	color=blue,
	label='Input data',
	zorder=2,
	alpha=0.7
)

seaborn.lineplot(
	data=graphing_data_frame,
	x="x", 
	y="dydx",
	color=orange,
	ax=ax2,
)

ax1.set_xlim([-7,7])
ax2.set_xlim([-7,7])
# ax3.axis('equal')
ax1.get_legend().remove()
fig.legend(loc='upper center', ncol=2)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax2.set_xlabel("X")
ax2.set_ylabel("dY/dX")

plt.savefig('idealised_plot_two.png', dpi=600)

plt.show()
# -----------------------------------------------




# -----------------------------------------------
# plot 3: y, yest vs z, dydx vs z
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(3)
fig.set_figwidth(6)

fig.tight_layout(pad=3)

seaborn.lineplot(
	data=graphing_data_frame,
	x="yest", 
	y="z_predicted",
	color=orange,
	ax=ax1,
	label='Z predicted',
	sort=False
)
seaborn.scatterplot(
	data=data_frame,
	x="yest", 
	y="z",
	ax=ax1,
	label='Z training',
	color=blue,
	alpha=0.7,
	zorder=2,
)


seaborn.lineplot(
	data=graphing_data_frame,
	x="dydx", 
	y="z_predicted",
	color=orange,
	ax=ax2,
	sort=False
)

seaborn.scatterplot(
	data=data_frame,
	x="dydx", 
	y="z",
	ax=ax2,
	color=blue,
	alpha=0.7,
	zorder=2,
)


ax1.set_xlim([-1.1,1.1])
ax2.set_xlim([-1.1,1.1])
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
ax2.set_yticks([-1, -0.5, 0, 0.5, 1])

ax1.set_xlabel("Y")
ax1.set_ylabel("Z")
ax2.set_xlabel("Gradient (dY/dX)")
ax2.set_ylabel("Z")

ax1.get_legend().remove()
fig.legend(loc='upper center', ncol=2)

plt.savefig('idealised_plot_three.png', dpi=600)

plt.show()
# -----------------------------------------------




# -----------------------------------------------
# plot four: training, validation split
fig_plot_2, ax_plot_2 = plt.subplots()
fig_plot_2.set_figheight(4)
fig_plot_2.set_figwidth(6)

fig_plot_2.tight_layout(pad=3)

ax_plot_2.axvspan(0, 7, facecolor='grey', alpha=0.1)

seaborn.lineplot(
	data=graphing_data_frame,
	x="x", 
	y="z_predicted",
	ax=ax_plot_2,
	color=orange,
	label='predicted with gradient',
	)

seaborn.scatterplot(
	data=data_frame,
	x="x", 
	y="z",
	ax=ax_plot_2,
	label='training',
	color=blue,
	alpha=0.7,
	zorder=2,
	)

ax_plot_2.set_xlabel("X")
ax_plot_2.set_ylabel("Z")

# ax_plot_2.legend(loc='upper center', ncol=2)
ax_plot_2.get_legend().remove()
fig_plot_2.legend(loc='upper center', ncol=2)

ax_plot_2.set_xlim([-7,7])
ax_plot_2.set_ylim([-1.05,1.05])
ax_plot_2.set_xticks([-6, -4, -2, 0, 2, 4, 6])
ax_plot_2.set_yticks([-1, -0.5, 0, 0.5, 1])

fig_plot_2.savefig('idealised_plot_four.png', dpi=600)

plt.show()








