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

# --------------------------------


# reading the input data (x=time, y=velocity, z=acceleration)
# files contain both training and validation data. up to X=420 it is training, past that it is validation.
# data for acceleration past X=420 is reserved for comparison with predictions.
# x: time, y:velocity (v), z: acceleration (a)
# training region (for x vs z) starts from x = 0 to x=420 and validation region starts at x = 420 till x=560
df_v = pd.read_csv('all_rides_v_output_inc_rough.csv', sep=',', names=['x', 'y'])
df_a = pd.read_csv('all_rides_a_output_inc_rough.csv', sep=',', names=['x', 'z'])

df_v_train = df_v.loc[(df_v['x'] <= 420)]
df_v_validation = df_v.loc[(df_v['x'] > 420)]

df_a_train = df_a.loc[(df_a['x'] <= 420)]
df_a_validation = df_a.loc[(df_a['x'] > 420)]

# this dataframe will be updated to include all predictions produced
data_frame = pd.merge(left=df_v, right=df_a_train, on='x', how='outer')

# pprint(data_frame.to_string())

# scaling
scaler_A = StandardScaler()
data_frame[['x', 'y','z']] = scaler_A.fit_transform(data_frame[['x', 'y','z']])


# Training a model to go from X -> Y:
x_to_y_mlp_model = MLPRegressor(
	hidden_layer_sizes=(50,50,50), 
	# hidden_layer_sizes=(10), 
	activation='tanh', 
	solver='adam', 
	alpha=0.01, 
	batch_size='auto', 
	learning_rate='adaptive', 
	learning_rate_init=0.002, 
	power_t=0.5, 
	max_iter=10000000, 
	# max_iter=100, 
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
	n_iter_no_change=5000, 
	max_fun=15000
	)


x_train = data_frame.dropna(subset=['x', 'y'])['x'].to_frame()
y_train = data_frame.dropna(subset=['x', 'y'])['y']
x_to_y_mlp_model.fit(x_train.values, y_train.values)


# adding columns to the dataframe for y estimated and the gradient at each point
data_frame['yest'] = data_frame.dropna(subset=['x'])['x'].apply(compute_prediction, mlp_model=x_to_y_mlp_model)
data_frame['dydx'] = data_frame.dropna(subset=['x'])['x'].apply(compute_mlp_derivative, mlp_model=x_to_y_mlp_model)



# -----------------------------------------------
# Training a model to go from y, dydx -> z:
dydx_y_to_z_mlp_model = MLPRegressor(
	hidden_layer_sizes=(1), 
	activation='tanh', 
	solver='sgd', 
	alpha=0.05, 
	batch_size='auto', 
	learning_rate='adaptive', 
	learning_rate_init=0.002, 
	power_t=0.5, 
	# max_iter=100, 
	max_iter=1000000, 
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
	n_iter_no_change=5000, 
	max_fun=15000
	)



# data_frame['dydx'].values[:] = 0

dydx_train = data_frame.dropna(subset=['z', 'dydx'])[['dydx']]
z_train = data_frame.dropna(subset=['z', 'dydx'])['z']
dydx_y_to_z_mlp_model.fit(dydx_train.values, z_train.values)

# predicting z in a separate column
data_frame['z_predicted'] = data_frame.dropna(subset=['dydx'])['dydx'].apply(compute_prediction, mlp_model=dydx_y_to_z_mlp_model)



# making a separate dataframe to contain the predictions only for graphing (using normalised input X)
graphing_data_frame = pd.DataFrame((np.linspace(0, 560, 5000) - scaler_A.mean_[0])/(scaler_A.var_[0])**(1/2), columns=['x'])
graphing_data_frame['yest'] = graphing_data_frame.dropna(subset=['x'])['x'].apply(compute_prediction, mlp_model=x_to_y_mlp_model)
graphing_data_frame['dydx'] = graphing_data_frame.dropna(subset=['x'])['x'].apply(compute_mlp_derivative, mlp_model=x_to_y_mlp_model)
graphing_data_frame['z_predicted'] = graphing_data_frame.dropna(subset=['dydx'])['dydx'].apply(compute_prediction, mlp_model=dydx_y_to_z_mlp_model)

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

df_misc = pd.merge(df_a_validation, data_frame, on=['x'], how='inner')
# print(df_misc)

# v vs vest
r_squared_y = sklearn.metrics.r2_score(data_frame.dropna(subset=['y'])['y'], data_frame.dropna(subset=['y'])['yest'])
# a vs a_est_smooth
r_squared_z_smooth = sklearn.metrics.r2_score(data_frame.dropna(subset=['z']).loc[data_frame.dropna(subset=['z'])['x'] <= 420, 'z'], data_frame.dropna(subset=['z']).loc[data_frame.dropna(subset=['z'])['x'] <= 420, 'z_predicted'])
# a vs a_est_rough
r_squared_z_rough = sklearn.metrics.r2_score(df_misc['z_x'], df_misc['z_predicted'])


print(r_squared_y, r_squared_z_smooth, r_squared_z_rough)


# ---------------------------
# plot 1
# plotting 2d graph of x vs y and yest
# x vs y in points
# x vs yest in line

fig_plot_1, ax_plot_1 = plt.subplots()
fig_plot_1.set_figheight(3)
fig_plot_1.set_figwidth(11)

fig_plot_1.tight_layout(pad=2.0)

# ax_plot_1.axvspan(0, 420, facecolor='grey', alpha=0.1)

seaborn.lineplot(
	data=graphing_data_frame,
	x="x", 
	y="yest",
	ax=ax_plot_1,
	color=orange,
	label='Model A'
	)

seaborn.scatterplot(
	data=df_v_train,
	x="x", 
	y="y",
	ax=ax_plot_1,
	color=blue,
	label='Smooth surface'
	)

seaborn.scatterplot(
	data=df_v_validation,
	x="x", 
	y="y",
	ax=ax_plot_1,
	color=purple,
	label='Rough surface'
	)

ax_plot_1.set_xlabel("X (time)")
ax_plot_1.set_ylabel("Y (velocity)")


# ax_plot_1.get_legend().remove()
# fig_plot_1.legend(loc='upper center', ncol=3)

ax_plot_1.set_xlim([-10, 570])
# ax_plot_1.set_ylim([-1.05,1.05])
# ax_plot_1.set_xticks([-6, -4, -2, 0, 2, 4, 6])
# ax_plot_1.set_yticks([-1, -0.5, 0, 0.5, 1])

fig_plot_1.savefig('bike_plot_one.png', dpi=600)

plt.show()
# ---------------------------






# ---------------------------
# plot 2
# plotting 2d graph of x vs dydx and z
# x vs z in points
# x vs dydx in line

fig_plot_2, ax_plot_2 = plt.subplots()
fig_plot_2.set_figheight(3)
fig_plot_2.set_figwidth(11)

fig_plot_2.tight_layout(pad=2.3)

ax_plot_2.axvspan(420, 570, facecolor='grey', alpha=0.1)

# seaborn.lineplot(
# 	data=graphing_data_frame,
# 	x="x", 
# 	y="dydx",
# 	# s=15,
# 	ax=ax_plot_2,
# 	color=cyan,
# 	label='Gradient'
# 	)

seaborn.lineplot(
	data=graphing_data_frame,
	x="x", 
	y="z_predicted",
	# s=15,
	ax=ax_plot_2,
	color=orange,
	label='Predicted'
	)

seaborn.scatterplot(
	data=df_a_train,
	x="x", 
	y="z",
	ax=ax_plot_2,
	color=blue,
	label='Smooth surface'
	)

seaborn.scatterplot(
	data=df_a_validation,
	x="x", 
	y="z",
	ax=ax_plot_2,
	color=purple,
	label='Rough surface'
	)

ax_plot_2.set_xlabel("X (time)")
ax_plot_2.set_ylabel("Z (acceleration)")

# ax_plot_2.get_legend().remove()
# fig_plot_2.legend(loc='upper center', ncol=3)

ax_plot_2.set_xlim([-10, 570])
ax_plot_2.set_ylim([-1.8,1.8])
# ax_plot_2.set_xticks([-6, -4, -2, 0, 2, 4, 6])
ax_plot_2.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

fig_plot_2.savefig('bike_plot_two.png', dpi=600)

plt.show()
# ---------------------------






# ---------------------------
# plot 3
# plotting 2d graph of dydx vs Z, and model B's fit

fig_plot_3, ax_plot_3 = plt.subplots()
fig_plot_3.set_figheight(6)
fig_plot_3.set_figwidth(6)

fig_plot_3.tight_layout(pad=3)

seaborn.lineplot(
	data=graphing_data_frame,
	x="dydx", 
	y="z_predicted",
	ax=ax_plot_3,
	color=orange,
	label='Model B'
	)

seaborn.scatterplot(
	data=data_frame,
	x="dydx", 
	y="z",
	ax=ax_plot_3,
	color=blue,
	label='Smooth surface'
	)


ax_plot_3.set_xlabel("Gradient")
ax_plot_3.set_ylabel("Z (acceleration)")
ax_plot_3.axis('equal')

ax_plot_3.get_legend().remove()
ax_plot_3.legend(loc='upper center', ncol=2)

ax_plot_3.set_xlim([-1.1,1.1])
ax_plot_3.set_ylim([-1.1,1.1])
ax_plot_3.set_xticks([-1, -0.5, 0, 0.5, 1])
ax_plot_3.set_yticks([-1, -0.5, 0, 0.5, 1])

plt.savefig('bike_plot_three.png', dpi=600)
plt.show()
# -----------------------------------------------





