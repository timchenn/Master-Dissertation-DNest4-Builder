import numpy as np
import dnest4.builder as bd
import pandas as pd

# The data, as a dictionary
Clocks = pd.read_csv('Clocks.csv')
data = {}
data = {'log_age': np.log(np.array(Clocks['age']).astype('float64')),\
        'log_num_bidders': np.log(np.array(Clocks['num_bidders'])).astype('float64'),\
        'log_y': np.log(np.array(Clocks['y']).astype('float64'))}
data["N"] = int(len(data["log_y"]))

# Model
model = bd.Model()

# Slopes and Intercept
model.add_node(bd.Node("beta0", bd.Normal(0, 100)))
model.add_node(bd.Node("beta1", bd.Normal(0, 100)))
model.add_node(bd.Node("beta2", bd.Normal(0, 100)))
model.add_node(bd.Node("beta3", bd.Normal(0, 100)))

# Noise standard deviation
model.add_node(bd.Node("sigma", bd.LogUniform(0.001, 100)))

# Sampling distribution
for i in range(0, data["N"]):
    name = "log_y{i}".format(i = i)
    mean = "beta0 + beta1 * log_age{i}  + beta2 * log_num_bidders{i} +"
    mean += "beta3 * log_age{i} * log_num_bidders{i}"
    mean = mean.format(i = i)
    model.add_node(bd.Node(name, bd.Normal(mean, "sigma"), observed = True))
        
# Create the C++code
bd.generate_h(model, data)
bd.generate_cpp(model, data)

# Compile the C++code so it 's ready to go
import os
os.system("make")