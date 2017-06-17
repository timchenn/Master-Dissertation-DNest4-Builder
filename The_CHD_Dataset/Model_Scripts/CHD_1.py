import numpy as np
import dnest4.builder as bd
import pandas as pd

CHD = pd.read_csv('CHD.csv')

data = {"CHD": np.array(CHD["CHD"]).astype("int64"),\
        "Age": np.array(CHD["Age"]).astype("int64"),\
        "AgeGrp": np.array(CHD["AgeGrp"]).astype("int64")}
data["N"] = len(data["CHD"])

# Create the model
model = bd.Model()

# Coefficients
model.add_node(bd.Node("beta_0", bd.Normal(0, 10)))
model.add_node(bd.Node("beta_1", bd.Normal(0, 10)))

# Sampling Distribution
for i in range(0, data["N"]):
    name = "CHD{i}".format(i=i)
    #prob = "exp(beta_0 + beta_1 * Age{i})/(1.0 + exp(beta_0 + beta_1 * Age{i}))"
    prob = "exp(beta_0 + beta_1 * Age{i})"
    prob += "/(1.0 + exp(beta_0 + beta_1 * Age{i}))"
    prob = prob.format(i = i)
    distribution = bd.Binomial(1, prob)
    node = bd.Node(name, distribution, observed = True)
    model.add_node(node)


# Extra node for prediction
name = "CHDnew"
prob = "exp(beta_0 + beta_1 * 35)/(1.0 + exp(beta_0 + beta_1 * 35))"
distribution = bd.Delta(prob)
node = bd.Node(name, distribution)
model.add_node(node)


# Create the C++ code
bd.generate_h(model, data)
bd.generate_cpp(model, data)

# Compile the C++ code so it's ready to go
import os
os.system("make")