import numpy as np
import dnest4.builder as bd
import pandas as pd

# Data Input as a Python Dictionary
CHD = pd.read_csv('CHD.csv')

data = {"CHD": np.array(CHD["CHD"]).astype("int64"),\
        "Age": np.array(CHD["Age"]).astype("int64")}
data["N"] = len(data["CHD"])
data["N_Input"] = 2
data["N_CHD"] = 2

# Add Some Ones to the Data for the One Trick
data["ones"] = np.ones(data["N"], dtype="int64")

# Create the model
model = bd.Model()

# Coefficients
model.add_node(bd.Node("beta_0_0", bd.Normal(0, 10)))
model.add_node(bd.Node("beta_0_1", bd.Normal(0, 10)))
model.add_node(bd.Node("beta_1_0", bd.Normal(0, 10)))
model.add_node(bd.Node("beta_1_1", bd.Normal(0, 10)))

# Sampling distribution
for i in range(0, data["N"]):
    for j in range(0, data["N_CHD"]):
        name = "p_{j}_{i}".format(i=i, j=j)
        formula = ""
        formula += "beta_{j}_0 + beta_{j}_1 * Age{i} " 
        formula = formula.format(i=i, j=j)
        model.add_node(bd.Node(name, bd.Delta(formula)))
    # Probability Distribution over Categories
    #name = "p_0_{i}".format(i=i)
    #formula = ""
    #formula += "beta_{0}_0 + beta_{0}_1 * Age{i} " 
    #formula = formula.format(i=i)
    #model.add_node(bd.Node(name, bd.Delta(formula)))
    
    #name = "p_1_{i}".format(i=i)
    #formula = ""
    #formula += "beta_{1}_0 + beta_{1}_1 * Age{i} " 
    #formula = formula.format(i=i)
    #model.add_node(bd.Node(name, bd.Delta(formula)))

    # Normalising constant
    name = "Z_{i}".format(i=i)
    formula = ""
    formula += "exp(p_0_{i}) + exp(p_1_{i})"
    formula = formula.format(i=i)
    model.add_node(bd.Node(name, bd.Delta(formula)))

    # Probability of the data point.
    name = "prob_{i}".format(i=i)
    formula = "exp(p_{CHD}_{i}) / Z_{i}"
    formula = formula.format(i=i, CHD = data["CHD"][i])
    model.add_node(bd.Node(name, bd.Delta(formula)))
 
    # One Trick - this is like the "zeroes trick" in JAGS
    name = "ones{i}".format(i=i)
    distribution = bd.Binomial(1, "prob_{i}".format(i=i))
    node = bd.Node(name, distribution, observed=True)
    model.add_node(node)

# Create the C++ code
bd.generate_h(model, data)
bd.generate_cpp(model, data)

# Compile the C++ code so it's ready to go
import os
os.system("make")