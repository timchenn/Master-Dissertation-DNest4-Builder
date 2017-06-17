import numpy as np
import dnest4.builder as bd
import pandas as pd

# Test
Iris = pd.read_csv('Iris.csv')

data = {"SL": np.array(Iris['Sepal.Length']).astype("float64"),\
        "SW": np.array(Iris['Sepal.Width']).astype("float64"),\
        "PL": np.array(Iris['Petal.Length']).astype("float64"),\
        "PW": np.array(Iris['Petal.Width']).astype("float64"),\
        "Species": np.array(Iris['Species']).astype("int64")}
data["N_Species"] = len(np.unique(data["Species"]))
data["N"] = len(data["Species"])
data["N_Input"] = 7


# Add Some Ones to the Data for the One Trick
data["ones"] = np.ones(data["N"], dtype="int64")

# Create the model
model = bd.Model()

# Coefficients
for j in range(0, data["N_Species"]):
    for k in range(0, data["N_Input"]):
        name = "beta_{j}_{k}".format(j=j, k=k)
        prior = bd.Normal(0.0, 10.0)
        node = bd.Node(name, prior)
        model.add_node(node)

# Sampling distribution
for i in range(0, data["N"]):

    # Probability Distribution over Categories
    for j in range(0, data["N_Species"]):
        name = "p_{j}_{i}".format(i=i, j=j)
        formula = ""
        formula += "beta_{j}_0 +"
        formula += "beta_{j}_1 * SL{i} + beta_{j}_2 * SW{i} + "
        formula += "beta_{j}_3 * PL{i} + beta_{j}_4 * PW{i} + "
        formula += "beta_{j}_5 * PL{i} * PW{i} + beta_{j}_6 * SL{i} * PW{i}"
        formula = formula.format(i=i, j=j)
        model.add_node(bd.Node(name, bd.Delta(formula)))

    # Normalising constant
    name = "Z_{i}".format(i=i)
    formula = ""
    formula += "exp(p_0_{i}) + exp(p_1_{i}) + exp(p_2_{i})"
    formula = formula.format(i=i)
    model.add_node(bd.Node(name, bd.Delta(formula)))

    # Probability of the data point.
    name = "prob_{i}".format(i=i)
    formula = "exp(p_{Species}_{i}) / Z_{i}".format(i=i, Species = data["Species"][i])
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
