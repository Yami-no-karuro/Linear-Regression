# C - Linear Regression

## Simple Linear Regression, in C

### Intro

**Linear regression** is a statistical method used to model the relationship between a dependent variable and one or more independent variables.  
Linear regression is a type of **Supervised Learning Algorithm** that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets.  
More on [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning), [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) and [Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression) on [Wikipedia](https://en.wikipedia.org/).

### What is this example about?

In this example we'll take a look at **Simple Linear Regression**, the simplest form of **Linear Regression**.  
This form involves only one independent variable and one dependent variable.  
For example, consider a scenario where a farmer wants to predict the yield of a crop based on the amount of fertilizer used.  
By analyzing past data, the farmer notices that increasing the amount of fertilizer generally leads to higher crop yields, up to a certain point.

Using **Simple Linear Regression**, we can establish the mathematical relationship `Y = mX + b` where:

- `Y` The dependend variable, in other words our output or prediction. (The predicted crop yield (es... kgs per hectare))
- `X` The indipendent variabile, in other words our input. (The amount of fertilizer applied (es... kgs per hectare))
- `m` The "slope", or how much **Y** changes for each unit of **X**. (How much the yield changes per unit of fertilizer)
- `b` The "bias", or **Y** when **X** is 0. (The expected yield with no fertilizer)

By fitting the model to historical data or **Training Data**, the farmer can estimate the optimal amount of fertilizer to use for maximizing yield while minimizing waste and cost.
