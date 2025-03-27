# C - Linear Regression

## Simple Linear Regression, in C

### Intro

**Linear regression** is a statistical method used to model the relationship between a dependent variable and one or more independent variables.  
Linear regression is a type of **Supervised Learning Algorithm** that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets.  
(More on [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning), [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) and [Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression) on [Wikipedia](https://en.wikipedia.org/))

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

### Implementation

The `predict()` function directly implements the mathematical equation of **Simple Linear Regression** `Y = mX + b`. 

```c
/**
 * Computes the predicted value based on current weight and bias.
 *
 * @param input - The input value, X.
 * @param weight - The current weight value, W.
 * @param bias - The current bias value, B.
 */
float predict(float input, float weight, float bias) 
{
    return (input * weight) + bias;
}
```

To train the model, we need a way to measure how well it performs.  
This is done using the `cost()` function, which quantifies the difference between the predicted values and the actual values from the dataset.

In this implementation, we use the **Mean Squared Error (MSE)**.  
(More on the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) on [Wikipedia](https://en.wikipedia.org/))

```c
/**
 * Computes the MSE (mean squared error), which quantifies the difference between the actual label and the predicted value.
 *
 * @param input - The input value, X.
 * @param label - The label value, Y.
 * @param weight - The current weight value, W.
 * @param bias - The current bias value, B.
 */
float cost(float input, float label, float weight, float bias)
{
    float predicted = predict(input, weight, bias);
    float loss = (label - predicted) * (label - predicted);

    return loss / 2;
}
```
