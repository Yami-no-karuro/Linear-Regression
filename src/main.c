#include <stdio.h>
#include <stdlib.h>

float X[] = {2, 4, 6, 8};
float Y[] = {20, 40, 60, 80};

float weight = 0;
float bias = 0;

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

/**
 * Approximates the derivative of the cost function with respect to weight using the Finite Difference method.
 *
 * @param input - The input value, X.
 * @param label - The label value, Y.
 * @param weight - The current weight value, W.
 * @param bias - The current bias value, B.
 * @param h - A small value for approximation.
 */
float weight_grad(float input, float label, float weight, float bias, float h) 
{
    float cost1 = cost(input, label, weight + h, bias);
    float cost2 = cost(input, label, weight, bias);
    return (cost1 - cost2) / h;
}

/**
 * Approximates the derivative of the cost function with respect to bias using the Finite Difference method.
 *
 * @param input - The input value, X.
 * @param label - The label value, Y.
 * @param weight - The current weight value, W.
 * @param bias - The current bias value, B.
 * @param h - A small value for approximation.
 */
float bias_grad(float input, float label, float weight, float bias, float h)
{
    float cost1 = cost(input, label, weight, bias + h);
    float cost2 = cost(input, label, weight, bias);
    return (cost1 - cost2) / h;
}

int main() 
{
    int epoch = 10000;
    float learning_rate = 1e-4;
    float h = 1e-4;
    int size = sizeof(X) / sizeof(X[0]);

    float loss = 0;
    float grad_w = 0;
    float grad_b = 0;

    for (size_t i = 0; i < epoch; i++) {
        for (size_t j = 0; j < size; j++) {
            loss = cost(X[j], Y[j], weight, bias);
            grad_w = weight_grad(X[j], Y[j], weight, bias, h);
            grad_b = bias_grad(X[j], Y[j], weight, bias, h);

            weight = weight - learning_rate * grad_w;
            bias = bias - learning_rate * grad_b;
        }
    }

    printf("\n");
    printf("Loss: %f \n", loss);
    printf("Weight: %f \n", weight);
    printf("Bias: %f \n", bias);

    float input_value;
    printf("\n");
    printf("Insert X: ");
    scanf("%f", &input_value);

    float predicted_value = predict(input_value, weight, bias);
    printf("Input(X) = %.2f: Predicted (Y) = %.2f\n", input_value, predicted_value);

    return 0;
}
