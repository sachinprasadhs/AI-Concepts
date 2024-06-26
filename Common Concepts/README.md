## Activation Functions: 

![Activation Functions](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/activation%20functions.png)

Activation functions are used to bring the non linearity in the equation.
If the activation functions are not present or if the linear activation function is used then the model will learn the same thing in each layer(basically an output of the linear equation). 

Composition of linear function is linear equation.

By having a non-linear activation function, basically it decides how much of the weighted sum of that particular node is carried forward in a layer of the network.
Some of the most commonly used activation functions are. 

**`Sigmoid (0,1)`** : <br> 
Mostly used in the output layer for binary classification.  <br>`Derivative`: `a(1-a)  range[0-0.25]`


**`Tanh (-1,1)`** : <br>
mMostly used in hidden layers but computation is slower compared to relu but superior to sigmoid since it centers your data since mean is closer to zero as it includes both positive and negative values which helps in easier learning of weights in the next layer.
 <br>`Derivative`:  `1 - a2 range(0 to 1)`


**`Relu max(0,Z)`** : <br>
Mostly used in hidden layers and computation is faster, it makes all the negative values to zero and also helps to overcome the vanishing gradient problem.
 <br>`Derivative`:  `0 if z < 0, 1 if z>= 0`


**`Leaky Relu (0.01Z, Z)`** : <br>
Adds an alpha value when the weights are negative which helps in retaining the derivative instead of making it to 0.
 <br>`Derivative`:  `0.01 if z < 0, 1 if z>= 0`


**`Softmax`:**
The softmax is the multiclass counterpart of the sigmoid. If you have $\(N\)$ mutually exclusive events, and their logits are given by $\(Y_j\)$, then softmax $\(\sigma(Y_j)\)$ provides the probability of the $\(j\)-th$ event. Mathematically, the softmax is given by:

$$\sigma(Y)j = \frac{e^{Y_j}}{\sum{i=1}^{N} e^{Y_i}}$$

## Sigmoid vs Relu:

![Sigmoid vs Relu](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/Relu%20vs%20Sigmoid.png)

Sigmoid causes a Vanishing gradient problem.
 Reason- The gradients(slope) near the boundary values( 0 and 1) are almost zero, which results in slower learning i.e weight updation.

The sigmoid is the inverse of the logit function. Thus, the sigmoid of 1.098 is 0.75.
Mathematically, the sigmoid is given by:

$$\sigma(Y) = \frac{1}{1 + e^{-Y}}$$


Relu helps in solving the above problem. 
Reason-  Relu function max(0, Z) . Which means gradients for negative weights are zero and for positive weights the gradients are always 1   to help faster training. I.e weight updation.


## Slope and Gradient:

The term “slope” is generally applicable when only 2 variables are in consideration (height divided by width). The slope then is the actual tangent or the derivative to the curve of the function that connects the 2 variables. It's a measure of the rate of change of a function f(x) with respect to the x .

The gradient is like the slope, except here, the function whose rate of change we wish to study depends on more than one variable, i.e. f = f(x,y) . In other words f is multidimensional.
It would be convenient if we had one operator which would operate on f and give a single output. This is the gradient operator

Notice that the gradient acts on a scalar function but returns a vector value. This is crucial since each component of the gradient indicates the rate of change with respect to that particular dimension.
Thus the slope is the special case of the gradient operator acting on a single dimensional function.


## 3.Cost Function & Loss Function:

Below cost function is used to make the graph convex which is easier to find global optimum.

Loss function is defined for 1 training example.

![Loss function](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/loss%20function.png)


Cost function is defined for the entire training set.

![Cost Function](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/cost%20function.png)


## 4. Gradient Descent: 

To find the global optimum value by updating weights and bias through backpropagation.
The weights and bias will be updated based on the derivative of the cost function.

Weights and bias will be updated with the new weights calculated by subtracting the existing weight with the product of learning rate and derivative of cost function wrt weight and similarly for bias. 

![weights and bais update](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/weights%20and%20bias%20update.png)


![Gradient Descent](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/gradient%20descent.png)



## 6. Formulas for derivatives

![Derivative Formula](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/derivative_formula.png)


![Logistic Regression serivatives](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/logistic%20regression%20derivative.png)


One iteration of Gradient descent is mentioned below.

![forward and backward functions](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/forward%20and%20back%20propagation.png)


![Back Propagation](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/back%20propagation.png)



## 7. Bias/ Variance:

**Bias:** <br>The model fits the training data poorly and produces the similar results in the test data. 
Bias oversimplifies the model because it pays very little attention to training data.
High bias means underfitting the model. 

For example, a linear regression model would have high bias when trying to model a non-linear relationship. It is because the linear regression model does not fit non-linear relationships well.
High bias means linear regression applied to quadratic relationships.
Low bias means second degree polynomials applied to quadratic data.


**Variance:**  <br> The model fits well on training data but can not generalize the pattern well which results in overfitting. It means they don’t fit well on data outside training.


**Bias Variance tradeoff:** <br>
 In the image below, the red ball is the target. Any hit close to it is considered as low bias data points. If each subsequent hit is close to the previous hit is considered as low variance cases.

![Bias varience tradeoff](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/bias%20varience.png)


**How to correct Bias-Variance Error:**


* Try smaller number of Features(only important ones) when you have high variance.
* Try larger number of Features or transform features when you have high bias.
* Get more training data when you have high variance. <br>
In deep learning if there is a hgh bias create a bigger network and if there is high variance then increase the amount of data or perform regularization. 


![Bias Varience example](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/bias%20varience%20example.png)


## 8. SVM:

In the support vector machine (SVM), cost (c) parameter decides bias-variance. A large C gives you low bias and high variance. Low bias because you penalize the cost of misclassification a lot. Large C makes the cost of misclassification high, thus forcing the algorithm to explain the input data stricter and potentially overfit. A small C gives you higher bias and lower variance. Small C makes the cost of misclassification low, thus allowing more of them for the sake of wider "cushion".

## 9. Regularization:

Adding Regularization techniques will avoid variance/overfitting problems.

**L2 Regularizer:**
It penalizes the weights and makes it smaller(near to zero) which ultimately makes Z to fall in the linear region of the activation functions(like tanh).

In backpropagation the parameters(W & B) will be updated based on the regularizer(L1 & L2) added to the previous derivative.
Below is the example for L2 Regularizer.

![L2 Regularizer](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/L2%20regularizer.png)


**Dropout:** <br>
It is a technique where it randomly zeros the neuron from a selected layer based on the dropout probability value(0-1).

In each iteration the dropout neuron will be selected at random which works well on a network.
But it affects gradient descent graph (it may not be convex) since the random neurons are dropped and during backpropagation the parameter adjustment is not constant.

**Data Augmentation:** <br>
Data augmentation will help increase the training set and help the model to reduce variance.

**Early Stopping:** <br>
With early stopping when there is not much improvement in the accuracy in the next subsequent iterations then you can stop training.(Mid size W).

Intuition for early stopping is to see the point where the loss of the training set is decreasing while the loss of the validation set is increasing. We apply early stopping of training at this point, which means the weights will be in the mid range and will not let the weights grow larger to overfit the model.

![Early Stopping](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/early%20stopping.png)



## 10. Normalization:

Normalization is applied on the features so that the model becomes computationally faster to find the global minima or converges faster. 
The unnormalized data will take much longer steps to converge.

![Normalization](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/normalization.png)


## 11. Vanishing/ Exploding gradients:

**Vanishing Gradient:**
Due to very small weights, the first order derivative will become smaller in the subsequent layer and thus causing the derivative to become almost zero in the deep network which will in turn reflect the new parameter(weights or bias) to be almost identical to the old one.

**Exploding Gradient:**
Due to large weights, the first order derivative will be large in number and in the subsequent weight updation process the weights will vary a lot with respect to  the previous one and gradient will never converge to the global minimum. 

![Exploding Gradient](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/exploding%20gradient.png)


## 12. Weights Initialization:

Usually in Tensorflow weights are initialized for dense layer is “glorot_uniform”(kernel initializer parameter) which is a random weights initializer for a given dense layer.

![Weights Init](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/weights%20initialization.png)


## 13. Mini-Batch Gradient Descent:

Splitting training data into multiple batches so that each step of a gradient descent will be performed first on the mini batch rather than the entire training examples.
* It is faster learning
* It uses only limited mini batch memory allocation for the training set rather than allocating memory for the entire training set.
* It shows the convergence towards loss with oscillations for each batch but gives clear idea on the direction it is moving towards after the entire mini batches iteration.

**Stochastic Gradient Descent:**
When the batch size= 1 then it is called stochastic gradient descent, which means for 1000 training examples it will iterate 1000 times and calculate gradient descent every time.

**Disadvantages:**
It is slower since it won’t make use of vectorization for mini-batch.
It never converges, rather it will take a long route towards minima and oscillate near global minima.
	

**Choosing mini-batch size:**
![Batch grad descent](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/batch%20grad%20descent.png)


![Mini Batch size](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/mini%20batch%20size.png)

## 14. Gradient Descent with momentum:

![Gradient Descent with momentum](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/gradient%20descent%20with%20momemntum.png)

![Gradient Descent example](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/gradient%20descent%20example.png)



## 15. RMSProp:

![RMS Prop](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/rms%20prop.png)


## 16. Adam(Adaptive moment estimation):

This is the combination of gradient descent with momentum and RMSprop.

![Adam Optimization](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/adam%20optimization.png)

## 17. Learning Rate Decay:

With the constant learning rate it is most unlikely to converge to a global minimum, rather it just oscillates near a global minimum.
With the use of learning rate decay, learning rate will reduce over iteration and hence Gradient descent will take much smaller steps in the later stages while it reaches a global minimum.
It is usually faster to converge since it takes bigger steps at the start and steps will become smaller at the later stage.


## 18. Local Optima problem:

With more features, all the dimensions aligning in the same direction as a minimum point is highly unlikely. What we face is a saddle point where few of the dimensions align in a particular direction.

![Local Optima](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/local%20optima.png)



Problem of plateaus - due to gradients being close to zero for a long time, GD takes small steps towards minimum.
-Use Adam/ RMSProp/ GD with momentum to speed up training and solve the problem of plateaus.

![Plateaus](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/plateaus.png)

## 19. Hyperparameter tuning:

* Try random values rather than grid search to find parameters.
* We can use the course to find methods to search for parameters. I.e. if we find good results of parameters in a cluster then we zoom in on the cluster and do extensive random search.
* Scale of the parameters should be in range such that resources are utilized equally.
* Use logarithmic scale for hyperparameters like learning rate. 


## 20. Batch Normalization:

Just like normalizing input values we can normalize the Z values coming from a layer which is called batch normalization. Z value is normalized by using the below equation.

![Batch Norm](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/batch%20norm.png)

![Batch Norm Algo](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/batch%20norm%20algo.png)

![Batch Norm Algo 1](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/batch%20norm%20algo%202.png)

## 21. Batch Norm Vs Regularization

![Batch Norm as regularizer](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/batch%20norm%20as%20regularization.png)


## 22. Orthogonalization( Way to fine tune model at different stages):
![Orthogonolization](https://github.com/sachinprasadhs/AI-Concepts/blob/main/Common%20Concepts/Images/orthogonolization.png)


## 23. Evaluation Metrics:

Single number evaluation metrics:
Rather than using separate  evaluation metrics like precision or recall, it is better to use combined metrics to generate a single number evaluation metrics like F1 score or Average. 

$$F1 \text{ score} = \frac{2}{\left(\frac{1}{p} + \frac{1}{r}\right)}$$
 - Harmonic mean
Accuracy vs running time:
	We can consider an optimizing metric and a satisficing metric which can help get a combined metric for evaluation.
Ex :- If we have a cat classifier and we have optimizing metric as Accuracy and running time as Satisficing metric, then we can tune for optimizing the accuracy given the threshold of the running time does not cross 100 ms.
Maximize 	Accuracy 
 Subject to	 running time <= 100ms 





