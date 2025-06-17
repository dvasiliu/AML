##  <font color='blue'> Supervised Machine Learning: Linear Regression </font>



### <font color='blue'> Modeling </font>

**In general:** it is the root of all sciences, a comprehensive framework of understanding reality. For many applications you can think of a model as a functional relationship between an input and an output.



<figure>
<center>
<img src='https://i.imgur.com/boVL9B5.png'width='300px'/>
<figcaption>Example of a Function by Arrow Diagram</figcaption></center>
</figure>



The important elements of a function are
> i\. the <font color='magenta'>*domain*</font> of the function
> ii\. the <font color='magenta'>*range*</font> of the function
> iii\. the <font color='magenta'>*definition*</font> of the function.

<font color='purple' size=4px> We believe that when we have a lot of data collected (a lot of examples,) we can train or update the model based on the minimization of the errors.</font>

### <font color='blue'> The Least Squares Method</font>

Carl Friedrich Gauss \(1777\-1855\): was a German mathematician and physicist, widely considered one of the greatest mathematicians of all time. His contributions to a variety of fields have earned him the nickname "Prince of Mathematicians."

<figure>
<center>
<img src='https://i.imgur.com/i1ZpmP3.jpg'width='275px'/>
<figcaption>Portrait of Carl F. Gauss Source: gettyimages</figcaption></center>
</figure>


Had great contributions to mathematics and astronomy\.

He proposed a rule to score the contributions of individual errors to overall error\.

[Reference](https://sites.math.rutgers.edu/~cherlin/History/Papers1999/weiss.html)

**Example:** Assume we have five imprecise measurements, as shown below, what would be the correct answer?



<figure>
<center>
<img src='https://i.imgur.com/lMEP7bj.png'width='600px'/>
<figcaption>Source: "Noise, A Flaw in Human Judgement", D. Kahneman et al.</figcaption></center>
</figure>

<figure>
<center>
<img src='https://i.imgur.com/hAt1PnE.png'width='600px'/>
<figcaption>Source: "Noise, A Flaw in Human Judgement", D. Kahneman et al.</figcaption></center>
</figure>

### <font color='blue'> Noise vs Bias </font>


<figure>
<center>
<img src='https://i.imgur.com/QoN7d6L.png'width='600px'/>
<figcaption>Source: "Noise, A Flaw in Human Judgement", D. Kahneman et al.</figcaption></center>
</figure>


<figure>
<center>
<img src='https://i.imgur.com/Y83uLpf.png'width='400px'/>
<figcaption>Source: "Noise, A Flaw in Human Judgement", D. Kahneman et al.</figcaption></center>
</figure>


### <font color='blue'> The Law of Probable Errors  </font>

The story began when an italian astronomer, Giuseppe Piazzi, discovered a new object in our solar system, the dwarf planet \(asteroid\) Ceres.

[Reference: NASA](https://www.jpl.nasa.gov/news/ceres-keeping-well-guarded-secrets-for-215-years)

Gauss helped relocate the position of Ceres and confirmed the discovery\.

<font color='green'>"*...for it is now clearly shown that the orbit of a heavenly body may be determined quite nearly from good observations embracing only a few days; and this without any hypothetical assumption.*" Carl F. Gauss  </font>

- Small errors are more likely than large errors.

- The likelihood of errors of the same magnitude but different signs, such as $x$ and $-x$, are equal \(the distribution is symmetrical\).

- When several measurements are taken of the same quantity, the average \(arithmetic mean\) is the most likely value.



### <font color='blue'> Introduction to Regression </font>

**Main Idea**:
 - the output (dependent) variable is continuous and we want to "predict" its value within the range of the input features. (<font color='red'>WARNING: doing otherwise could lead to flawed inferences</font>).
 - there is "noise" which means that for essentially the same input values there may be different slightly different values of the output variable or there is "noise" in the measurement of all the variables.  
 - we assume that the noise (i.e. the errors in measurement) are following a normal distribution with mean 0 and some unknown standard deviation.

**Visual Intuition**:

<figure>
<center>
<img src='https://i.imgur.com/djs0rro.png' width='600px' />
<figcaption>The Linear Correlation Concept</figcaption></center>
</figure>

**Seminal Work**:

 The linear correlation coefficient between two variables was introduced by Pearson, Karl (20 June 1895). "Notes on regression and inheritance in the case of two parents". Proceedings of the Royal Society of London. 58: 240–242.)

$$
\large r: = \frac{1}{n-1}\sum\limits_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s_x}\right)\left(\frac{y_i - \bar{y}}{s_y}\right)
$$

Here $\bar{x}$ is the mean of $x$, $\bar{y}$ is the mean of $y$ and, $s_x$ is the standard deviation of $x$ and $s_y$ is the standard deviation of $y.$



**Test statistic** for the linear correlation coefficient:

$$
\large t=\frac{r\sqrt{n-2}}{\sqrt{1-r^2}}
$$

where $n$ is the number of observations and $r$ represents the correlation coefficient computed from the data.

The slope of the regression line is

$$
\large m = r\cdot\frac{s_y}{s_x}
$$


**Theoretical Perspective**:
- we want to estimate the expected value of the dependent variable as a function of the input features. This means that we want to predict the dependent variable *on average* given the input data. Conceptually this translate as an approximation of the conditional expectation $\mathbb{E}(Y|\text{input features})$ such as  

$$
\large\mathbb{E}(Y|X=x) = f(x)
$$

- we want to determine the simplest form of the function $f$ (principle of parsimony) and we assume that

$$
\large Y = f(X) + \sigma \epsilon
$$

where $\epsilon$ is the "noise", in statistical terms, $\epsilon$ is independent and identically distributed, it follows a standard normal distribution and, $\sigma>0$ is generally unknown.

**The Coefficient of Determination**

$$
\large R^2:=1-\frac{\sum (residual_i)^2}{\sum(y_i-\bar{y})^2}
$$



###  <font color='blue'>  The Least Squares Method for Regression</font>

*Visual Intuition*:



<figure>
<center>
<img src='https://i.imgur.com/C6qYyRQ.gif' width='700px'/>
<figcaption>How OLS works.</figcaption></center>
</figure>

### <font color='blue'>Multiple Linear Regression (Linear models with more features)</font>

**Important** The matrix vector product is a linear combination of the columns of the matrix:

$$
\large Xw =w_1\begin{bmatrix}
           x_{11} \\
           x_{21} \\
           \vdots \\
           x_{n1}
         \end{bmatrix}
         +
         w_2\begin{bmatrix}
           x_{11} \\
           x_{21} \\
           \vdots \\
           x_{n1}
         \end{bmatrix}
                  + ...
         w_p\begin{bmatrix}
           x_{1p} \\
           x_{2p} \\
           \vdots \\
           x_{np}
         \end{bmatrix}
$$


where

$$
\large X = \begin{bmatrix}
x_{11}, x_{12}, ... x_{1p} \\
x_{21},x_{22}, ...x_{2p} \\
\vdots \\
x_{n1}, x_{n2}, ... x_{np}
\end{bmatrix}
$$



### <font color='blue'> Examples in different dimensions </font>

<figure>
<center>
<img src='https://i.imgur.com/OI36MLh.png'/>
<figcaption>Linear Models</figcaption></center>
</figure>


<figure>
<center>
<img src='https://i.imgur.com/bGaC2LS.png'width='600x'/>
<figcaption>Linear Models</figcaption></center>
</figure>

### <font color='blue'> Vector Spaces </font>

#### Intuition

* A vector represents a way of encoding information in an organized way that allows also useful operations to be carried out.

* Building intuition: think of position vectors. The addition/subtraction operations can be carried out componentwise. Multiplication by scalars (e.g. real numbers) are also distributed to all of the vector's commponents.



<figure>
<center>
<img src='https://i.imgur.com/hL5eCrq.png'width='300px'/>
<figcaption>Vector Addition</figcaption></center>
</figure>


* Example: an array of prescribed dimensionality can be regarded as a vector.

* More abstract objects, such as continuous or integrable functions can be regarded as vectors, as well.

### Vector Spaces - Formal Definition

<font color='red' size =4px>A vector space $V$ is a set whose elements can be multiplied with scalars and combined among themselves by using a scaler multiplication $'\cdot'$ and a $'+'$ operation, respectively.  These operations mentioned above, must satisfy the following properties:</font>


| **Axiom**                                                                                                                                | **Meaning**                                                                                                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Associativity](https://en.wikipedia.org/wiki/Associativity "Associativity") of vector addition                                          | **u** + (**v** + **w**) = (**u** + **v**) + **w**                                                                                                                                                            |
| [Commutativity](https://en.wikipedia.org/wiki/Commutativity "Commutativity") of vector addition                                          | **u** + **v** = **v** + **u**                                                                                                                                                                                |
| [Identity element](https://en.wikipedia.org/wiki/Identity_element "Identity element") of vector addition                                 | There exists an element **0**∈ _V_, called the _[zero vector](https://en.wikipedia.org/wiki/Zero_vector "Zero vector")_, such that **v** + **0** = **v** for all **v**∈ _V_.                                 |
| [Inverse elements](https://en.wikipedia.org/wiki/Inverse_element "Inverse element") of vector addition                                   | For every **v**∈ _V_, there exists an element −**v** ∈ _V_, called the _[additive inverse](https://en.wikipedia.org/wiki/Additive_inverse "Additive inverse")_ of **v**, such that **v** + (−**v**) = **0**. |
| Compatibility of scalar multiplication with field multiplication                                                                         | _a_(_b_**v**) = (_ab_)**v**<sup>[\[nb 3\]](https://en.wikipedia.org/wiki/Vector_space#cite_note-4)</sup>                                                                                                     |
| Identity element of scalar multiplication                                                                                                | 1**v** = **v**, where 1 denotes the [multiplicative identity](https://en.wikipedia.org/wiki/Multiplicative_identity "Multiplicative identity") in _F_.                                                       |
| [Distributivity](https://en.wikipedia.org/wiki/Distributivity "Distributivity") of scalar multiplication with respect to vector addition | _a_(**u** + **v**) = _a_**u** + _a_**v**                                                                                                                                                                     |
| Distributivity of scalar multiplication with respect to field addition                                                                   | (_a_ + _b_)**v** = _a_**v** + _b_**v**                                      


**Linear Mappings**

Formal Definition: A mapping T between two vector spaces is called linear if and only if

> i\. $T\left(\lambda\vec{u}\right) = \lambda T\left(\vec{u}\right)$

> ii\. $T\left(\vec{u}+\vec{v}\right)=T\left(\vec{u}\right)+T\left(\vec{v}\right)$

for any scaler $\lambda$ and any two vectors $\vec{u}$ and $\vec{v}.$


An example of a linear model with two features is $\hat{y}_i = 1+3x_{i1}+5x_{i2}.$

In this example the value $1$ is referred to as the *intercept*.


If $p$ features in the data and we want to create a linear model, the *input-output* mechanism is
$$
\underbrace{Y}_\text{Output}  = \underbrace{w_1 X_1+w_2 X_2+...+w_p X_p}_\text{Linear combination of features}
$$
This could represented as a matrix-vector product:
$$
\underbrace{Y}_\text{Output}  = \underbrace{X\cdot w}_\text{Linear combination of the columns of matrix X}
$$
In this model the features are $X_1, X_2, ...X_p$ and $w_1, w_2,...w_p$ are a set of weights (real numbers).

The assumption for multiple linear regression is

$$
\large Y= Xw + \sigma \epsilon
$$

where $\sigma$ is the standard deviation of the noise. Further, we assume that the "noise" $\epsilon$ is independent and identically distributed with a zero mean.

We believe that the output is a linear combination of the input features.

Thus, if we would like to solve for the "weights" $w$ we may consider

$$
\large X^tY = X^tXw+\sigma X^t\epsilon
$$
And if the matrix $X^tX$ is invertible then we can solve for expected value of $w$:

$$
\large\mathbb{E}(w) = (X^tX)^{-1}X^t Y
$$
We can show by using *Linear Algebra* that the OLS solution obtained form minimizing the sum of the square residuals is equivalent.


### <font color='blue'> Linear vs Non-linear models </font>

This is a linear model in terms of the weights $w$:

$$
\large\hat{y} = w_0 + w_1x_1 + w_2x_2 - w_3x_3
$$

An example for what linear in weights means

$$
\hat{y}(2w+3\alpha) = 2\hat{y}(w)+3\hat{y}(\alpha)
$$


The following is a non-linear model in terms of the coefficients (weights):
$$
\large\hat{y} = w_0 + w_1^3x_1 + \frac{1}{w_2+w_3}x_2 - e^{w_3}x_3
$$

$$
\hat{y}(2w+3\alpha) \neq 2\hat{y}(w)+3\hat{y}(\alpha)
$$

The main point of linear regression is to assume that predictions can be made by using a linear combination of the features.

For example, if the data from feature $j$ is a column vector, e.g.
$$
\begin{bmatrix}
           x_{1j} \\
           x_{2j} \\
           \vdots \\
           x_{nj}
         \end{bmatrix}
$$
then we assume that the depdendent variable is predicted by a linear combination of these columns populated with features' data. Each column represents a feature and each row an independent observation.

The predicted value is denoted by $\hat{y}$ and
$$
\hat{y} = w_1\begin{bmatrix}
           x_{11} \\
           x_{21} \\
           \vdots \\
           x_{n1}
         \end{bmatrix}
         +
         w_2\begin{bmatrix}
           x_{11} \\
           x_{21} \\
           \vdots \\
           x_{n1}
         \end{bmatrix}
                  + ...
         w_p\begin{bmatrix}
           x_{1p} \\
           x_{2p} \\
           \vdots \\
           x_{np}
         \end{bmatrix}
$$
### We have a vector of weights:
$$
w = \begin{bmatrix}
           w_{1} \\
           w_{2} \\
           \vdots \\
           w_{p}
         \end{bmatrix}
$$
#### <span style="color:green;font-size:14pt"> Critical thinking: what exactly is $\hat{y}$? </span>



<figure>
<center>
<img src='https://i.imgur.com/V67R96W.png'/>
<figcaption>Linear Models</figcaption></center>
</figure>

<font color='magenta'> The matrix-vector product between the feaures and the weights</font>
$$
\hat{y} = X\cdot w
$$
The main idea is that
$$
\hat{y}= \begin{bmatrix}
           \hat{y}_{1} \\
            \hat{y}_{2}  \\
           \vdots \\
             \hat{y}_{n}
         \end{bmatrix}
$$
represents the predictions we make by training (or as we say in ML *learning*) the weights $w.$
Training means running an optimization algorithm and determining the values of the weights that minimize an objective function.

#### <font color='darkgreen'> We want to *learn* the weights $w_1,w_2,...w_p$ that minimize the sum of the squared residuals: </font>
$$
\large\sum\limits_{i=1}^{n}\left(y_i-\sum\limits_{j=1}^{p}X_{i,j}\cdot w_j\right)^2
 = \sum\limits_{i=1}^{n}\left(y_i-X_{i,1}w_1-X_{i,2}w_2 - ...X_{i,p}w_p\right)^2
$$

#### <font color='red'> How do we know we are on the right track after we perform the minimization of the square residuals? </font>

### <font color='blue'>Ordinary Least Squares Regression (OLS) with Gradient Descent</font>

First, we assume the simplest case: data has only one input feature that is continuous.

The main idea of linear regression is the expected value of the output is a linear function of the input variable(s).

$$
\mathbb{E}(Y|X=x)\approx m\cdot x + b
$$

To determine the line of best fit the goal is to minimize the sum of squared residuals:

$$
\min_{m,b} \sum\limits_{i=1}^{n}(y_i-mx_i-b)^2
$$
So the sum of the squared residuals is
$$
\sum\limits_{i=1}^{n}(y_i-mx_i-b)^2
$$
If $N$ represents the number of observations (the number of rows in the data) then the cost function may be defined
$$
L(m,b) = \frac{1}{N} \sum\limits_{i=1}^{N}(y_i-mx_i-b)^2
$$
where
$$
\hat{y_i} = m\cdot x_i +b.
$$

If we get our predictions $\hat{y}_i$ then we have that the Mean Squared Error is
$$
\frac{1}{N} \sum\limits_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

<font color='forestgreen'> Critical Thinking: at the optimal values $(m,b)$ the partial derivatives of the cost function $L$ are equal to 0.</font>

The <font color='deepskyblue'>*gradient descent algorithm*</font> is based on this idea.

Thus, the equation of the best fit line is 

$$
y = mx + b.
$$

<font color='red'> CRITICAL THINKING: How *exactly* are we obtaining the slope and the intercept?</font>

<font color='forestgreen'>ANSWER: One way to obtain the slope and the intercept is by applying the *Ordinary Least Squares* method.</font>

We determine the values of <font color='blue'>$m$</font> and <font color='red'>$b$</font> such that the sum of the square distances between the points and the line is *minimal*.

An example of a loss function with two coefficients is

$$
L(w_1,w_2): = \frac{1}{n}\sum (y_i - w_1\cdot x_{i1} - w_2\cdot x_{i2})^2
$$

Here we have a vector 
$$
\vec{w}:=(w_1,w_2)
$$

We can think of the vector $\vec{w}$ having (in this case) two components and a perturbation of $\vec{w}$ in some direction such as $\vec{v}.$

We consider the function $g(t):=L(\vec{w}+t\cdot \vec{v})$ we get some important ideas:

i. if $\vec{w}$ is ideal for the cost then $t=0$ is min of the function $g$ and so $g'(0)=0.$

ii. if $\vec{w}$ is not minimizing the cost then we want to decrease the function $g$

Here we have that 
$$
g'(t):= \nabla L \cdot \vec{v}
$$
should be negative (because we want to decrease the output of $g$) and we'll improve the outcome if

$$
\vec{v}:= -\nabla L
$$

This means that the coefficients should be updated in the negative direction of the gradient, such as:

$$
\vec{w}_{new} := \vec{w}_{old} - \text{lr}\cdot\nabla L
$$

where "lr" stands for "learning rate."

<figure>
<center>
<img src='https://i.imgur.com/KzUuU3q.gif'width='650px'/>
<figcaption>Gradient Descent Animation</figcaption></center>
</figure>

<figure>
<center>
<img src='https://i.imgur.com/6S1weRY.gif'
width='600px' />
<figcaption>Source: Simple Linear Regression (Tobias Roeschl)</figcaption></center>
</figure>

### <font color='blue'> Polynomial Regression</font>

####  1. Polynomials in one variable

<font color='slateblue'> Main idea: Linear combination of different powers of the feature values. </font>

$$
\large P(x):= \beta_px^p+\beta_{p-1}x^{p-1}+...+\beta_1x+\beta_0
$$

What we hope to achieve:

$$
\large \mathbb{E}(Y|X=x)\approx \beta_p x^p+\beta_{p-1}x^{p-1}+...+\beta_1x+\beta_0
$$

IMPORTANT: P(x) is nonlinear in x. However if x is fixed (x is your data) and $\beta$ is the input we have
$$
\large L(\beta):= \beta_px^p+\beta_{p-1}x^{p-1}+...+\beta_1x+\beta_0
$$

is linear in $\beta=(\beta_0,\beta_1,\beta_2,...\beta_p).$

$$
\large L(\beta+\gamma)= L(\beta)+L(\gamma)
$$

and

$$
\large L(c\cdot \beta) = c\cdot L(\beta)
$$

for any two vectors $\beta$ and $\gamma$, and any scalar (real number) $c.$

Example with one independent variable:

<figure>
<center>
<img src='https://i.imgur.com/JvLP3j0.png'
width='500px' />
<figcaption>Example of Polynomial Regression with One Independent Variable</figcaption></center>
</figure>

**Critical Thinking**: How does a cubic polynomial in two variables look like?

### <font color='blue'> Support Vector Regression </font>


Support Vector Machines Seminal Paper (1992): http://www.svms.org/training/BOGV92.pdf


Support Vectors is a method used in Machine Learning for both regression and classification problmes. The main idea is to map the input features into a higher dimensional space and then, in that higher dimensional space, address the problem to solve.

For regression, SVM consists of an algorithm that solves a quadratic optimization problem with constraints:

$$
\text{minimize}\frac{1}{2}\|w\|^2
$$

subject to

$$
\left\{
    \begin{array}{l}
        y_i -wx_i - b\leq\epsilon &\\
        wx_i+b-y_i \leq \epsilon
    \end{array}
\right.
$$

We can express the constraints as:

$$
\large |y_i - (wx_i+b)|\leq\epsilon
$$

Intuitively we have

<figure>
<center>
<img src='https://i.imgur.com/1dJCvHw.png'
width='500px' />
<figcaption>Main Idea for SVR in 1-D</figcaption></center>
</figure>


#### Slack Variables

SVR with slack variables consists of an algorithm that solves a quadratic optimization problem with constraints:

$$
\text{minimize}\frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}|\xi_i|
$$

subject to

$$
|y_i - x_i\cdot w - b|\leq\epsilon + |\xi_i|
$$

where $x_i = (x_{i1},x_{i2},...x_{ip})$ represents the ith observation that has $p$ features and $w$ is the vector of the weights. We have $1\times p \cdot p\times 1 = 1\times 1$ The main idea is that the slack variables will accommodate points that are "close" to the epsilon margins and that may influence the influence the value of the weights.

<font color='green'> This means that we have at least two different hyperparameters in this case such as $\epsilon$ and $C$.</font>

### <font color='blue'> Diagnostics for Regression </font>



#### <font color='blue'> MSE </font>

$$
\text{MSE}:=\frac{1}{n}\sum_{i=1}^{n}(y_i-x_i\cdot\vec{w})^2
$$

here the i-th observation has multiple features:

$$
x_i = \left(x_{i1},x_{i2},...x_{ip}\right)
$$

where the "dot" product is defined as

$$
x_i\cdot\vec{w} = \sum_{j=1}^{p} x_{ij}w_j
$$

####  <font color='blue'> RMSE </font>

Root mean squared error:

$$
\text{RMSE}:=\left(\frac{1}{n}\sum_{i=1}^{n}(y_i-x_i\cdot\vec{w})^2\right)^{1/2}
$$

#### <font color='blue'> MAE </font>

Mean absolute error:

$$
\text{MAE}:=\frac{1}{n}\sum_{i=1}^{n}\left|y_i-x_i\cdot\vec{w}\right|
$$


#### <font color='blue'> The Coefficient of Determination </font>



<figure>
<center>
<img src='https://i.imgur.com/K39C2NC.png'/>
<figcaption>Linear Models</figcaption></center>
</figure>

$$
\large R^2:=1-\frac{\sum (residual_i)^2}{\sum(y_i-\bar{y})^2}
$$

We know we make a good job when R2 is very close to 1. We make a very poor job if R2 is close to 0 or even negative.

### <font color='blue' size=5pt> Tests for Normality </font>

We believe that, if the residuals are normally distributed, then the average of the errors is a meaningful estimator for the model's performance.

####  <font color='blue'> What is a Normality Test? </font>

Assume we have a univariate set of values ( we have a sample from a univariate distribution.) We are checking, based on some calculations with the sample data, if the univariate distribution is a normal one.

**Main Idea**: We are measuring the nonlinear correlation between the empirical density function of the data vs the theoretical density of a standard normal distribution.

We want to recast this matching procedure onto the backdrop of a linear correlation situation; this means we want to compare the two cumulative distribution functions. To explain, we want the empirical percentiles to correlate linearly with the theoretical percentiles of a standard normal distribution.

### <font color='blue'> The Kolmogorov-Smirnov test </font>

The Kolmogorov-Smirnov test uses the concept of cummulative distribution functions:

$$
CDF(x):= P(X<x) = \int_{-∞}^{x}f(t)dt
$$

The concept is useful in applications where we want to check if a random variable follows a certain distribution.

IMPORTANT: In most cases we standardize the values of the random variable, e.g we compute z-scores.

The test is defined as:

  **H0 (the null hypothesis):**	The data follow a specified distribution.

  **H1 (the alternative hypothesis):** The data do not follow the specified distribution

The main idea is that we focus on how much the empirical cummulative distribution function is different from the theoretical cummulative distribution function, and we may consider:

$$
\sup_{x} |ECDF(x) - CDF(x)|
$$

where $\text{ECDF}(x)$ means the emprirical cummulative distribution function:

$$
ECDF(x):= \frac{1}{n}\sum \mathbb{1}(t)_{t<x}
$$

and, $CDF$ stands for the cummulative distribution function:

$$
CDF(x):= \int_{-\infty}^{x}f(t)dt.
$$

Here $f$ is the probability density function.

If we order the observations, such as $x_i\leq x_j$ when $i\leq j$, then the test statistic is formally defined by:

$$
D:=\max_{1\leq i\leq n}\left\{CDF(x_i)-\frac{i-1}{n},\frac{i}{n}-CDF(x_i)\right\}
$$

The mathematical notation means that we add $1$ for each $t$ less than $x$ and $n$ represents the sample size.

If the p-value is high (much greater then 5%) we do not reject the null hypothesis which means that the normality assumption is not violated.

### <font color='blue'> The Anderson-Darling Test</font>

The test is defined as:

  **H0 (the null hypothesis)**:	The data follow a specified distribution.

  **H1 (the alternative hypothesis)**: The data do not follow the specified distribution

  The test statistic is defined as:

$$
\large AD := -n - \sum_{i=1}^{n} \frac{2i-1}{n}\left[\ln(CDF(x_i))+\ln(1-CDF(x_{n+1-i})\right]
$$

The critical values for the Anderson-Darling test are dependent on the specific distribution that is being tested.

### <font color='blue'> When Linear Regression Fails </font>

In this context *fails* means that OLS is unable to provide a unique solution, such as a unique set of coefficients for the input variables.

Plain *vanilla* multiple linear regression (OLS) fails if the number of observations is smaller than the number of features.


**Example:** If the dependent variable is the Sales Price, we cannot uniquely determine the weights for the features if we have only 4 observations.


| Dist. to School | Prop. Area | Housing Area | Value | Prop. Tax | Bathrooms | Sales Price |
|----------------:|-----------:|-------------:|------:|----------:|----------:|------------:|
|               7 |        0.4 |         1800 |   234 |       9.8 |         2 |       267.5 |
|             2.3 |        0.8 |         1980 |   244 |      10.5 |       2.5 |       278.2 |
|             4.3 |        1.1 |         2120 |   252 |      16.2 |         3 |       284.5 |
|             3.8 |        0.6 |         2500 |   280 |      18.4 |       3.5 |       310.4 |


Suppose we want to predict the "Sales Price" by using a <font color='red'>linear combination</font> of the feature variables, such as "Distance to School", "Property Area", etc.

$$
\large M\cdot \vec{\beta} = \beta_1\cdot col_1(M)+\beta_2\cdot col_2(M)+...+\beta_p\cdot col_p(M)
$$

The Ordinary Least Squares (OLS) method aims at finding the coefficients $\beta$ such that the the sum or squared errors is minimal, i.e.

$$
\large \underset{\beta}{\operatorname{argmin}}\|\text{Sales Price}- M\cdot \vec{\beta}\|^2
$$

**Important Question**: Why does OLS fail in this case? Hint: the problem to solve is ill-posed, in the sense that it allows many perfect solutions.

### <font color= 'blue'> What does Rank Deficiency mean, and why we need Regularization?</font>

The assumption for multiple linear regression is

$$
\large Y = X\beta + \sigma \epsilon
$$

where $\sigma$ is the standard deviation of the noise. Further, we assume that the "noise" $\epsilon$ is independent and identically distributed with a zero mean.

We believe that the output is a linear combination of the input features.

Thus, if we would like to solve for the "weights" $\beta$ we may consider

$$
\large X^tY = X^tX\beta+\sigma X^t\epsilon
$$

And if the matrix $X^tX$ is invertible then we can solve for expected value of $\beta$:

$$
\large \mathbb{E}(\beta) = (X^tX)^{-1}X^t Y
$$

We can show by using *Linear Algebra* that the OLS solution obtained form minimizing the sum of the square residuals is equivalent.

We can test whether the matrix $X^tX$ is invertible by simply computing its determinant and checking that it is not zero.

### IMPORTANT: When the matrix $X^tX$ is not invertible we cannot apply this method to get $\mathbb{E}(\beta)$. In this case if we minimize the sum of the squared residuals the algorithm cannot find just *one* best solution.


###  <font color='blue'>A solution for rank defficient Multiple Linear Regression: L2 (Ridge) Regularization</font>

#### <font color='red'> Main Idea: minimize the sum of the square residuals plus a constraint on the vector of weights</font>

The L2 norm is

$$
\|\beta\|_2:=\left(\sum_{j=1}^{p}\beta_j^2\right)^{1/2}
$$

The Ridge model (also known as the *Tikhonov regularization*) consists of *learning* the weights by the following optimization:

$$
\text{minimize} \frac{1}{n}\sum_{i=1}^{n}\left(\text{Residual}_i\right)^2 + \alpha \sum\limits_{j=1}^{p}\beta_j^2
$$

where $\alpha$ is a constant that can be adjusted based on a feedback loop so it is a hyper-parameter ("tunning" parameter).

This optimization is equivalent to minimizing the sum of the square residuals with a constraint on the sum of the squared weights

$$
\text{minimize} \frac{1}{n}\sum_{i=1}^{n}\left(\text{Residual}_i\right)^2
$$

subject to

$$
\sum\limits_{j=1}^{p}\beta_j^2 < M
$$

**Important**: What happens with the solution $\beta$ when the hyperparameter $\alpha$ grows arbitrarily large? Hint: generate a mapping and plot the coefficient paths.

### <font color= 'blue'> L1 (Lasso) Regularization</font>

The L1 norm is

$$
\|\beta\|_1:=\sum_{j=1}^{p}|\beta_j|
$$

The Lasso model  consists of *learning* the weights by the following optimization:

$$
\text{minimize} \frac{1}{n}\sum_{i=1}^{n}\left(\text{Residual}_i\right)^2 + \alpha \|\beta\|_1
$$

where $alpha$ is a constant that can be adjusted based on a feedback loop so it is a hyperparameter.

This optimization is equivalent to minimizing the sum of the square residuals with a constraint on the sum of the squared weights

$$
\text{minimize} \frac{1}{n}\sum_{i=1}^{n}\left(\text{Residual}_i\right)^2
$$

subject to

$$
\sum\limits_{j=1}^{p}|\beta_j| < M
$$

### <font color='blue'> The difference between L1 and L2 norms </font>

In the following example the L2 norm of the vector $\vec{AB}$ is 5 and the L1 norm is $4+3=7$.

<figure>
<center>
<img src='https://i.imgur.com/J8Mda3S.png'
width='500px' />
<figcaption>The difference between the L1 and L2 norms</figcaption></center>
</figure>


### <font color='blue'> A Geometric Comparison between Lasso and Ridge Regularizations </font>

<figure>
<center>
<img src='https://i.imgur.com/e9EEJL7.png'
width='700px' />
<figcaption>Difference between Lasso and Ridge</figcaption></center>
</figure>

### <font color= 'blue'> Elastic Net Regularization </font>

The main idea is to combine the L2 and L1 regularizations in a *weighted* way, such as:

$$
\lambda\cdot \sum\limits_{j=1}^{p}|\beta_j| + 0.5\cdot (1-\lambda)\cdot\sum\limits_{j=1}^{p}\beta_j^2
$$

Here, for $0\leq\lambda\leq1$, the term $\lambda$  is called the L1_ratio.

The Elastic Net regularization consists of *learning* the weights by solving the following optimization problem:

$$
\text{minimize} \frac{1}{2n}\sum_{i=1}^{n}\left(\text{Residual}_i\right)^2 + \alpha\left( \lambda\cdot \sum\limits_{j=1}^{p}|\beta_j| + 0.5\cdot (1-\lambda)\cdot\sum\limits_{j=1}^{p}\beta_j^2\right)
$$

So, for this regularization we have two hyper-parameters that we need to decide on. **Think:** how can we determine the best choice of hyper-parameters?

### <font color= 'blue'> Model Validation via k-Fold Cross-Validations</font>

In general: how do we know that the predictions made are good?

For many applications you can think of a data set representing both “present” and “future.”

<figure>
<center>
<img src="https://i.imgur.com/sWRVXjk.png" width='550px' title="Train-Test Split" />
</center>
</figure>



In order to compare the predictive power of different models we use K-fold cross-validation.

How do we get unbiased estimates of errors?

<figure>
<center>
<img src="https://i.imgur.com/LCNRuF3.png" width='550px' title="CV" />
</center>
</figure>

Example schematic of 5-fold cross-validation:


<figure>
<center>
<img src='https://i.imgur.com/SzeE4st.png'
width='550px' />
<figcaption>Step 1 in the 5-fold cross-validation</figcaption></center>
</figure>
