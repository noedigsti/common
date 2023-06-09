`Perceptron`: Basic artificial neuron used in machine learning. Learns from input data, computes weighted sum, applies activation function, and adjusts weights to improve predictions. Forms building block for neural networks.

`Linear regression`: Statistical technique to model the relationship between variables by finding the best-fitting line. Estimates coefficients that minimize the difference between predicted and actual values. Used for prediction and analysis.

`Gradient descent`: Optimization algorithm for minimizing a function by iteratively adjusting parameters in the direction of steepest descent. Used in machine learning and neural network training.

`Logistic regression`: is a _binary classification_ algorithm that models the probability of belonging to a class based on input features using a sigmoid function. It finds the optimal weights through training and assigns class labels based on a threshold.

`Locally weighted regression (LWR)`: is a non-parametric regression algorithm that predicts by considering nearby data points with varying weights based on their proximity. It adapts to local data characteristics and captures complex relationships.

`Stochastic`: means using random mini-batches of training examples for computing gradients and updating parameters, making it more efficient but introducing some randomness and noise into the process.

`Stochastic gradient descent (SGD)`: Optimization algorithm that computes gradients on randomly selected mini-batches of data, allowing for efficient and faster convergence in machine learning models.

`Affine transformation`: is a geometric mapping that preserves parallel lines and ratios of distances through a combination of scaling, rotation, shearing, and translation operations.

`Gaussian Discriminant Analysis (GDA)`: is a statistical classification technique that models class distributions as Gaussians to make predictions.

`Gaussian` or `Normal Distribution`: refers to the bell-shaped probability distribution that is commonly used to model real-world data.

`Memoization`: improves function performance by caching computed results for specific inputs, allowing subsequent calls with the same inputs to retrieve the result from the cache instead of recomputing it. This reduces redundant calculations and enhances overall efficiency.

`argmax`: is a function that finds the input value that maximizes a given expression or function.

`Bayes' theorem`: is a formula that updates the probability of an event based on new information. It involves the prior probability, likelihood, and evidence to calculate the revised probability.

`Softmax function`: transforms a vector into a probability distribution by exponentiating and normalizing its elements, commonly used for multi-class classification as it assigns higher probabilities to the elements with larger values, making it easier to interpret the output as class probabilities.

`Linear algebra`: deals with vectors, matrices, and operations like addition, subtraction, and multiplication. It is used to solve linear equations and has applications in various fields.

`Entropy`: is a measure of uncertainty or randomness in data or probability distributions. It quantifies the amount of information or surprise in a random variable. High entropy means more uncertainty, low entropy means less uncertainty.

`Cross-entropy Loss` or `Negative Log-likelihood Loss`: is a loss function that compares predicted probabilities with true probabilities, quantifying the dissimilarity between them. It calculates the _average_/_mean_ information required to describe the true distribution using the predicted distribution. Minimizing cross-entropy improves the accuracy of predictions in classification tasks.

`Kullback-Leibler (KL) divergence`: is a measure of how one probability distribution differs from another. It is the average difference between the log-likelihood of the first distribution and the log-likelihood of the second distribution. KL divergence is always non-negative and zero only when the two distributions are identical.

An `episode` is a single sequence of interactions between an agent and the environment, representing a complete trajectory from the initial state to a terminal state.

`Policy`: is a function that maps states to actions. It is the agent's behavior, and it is what the agent uses to decide what action to take in a given state.

`Markov Decision Process (MDP)`: is a mathematical framework for modeling decision-making problems where outcomes are _partly random_ and _partly under the control_ of the decision-maker. It is a discrete-time stochastic control process with discrete states and actions, and the assumption that _the next state and reward only depend on the current state and action_.

`Online Learning`: can be used to train systems on huge datasets that cannot fit in one machine's main memory (out-of-core learning). It is also useful for training on data that is continuously generated at a fast rate (real-time learning). It is also known as incremental learning. If bad data is fed to the system, it will learn from it and its performance will degrade.