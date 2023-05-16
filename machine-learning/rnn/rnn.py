import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, embedding_size):
        # Define the size of the input, hidden, and output layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size

        # Initialize the weights and biases for the network
        self.Wxh = (
            np.random.randn(hidden_size, embedding_size) * 0.01
        )  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias

        # Embedding matrix
        self.embedding_matrix = np.random.randn(embedding_size, input_size) * 0.01

        # Initialize the hidden state to zero
        self.hs = np.zeros((hidden_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.input_embeddings = []
        self.hidden_states = [np.copy(self.hs)]
        self.outputs = []

        for i in range(len(inputs)):
            # Convert one-hot encoded input to an embedding
            input_embedding = np.dot(self.embedding_matrix, inputs[i].reshape(-1, 1))
            self.input_embeddings.append(input_embedding)

            # Calculate hidden state (using tanh activation function)
            self.hs = np.tanh(
                np.dot(self.Wxh, input_embedding) + np.dot(self.Whh, self.hs) + self.bh
            )
            self.hidden_states.append(np.copy(self.hs))

            # Calculate output (using softmax activation function)
            y = np.dot(self.Why, self.hs) + self.by
            y = np.exp(y) / np.sum(np.exp(y))
            self.outputs.append(y)

        return self.outputs

    def backward(self, doutput, learning_rate=0.1):
        # Store all gradient values
        dWxh, dWhh, dWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhsnext = np.zeros_like(self.hs)

        # Backpropagation through time
        for t in reversed(range(len(self.inputs))):
            dy = np.copy(doutput[t])
            dWhy += np.dot(dy, self.hidden_states[t + 1].T)
            dby += dy
            dhs = np.dot(self.Why.T, dy) + dhsnext
            dhsraw = (1 - self.hidden_states[t + 1] * self.hidden_states[t + 1]) * dhs
            dbh += dhsraw
            dWxh += np.dot(dhsraw, self.input_embeddings[t].T)
            dWhh += np.dot(dhsraw, self.hidden_states[t].T)
            dhsnext = np.dot(self.Whh.T, dhsraw)

        # Clip to mitigate exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update weights and biases using gradient descent
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby


# Define a sequence of 5 inputs (one-hot encoded)
inputs = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
)

# Assume some gradients with respect to outputs
doutput = [np.random.randn(5, 1) for _ in range(len(inputs))]

# Create a SimpleRNN object
rnn = SimpleRNN(input_size=5, hidden_size=4, output_size=5, embedding_size=3)

# Forward pass
outputs = rnn.forward(inputs)
for i, output in enumerate(outputs):
    print(f"Output at time step {i}:", output)

# Backward pass
rnn.backward(doutput)
