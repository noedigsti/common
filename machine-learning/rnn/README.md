1. **Sequential Data Processing:** RNNs are designed for processing data with a sequential nature, such as time series or language, by capturing dependencies and patterns.

2. **Recurrent Connections:** RNNs have connections that allow information to persist across time steps, enabling the network to maintain memory of past inputs.

3. **Time Unfolding:** RNNs can be conceptually "unfolded" over time, enabling independent processing at each time step and facilitating training with backpropagation.

4. **Vanishing/Exploding Gradient:** RNNs can suffer from the vanishing or exploding gradient problem, limiting their ability to learn long-term dependencies. Techniques like LSTM and GRU help address this issue.

5. **LSTM and GRU:** LSTM and GRU are popular variants of RNNs that use gating mechanisms to better capture long-term dependencies.

6. **Bidirectional RNNs:** Bidirectional RNNs process sequences in both forward and backward directions, incorporating past and future context for predictions.

7. **Applications:** RNNs have been successfully used in language modeling, machine translation, sentiment analysis, speech recognition, and more.

8. **Training:** RNNs are trained using backpropagation through time (BPTT) with optimization algorithms like SGD.

9. **Sequence Length Handling:** Techniques such as mini-batching, truncation, and padding are used to handle sequences of varying lengths.

10. **Deep and Stacked RNNs:** Deep RNNs with multiple layers and stacked RNNs allow for increased modeling capacity and hierarchical feature extraction.

### REFERENCES

1. https://youtu.be/ySEx_Bqxvvo?t=725
