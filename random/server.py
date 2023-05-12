import socket
import time
import threading
import torch
import torch.nn as nn
import queue


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


def run_model_and_print_output(input_queue, continue_running):
    model = SimpleRNN(input_size=2, hidden_size=20, output_size=1)
    hidden = model.init_hidden(batch_size=1)
    last_input_time = time.time()

    while continue_running.is_set():
        user_input = input_queue.get()  # get input

        if user_input.isdigit():
            current_time = time.time()
            time_elapsed = current_time - last_input_time

            while continue_running.is_set() and input_queue.empty():
                current_time = time.time()
                time_elapsed = current_time - last_input_time
                input_tensor = torch.tensor([[[float(user_input), -time_elapsed]]])
                output, hidden = model(input_tensor, hidden)
                print(f"Input: {user_input} - Model output: {output.item():.4f}")
                last_input_time = current_time
                time.sleep(1 / 20)  # Refresh rate

        elif user_input == "???":
            print("???")
            hidden = model.init_hidden(batch_size=1)
        elif user_input == "stop":
            print("Stopping server...")
            continue_running.clear()


def receive_input(c, input_queue, continue_running):
    while continue_running.is_set():
        user_input = c.recv(1024).decode()
        if not user_input:
            break
        elif user_input == "stop":
            print("Stopping server...")
            continue_running.clear()
            break
        elif user_input.isdigit():
            input_queue.put(user_input)
        else:
            print(f"Received input {user_input}")
            input_queue.put("???")
    c.close()


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 6666
    s.bind(("", port))
    s.listen(5)

    # Set a timeout for the accept() method
    s.settimeout(1)

    input_queue = queue.Queue()
    continue_running = threading.Event()
    continue_running.set()

    while continue_running.isSet():
        try:
            c, addr = s.accept()
            t1 = threading.Thread(
                target=receive_input, args=(c, input_queue, continue_running)
            )
            t2 = threading.Thread(
                target=run_model_and_print_output, args=(input_queue, continue_running)
            )

            t1.start()
            t2.start()

            # Waiting for the threads to finish
            t1.join()
            t2.join()
        except socket.timeout:
            # If a timeout occurs, check if the server should still be running
            if not continue_running.isSet():
                print("Server stopped due to timeout!")
                break
        except OSError:
            print("Server stopped due to an error!")
            break

    # Closing the server socket
    s.close()

    print("Connection closed!")


if __name__ == "__main__":
    main()
