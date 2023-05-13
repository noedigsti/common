from collections import deque
import concurrent.futures
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


def run_model_and_print_output(model, hidden, input_queue, continue_running):
    try:
        last_input_time = time.time()

        while continue_running.is_set():
            if not input_queue.empty():
                user_input = input_queue.get()

                if user_input.isdigit():
                    current_time = time.time()
                    time_elapsed = current_time - last_input_time

                    while continue_running.is_set() and input_queue.empty():
                        current_time = time.time()
                        time_elapsed = current_time - last_input_time
                        input_tensor = torch.tensor(
                            [[[float(user_input), -time_elapsed]]]
                        )
                        output, hidden = model(input_tensor, hidden)
                        print(
                            f"Input: {user_input} - Model output: {output.item():.4f}"
                        )
                        last_input_time = current_time
                        time.sleep(1 / 20)  # Refresh rate

                elif user_input == "???":
                    print("???")
                    hidden = model.init_hidden(batch_size=1)
                elif user_input == "stop":
                    print("Stopping server...")
                    continue_running.clear()
    except Exception as e:
        print(f"Exception in run_model_and_print_output: {e}")


def receive_input(c, input_queue, continue_running, active_connections):
    try:
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
        active_connections.remove(c)
    except Exception as e:
        print(f"Exception in receive_input: {e}")
    finally:
        c.close()
        active_connections.remove(c)


def handle_client(c, input_queue, continue_running, active_connections):
    active_connections.add(c)
    is_active = True  # Flag to track the active state of the client connection
    c.settimeout(1)  # Set timeout for client's socket object

    try:
        while continue_running.is_set() and is_active:
            try:
                user_input = c.recv(1024).decode()
            except socket.timeout:
                continue

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
    except Exception as e:
        print(f"Exception in handle_client: {e}")
    finally:
        c.close()
        active_connections.remove(c)
        is_active = (
            False  # Set the active state to False when the client connection is closed
        )


def main():
    port = 6666
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", port))
        s.listen(5)

        input_queue = queue.Queue()
        continue_running = threading.Event()
        continue_running.set()

        active_connections = set()

        model = SimpleRNN(input_size=2, hidden_size=20, output_size=1)
        hidden = model.init_hidden(batch_size=1)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        ) as executor:  # 3 clients
            model_future = executor.submit(
                run_model_and_print_output, model, hidden, input_queue, continue_running
            )

            s.settimeout(2)  # Set timeout
            while True:
                try:
                    c, addr = s.accept()
                    active_connections.add(c)
                    executor.submit(
                        handle_client,
                        c,
                        input_queue,
                        continue_running,
                        active_connections,
                    )

                except socket.timeout:
                    # If a timeout occurs, check if the server should still be running
                    if not continue_running.isSet() and len(active_connections) == 0:
                        print("Server stopped due to timeout!")
                        break
                except OSError:
                    print("Server stopped due to an error!")
                    break

            # Close all active connections
            for c in active_connections:
                c.close()

            # If the function raised an exception, .result() will re-raise that exception here
            model_future.result()

        print("Server closed!")


if __name__ == "__main__":
    main()
