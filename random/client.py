import socket


def main():
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Define the port on which you want to connect
    PORT = 6666

    # Connect to the server (in this case, we're connecting to the same machine)
    s.connect(("127.0.0.1", PORT))

    while True:
        # Get user input
        user_input = input("Enter a number: ")

        # Send user input to Script B
        s.send(user_input.encode())

        if user_input == "stop":  # If user input is the exact string "STOP"
            print("Stopping...")
            s.send("stop".encode())
            break

    # Close the connection
    s.close()

    print("Connection closed!")


if __name__ == "__main__":
    main()
