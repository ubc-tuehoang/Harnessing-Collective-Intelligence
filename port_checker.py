import socket
import subprocess

def check_port(port):
    try:
        # Create a socket object
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set a timeout for the connection attempt
            sock.settimeout(5)
            # Try to connect to the port
            result = sock.connect_ex(('localhost', port))
            # Return True if the port is open, False otherwise
            return result == 0
    except Exception as e:
        print(f"Error checking port: {e}")
        return False

def check_curl_response(port):
    try:
        # Run the curl command and capture the output
        result = subprocess.run(['curl', f'localhost:{port}'], capture_output=True, text=True)
        # Check if curl executed successfully
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        print(f"Error executing curl: {e}")
        return False, str(e)


