from port_checker import check_port, check_curl_response

port = 11434

# Check if the port is open
if check_port(port):
    print(f"Port {port} is open.")
    # Check the curl response
    success, response = check_curl_response(port)
    if success:
        print(f"Curl response: {response}")
    else:
        print(f"Curl error: {response}")
else:
    print(f"Port {port} is not open.")

