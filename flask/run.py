from app import app
import socket

address = socket.gethostbyname(socket.gethostname())
serve_lan = False

if __name__ == "__main__":
    if serve_lan:
        app.run(host=address, port=5000, debug=False)
    else:    
        app.run(host='127.0.0.1', port=5000, debug=True)