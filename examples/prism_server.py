from flask import Flask, request

app = Flask(__name__)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.json
    print(f"Received from bot: {data}")
    return 'Data received', 200

if __name__ == '__main__':
    app.run(port=5000)
