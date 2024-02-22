from flask import Flask
import subprocess

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':

    subprocess.run(['../frp_server/frp_0.54.0_linux_amd64/frpc', '-c', '../frp_server/frpc/frpc.toml'])
    app.run(port=7860)