from flask import Flask, request, redirect
import re

app = Flask(__name__)

@app.route('/some/path')
def safe():
    target = request.args.get('target', '')
    if rv.match(target):
        return redirect(target)
    else:
        return redirect("/")
    
# define a regex variable "rv" which will check if a URL is for the example.com domain
rv =