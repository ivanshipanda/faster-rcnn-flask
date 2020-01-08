from app import app
from flask import render_template, request


@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/post_image', methods=['POST'])
def post_image():
    if request.method == 'POST':
        return render_template('index.html')