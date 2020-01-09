from app import app
from flask import render_template, request


@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/post_image', methods=['POST'])
def post_image():
    print('reached here')
    if request.method == 'POST':
        print(request.files)
        return render_template('index.html')