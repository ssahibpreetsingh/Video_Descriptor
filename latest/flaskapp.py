from flask import Flask, render_template, request
import os
from end import final_summary




app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload():

    if 'file' not in request.files:
        return 'No file found'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file.save('uploads/' + file.filename)
    
    
    return final_summary('uploads/'+file.filename)


if __name__ == '__main__':
    app.run(debug=True)
