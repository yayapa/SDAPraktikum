"""
This is a flask app - connection between frontend and backend.
Start this on the server.
Start from SDAPraktikum folder i.e. above /E2/ directory.
"""
import os.path
import sys
import json

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
UPLOAD_DIR = os.path.join(ROOT_DIR, "E2", "styletransfer", "NeuralNeighborStyleTransfer", "inputs", "content")
DOWNLOAD_DIR = os.path.join(ROOT_DIR, "E2", "styletransfer", "outputs")
from flask import Flask, jsonify, request, render_template, make_response, send_file
import io
from E2.styletransfer.execute_one import execute_one, execute_one_stub, OUTPUT_PATH
from E2.similarity.prediction import predict
from E2.similarity.filters import find_age, find_location, find_hashtags

QUERY = ""
QUERY_TYPE = ""
INPUT_FILENAME = ""
CLUSTERING_QUERY_RESULT=""

app = Flask(__name__)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global INPUT_FILENAME
    if request.method == 'POST':
        fs = request.files.get('snap')
        if fs:
            print('FileStorage:', fs)
            print('filename:', fs.filename)
            INPUT_FILENAME = fs.filename
            upload_path = os.path.join(UPLOAD_DIR, str(fs.filename))
            print("UPLOAD PATH", upload_path)
            fs.save(upload_path)
            return 'Got Snap!'
        else:
            return 'You forgot Snap!'

    return 'Strange behaviour'


@app.route('/download')
def download():
    filename = INPUT_FILENAME+"_styled.jpg"
    filename_path = os.path.join(DOWNLOAD_DIR, filename)
    print("DOWNLOAD STYLE", filename_path)
    return send_file(filename_path, mimetype='image/jpg')


@app.route('/getstyletransfer/<style>', methods=['GET', 'POST'])
def get_style_transfer(style):
    global INPUT_FILENAME
    output_path = execute_one(INPUT_FILENAME, style)
    return output_path



@app.route('/getclustering/<path>', methods=['GET', 'POST'])
def get_clustering(path):
    #print("ingetclustering")
    #print(path)
    path = os.path.join(UPLOAD_DIR, path)
    images = predict(path)
    #images = ["1", "2"]
    if len(images)>20:
        images = images[:19]
    if request.method == 'POST':  # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200
    else:
        print(images)
        return jsonify({'result': images})
        #return render_template("getclustering.html", data=json.dumps(images))
        #return images


@app.route('/getage/<query>', methods=['GET', 'POST'])
def get_age(query):
    query = int(query)
    result = find_age(query)
    if request.method == 'POST':  # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200
    else:  # GET request
        return jsonify({'result': result})


@app.route('/getlocation/<query>', methods=['GET', 'POST'])
def get_location(query):
    result = find_location(query)
    if request.method == 'POST':  # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200
    else:  # GET request
        return jsonify({'result': result})


@app.route('/gethashtags/<query>', methods=['GET', 'POST'])
def get_hashtags(query):
    query = query.split(";")
    result = find_hashtags(query)
    if request.method == 'POST':  # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200
    else:  # GET request
        return jsonify({'result':result})

@app.route('/uploadquery/<query>/<queryType>', methods=['GET', 'POST'])
def upload_query(query, queryType):
    global QUERY
    QUERY = query
    global QUERY_TYPE 
    QUERY_TYPE = queryType
    global CLUSTERING_QUERY_RESULT
    print("UPLOAD QUERY", QUERY, QUERY_TYPE)
    if QUERY_TYPE == 'clustering':
        path = os.path.join(UPLOAD_DIR, INPUT_FILENAME)
        CLUSTERING_QUERY_RESULT = predict(path)
    if request.method == 'POST':  # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200
    else:  # GET request
        return 'OK'

@app.route('/downloadquery', methods=['GET', 'POST'])
def download_query():
    global QUERY
    global QUERY_TYPE
    #print(QUERY, QUERY_TYPE)
    if QUERY_TYPE == 'age':
        query = int(QUERY)
        result = find_age(query)
    elif QUERY_TYPE == 'location':
        result = find_location(QUERY)
    elif QUERY_TYPE == 'hashtags':
        query = QUERY.split(";")
        result = find_hashtags(query)
    elif QUERY_TYPE == 'clustering':
        result = CLUSTERING_QUERY_RESULT
    else:
        result = "error"
    print("DOWNLOAD QUERY", result)
    if request.method == 'POST':  # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200
    else:  # GET request
        return jsonify({'result': result})


@app.route('/downloadpic/<filename>')
def download_pic(filename):
    filename_path = os.path.join(DOWNLOAD_DIR, filename)
    return send_file(filename_path, mimetype='image/jpg')

@app.route('/downloadpic/<filename>')
def download(filename):
    #filename = os.path.join(ROOT_DIR + "E2/styletransfer/outputs", filename)
    #filename_path = "/home/uwgdz/tmp/SDAPraktikum/E2/styletransfer/outputs/" + filename
    filename_path = os.path.join(DOWNLOAD_DIR, filename)
    return send_file(filename_path, mimetype='image/jpg')


@app.route('/getstyletransferanddownload/<style>')
def get_style_transfer_and_download(style):
    global INPUT_FILENAME
    output_path = execute_one(INPUT_FILENAME, style) # execute_one_stub for imitation of functionality of execute_one!!!
    filename = INPUT_FILENAME+"_styled.jpg"
    filename_path = os.path.join(DOWNLOAD_DIR, filename)
    print("DOWNLOAD STYLE", filename_path)
    return send_file(filename_path, mimetype='image/jpg')

if __name__ == "__main__":
    app.run()
