
import numpy as np 
import cv2
import sys
import time
import os
import zipfile
import shutil
from PIL import Image 
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug import secure_filename

UPLOAD_FOLDER_DATASET = 'pictures/'
UPLOAD_FOLDER_SELFIE = 'person/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_path(p):
	folder = os.path.join(p)
	for files in os.listdir(folder):
		path = os.path.join(folder, files)
		try:
			if os.path.isfile(path):
				os.unlink(path)
		except Exception as e:
			print(e)

def zip_create(outputfolder):
	shutil.make_archive('output', 'zip', outputfolder)

	saved = outputfolder + '.zip'
	return 'Succes! Your pictures are saved at ' + saved

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/sorter/')
def sorter():
	return render_template('sorter.html')

@app.route('/sorter/dataset/', methods = ['GET', 'POST'])
def upload_dataset():
	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_DATASET
	clear_path(app.config['UPLOAD_FOLDER'])

	#Get new images and save to folder
	if request.method == 'POST':
		if 'dataset' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files.getlist('dataset')
		print(len(file))
		for f in file:
			if f and allowed_file(f.filename):
				filename = secure_filename(f.filename)
				f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

	return render_template('sorter.html')

@app.route('/sorter/selfie/', methods = ['GET', 'POST'])
def upload_selfie():
	app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_SELFIE
	clear_path(app.config['UPLOAD_FOLDER'])

	if request.method == 'POST':
		if 'selfie' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['selfie']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], "person.png"))

	return render_template('sorter.html')

@app.route('/sorter/name/', methods = ['GET', 'POST'])
def save_name():
	if request.method == 'POST':
		try:
			n = request.form['name']
			print("Welcome: ", n)
		except:
			print("fail")
	return render_template('sorter.html')

@app.route('/sorter/algorithm/')
def detectFaces():
	cascPath = path+r'\classifiers\haarcascade_frontalface_default.xml'
	dataPath = path+r'\pictures'
	person = path+r'\person\person.png'
	outputPath = path+r'\output'

	faceCascade = cv2.CascadeClassifier(cascPath)
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read(ymlTrainer(person))

	image_paths = [os.path.join(dataPath, f) for f in os.listdir(dataPath)]
	images = []

	for imag in image_paths:
		image_pil = Image.open(imag).convert('L')
		image = np.array(image_pil, 'uint8')
		faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

		for (x, y, w, h) in faces:
			nbr_predicted, conf = recognizer.predict(image[y:y+h,x:x+w])
			if nbr_predicted==1 and conf<60:
				images.append(str(imag))
			else:
				break

		#os.unlink(imag)
	#os.unlink(person)

	for ims in images:
		shutil.copyfile(ims, os.path.join(outputPath, os.path.basename(ims)))

	return	zip_create(outputPath)


def ymlTrainer(per):
	name = 1
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	cascPath = path+r'\classifiers\haarcascade_frontalface_default.xml'
	ymlLoc = path+r'\trainer\trainer.yml'
	faceCascade = cv2.CascadeClassifier(cascPath)

	mijnheer = []
	image_pil = Image.open(per).convert('L')
	image = np.array(image_pil, 'uint8')
	faces = faceCascade.detectMultiScale(image)

	for (x, y, w, h) in faces:
		mijnheer.append(image[y: y + h, x: x + w])

	recognizer.train(mijnheer, np.array(name))
	recognizer.save(ymlLoc)

	return ymlLoc

if __name__ == '__main__':
	path = os.path.dirname(os.path.abspath(__file__))

	app.secret_key = 'test'
	app.config['SESSION_TYPE'] = 'filesystem'
	app.run(debug=True)
