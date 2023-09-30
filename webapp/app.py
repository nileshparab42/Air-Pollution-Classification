from flask import Flask, render_template, request
from PIL import Image
import keras.utils as image
from keras.models import load_model
from keras.utils import load_img, img_to_array
# from keras.preprocessing import image

app = Flask(__name__)

model = load_model('VGG16.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)
	# print("ANS: "+i)
	i = i.reshape(1, 224,224,3)
	p = model.predict(i)
	n = p[0]
	ans = ''
	if n[0]==1:
		ans += 'A: Good'
	if n[1]==1:
		ans += 'B: Moderate'
	if n[2]==1:
		ans += 'C: Unhealthy for Sensitive Groups'
	if n[3]==1:
		ans += 'D: Unhealthy'
	if n[4]==1:
		ans += 'E: Very Unhealthy'
	if n[5]==1:
		ans += 'F: Severe'

	return ans


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)