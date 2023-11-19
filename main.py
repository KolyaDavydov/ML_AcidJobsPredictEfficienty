
from flask import Flask, request, render_template
from main_app import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getprediction', methods=['POST'])
def getprediction():
    input = [(x) for x in request.form.values()]
    prediction = predict(input)

    return render_template('index.html', output='Оценка скважины :{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)