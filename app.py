import torch
from flask import Flask, render_template, Response
from yolovision import detect_and_stream

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_and_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
