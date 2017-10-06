from flask import Flask
from car import *
app = Flask(__name__)          

@app.route('/forward')
def fwd():	
    forward()
    return "Forward"

@app.route('/left')
def left_turn():
    left()
    return "Turn Left"

@app.route('/right')
def right_turn():
    right()
    return "Turn Right"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
