from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from pipeline.prediction import PredictionPipeline

app = Flask(__name__) #initializing the flask app

@app.route("/")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)
