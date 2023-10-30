from app import app
from flask import render_template, request, redirect

@app.route('/')
def mainpage():
    return render_template('index.html')