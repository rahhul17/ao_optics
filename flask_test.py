from flask import Flask
app=Flask(_name_)
@app.poute('/')
def hello_world():
    return 'Hello World!'