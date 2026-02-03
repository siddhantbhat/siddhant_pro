


from flask import Flask


import os 

def create_app():
    # Define project root paths based on __file__ location
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app = Flask(__name__, template_folder='templates', static_folder='static')

    
    
    
    app.config['SECRET_KEY'] = "supersecretkey" # Required for flash messages

    # Register the Blueprint, making all routes active
    from App.routes.main import  main_bp
    app.register_blueprint(main_bp)

    return app
