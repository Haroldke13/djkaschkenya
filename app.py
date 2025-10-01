from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import cv2, os, numpy as np
from bson.objectid import ObjectId
from datetime import datetime
import base64, json
# Import Flask core framework and render_template function
from flask import Flask, render_template
# Import SQLAlchemy ORM for database management
from flask_sqlalchemy import SQLAlchemy
# Import Flask-Login for user authentication
from flask_login import LoginManager
# Import Flask-Migrate for database migrations
from flask_migrate import Migrate
# Import OAuth for third-party authentication
from authlib.integrations.flask_client import OAuth
# Import OS module for path handling
import os
# Import Flask-Mail for sending emails
from flask_mail import Mail


# Flask Extensions
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user
from flask_migrate import Migrate
from authlib.integrations.flask_client import OAuth
from flask_mail import Mail, Message

# Import dotenv
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)




# Secret key
app.secret_key = os.getenv("SECRET_KEY")
app.config["SECRET_KEY"] = app.secret_key

# -------------------- DATABASE --------------------
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["MONGO_DBNAME"] = os.getenv("MONGO_DB_NAME")
mongo = PyMongo(app)

# Flask environment
app.config["ENV"] = os.getenv("FLASK_ENV", "development")

mongo = PyMongo(app)

# Create Flask-Migrate object for migrations
migrate = Migrate()
# Create OAuth object for external login
oauth = OAuth()
# Create Mail object for sending emails
mail = Mail()

# -------------------- EXTENSIONS INIT --------------------
migrate.init_app(app, mongo)   # ⚠️ SQLAlchemy usually expected here
oauth.init_app(app)

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'juellharold@gmail.com'
app.config['MAIL_PASSWORD'] = 'oykeghzvmoddyhns'
app.config['MAIL_DEFAULT_SENDER'] = 'juellharold@gmail.com'
mail.init_app(app)

# PDF output directory
app.config['PDF_OUTPUT_DIR'] = os.path.join(app.root_path, 'static/billing_pdfs')

# Google OAuth configuration
app.config['GOOGLE_CLIENT_ID'] = '1052110297217-3pi3l4eqhktgocn2cjrvt03bqurvu2qq.apps.googleusercontent.com'
app.config['GOOGLE_CLIENT_SECRET'] = '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'
app.config['GOOGLE_DISCOVERY_URL'] = 'https://accounts.google.com/.well-known/openid-configuration'

# Register Google OAuth service
oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url=app.config['GOOGLE_DISCOVERY_URL'],
)





# -------------------- GOOGLE SIGNUP ROUTES --------------------
import uuid, tempfile, pathlib

TEMP_FACE_DIR = os.path.join(app.root_path, "temp_faces")
os.makedirs(TEMP_FACE_DIR, exist_ok=True)

@app.route('/google-signup', methods=["POST"])
def google_signup():
    face_data = request.form.get("face_data")
    if not face_data:
        flash("Please capture your face before signing up with Google.", "danger")
        return redirect(url_for("signup"))

    # Generate a unique temp file name
    token = str(uuid.uuid4())
    temp_path = os.path.join(TEMP_FACE_DIR, f"{token}.json")

    # Save base64 JSON to disk
    with open(temp_path, "w") as f:
        f.write(face_data)

    # Keep only the token in session (small string)
    session["pending_face_token"] = token

    redirect_uri = url_for('google_signup_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/google-signup-callback')
def google_signup_callback():
    try:
        token = oauth.google.authorize_access_token()
        userinfo = oauth.google.get('userinfo').json()

        if not userinfo or 'email' not in userinfo:
            flash('Google signup failed. No email found.', 'danger')
            return redirect(url_for('signup'))

        # Already exists?
        user = mongo.db.users.find_one({"email": userinfo['email']})
        if user:
            flash("Account already exists. Please log in.", "info")
            return redirect(url_for("login"))

        # Create new user
        new_user = {
            "fname": userinfo.get("given_name", ""),
            "lname": userinfo.get("family_name", ""),
            "email": userinfo['email'],
            "username": userinfo['email'].split('@')[0],
            "created_at": datetime.now(),
            "face_registered": False
        }
        inserted = mongo.db.users.insert_one(new_user)
        user_id = str(inserted.inserted_id)

        # ---- Load pending face data from disk ----
        token = session.pop("pending_face_token", None)
        if token:
            temp_path = os.path.join(TEMP_FACE_DIR, f"{token}.json")
            if os.path.exists(temp_path):
                with open(temp_path, "r") as f:
                    face_data = f.read()
                os.remove(temp_path)  # cleanup

                # Decode and train
                images = json.loads(face_data)
                face_images = []
                for idx, img_str in enumerate(images):
                    img_data = base64.b64decode(img_str.split(",")[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    face_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if face_img is not None:
                        face_images.append(face_img)
                        cv2.imwrite(os.path.join(FACE_DIR, f"user_{user_id}_{idx}.png"), face_img)

                if train_and_save(user_id, face_images):
                    mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"face_registered": True}})
                    flash("Google signup complete. Face model trained!", "success")
                else:
                    flash("Face training failed. Try again.", "danger")
                    return redirect(url_for("signup"))

        session["user_id"] = user_id
        return redirect(url_for("home"))

    except Exception as e:
        print("OAuth signup error:", e)
        flash('Google signup failed due to an error.', 'danger')
        return redirect(url_for('signup'))


# -------------------- GOOGLE LOGIN ROUTES --------------------
@app.route('/google-login')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/callback')
def google_callback():
    try:
        # Step 1: Exchange code for access token
        token = oauth.google.authorize_access_token()

        # Step 2: Fetch user info
        userinfo = oauth.google.get('userinfo').json()

        if not userinfo or 'email' not in userinfo:
            flash('Google authentication failed. No email found.', 'danger')
            return redirect(url_for('login'))

        # Step 3: Check if user already exists in Mongo
        user = mongo.db.users.find_one({"email": userinfo['email']})
        if not user:
            user = {
                "fname": userinfo.get("given_name", ""),
                "lname": userinfo.get("family_name", ""),
                "email": userinfo['email'],
                "username": userinfo['email'].split('@')[0],
                "created_at": datetime.now(),
                "face_registered": False
            }
            mongo.db.users.insert_one(user)

        # Step 4: Set session
        session["user_id"] = str(user["_id"])
        flash("Logged in with Google successfully!", "success")
        return redirect(url_for("home"))

    except Exception as e:
        print("OAuth callback error:", e)
        flash('Google login failed due to an error.', 'danger')
        return redirect(url_for('login'))

# Face dataset directory
FACE_DIR = "static/faces"
os.makedirs(FACE_DIR, exist_ok=True)

# -------------------- HELPER: Train and Save --------------------
def train_and_save(user_id, face_images):
    """Train LBPH model for a given user and save to disk."""
    if not face_images:
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.array([1] * len(face_images))  # one label per user model
    recognizer.train(face_images, labels)

    model_path = os.path.join(FACE_DIR, f"user_{user_id}_model.xml")
    recognizer.save(model_path)
    print("✅ Model saved at:", model_path)
    return True

# app.py (Flask)

from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup




@app.route("/")
def home():
    
    return render_template("home.html")


# -------------------- SIGNUP --------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        face_data = request.form.get("face_data")
        if not face_data:
            flash("Please capture your face before signing up.", "danger")
            return redirect(url_for("signup"))

        # Save user info first
        user = {
            "fname": request.form["fname"],
            "lname": request.form["lname"],
            "dob": request.form["dob"],
            "phone": request.form["phone"],
            "email": request.form["email"],
            "passwd": generate_password_hash(request.form["passwd"]),
            "town": request.form["town"],
            "county": request.form["county"],
            "face_registered": False,
            "created_at": datetime.now()
        }
        user_id = str(mongo.db.users.insert_one(user).inserted_id)

        # Decode images sent from frontend
        images = json.loads(face_data)
        face_images = []
        for idx, img_str in enumerate(images):
            img_data = base64.b64decode(img_str.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if face_img is not None:
                face_images.append(face_img)
                cv2.imwrite(os.path.join(FACE_DIR, f"user_{user_id}_{idx}.png"), face_img)

        # Train and save model
        if train_and_save(user_id, face_images):
            mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"face_registered": True}})
            flash("Signup successful. Face model trained!", "success")
        else:
            flash("Face training failed. Try again.", "danger")

        return redirect(url_for("login"))

    return render_template("signup.html")

# -------------------- LOGIN --------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        passwd = request.form["passwd"]
        user = mongo.db.users.find_one({"email": email})

        if user and check_password_hash(user["passwd"], passwd):
            if user.get("face_registered", False):
                return redirect(url_for("login_face", user_id=str(user["_id"])))
            else:
                session["user_id"] = str(user["_id"])
                return redirect(url_for("home"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")

# At app startup
recognizers = {}

def get_recognizer(user_id):
    if user_id not in recognizers:
        model_path = os.path.join(FACE_DIR, f"user_{user_id}_model.xml")
        if os.path.exists(model_path):
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.read(model_path)
            recognizers[user_id] = rec
    return recognizers.get(user_id)



@app.route("/login_face/<user_id>")
def login_face(user_id):
    return render_template("login_face.html", user_id=user_id)

# -------------------- VERIFY FACE --------------------
import base64, os, cv2, numpy as np
from flask import request, session

FACE_DIR = "static/faces"

@app.route("/verify_face/<user_id>", methods=["POST"])
def verify_face(user_id):
    try:
        data = request.get_json()
        img_data = data["image"].split(",")[1]  # remove "data:image/png;base64,"
        img_bytes = base64.b64decode(img_data)

        # Decode base64 → numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_color is None:
            return {"message": "Failed to decode image"}, 400

        # Save a debug image to inspect what the backend receives
        debug_path = f"debug_{user_id}.jpg"
        cv2.imwrite(debug_path, frame_color)
        print(f"✅ Saved debug frame: {debug_path}, shape: {frame_color.shape}")

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        # Load Haar cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Try to detect faces (looser parameters for testing)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # lower for more sensitivity
            minNeighbors=3,     # fewer neighbors, less strict
            minSize=(50, 50)    # ignore tiny blobs
        )

        print(f"🔍 Faces detected: {len(faces)}")

        if len(faces) == 0:
            return {"message": "No face detected"}, 400

        # Optionally draw rectangle and save debug detection result
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        detected_path = f"debug_{user_id}_detected.jpg"
        cv2.imwrite(detected_path, frame_color)
        print(f"✅ Saved detection result: {detected_path}")

        # Load recognizer
        recognizer = get_recognizer(user_id)
        if not recognizer:
            return {"message": "No trained model found"}, 400

        # Predict using first face found
        (x, y, w, h) = faces[0]
        face_region = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_region)
        print(f"Prediction → label: {label}, confidence: {confidence:.2f}")

        if confidence < 70:  # threshold
            return {"message": "Face verified successfully!"}, 200
        else:
            return {"message": "Face not recognized"}, 400

    except Exception as e:
        print("❌ Error in /verify_face:", str(e))
        return {"message": "Internal Server Error"}, 500



# -------------------- PROFILE --------------------
@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = mongo.db.users.find_one({"_id": ObjectId(session["user_id"])})
    return render_template("profile.html", user=user)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))







if __name__ == "__main__":
    app.run(debug=False,port=5000)
