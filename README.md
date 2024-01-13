# Yoga Pose Detection Web Application

![Yoga Pose Detection](static/images/yoga_pose_detection.png)

## Overview

This web application uses Flask, Mediapipe, and machine learning to detect yoga poses either from an uploaded image or through a webcam. It provides real-time feedback on yoga poses, making it a useful tool for yoga enthusiasts and practitioners.

## Features

- Real-time yoga pose detection using webcam.
- Pose detection from an uploaded image.
- User authentication and signup functionality.
- Secure password storage using SQLite.

## Prerequisites

- Python 3.7 or higher
- Install required dependencies:
'''bash
  pip install -r requirements.txt


## Installation
- Clone the repository:
'''bash
  git clone https://github.com/your-username/Yoga-Pose-Detection.git
  cd Yoga-Pose-Detection


## Install dependencies:
'''bash
  pip install -r requirements.txt

## Usage

- Run the Flask application: python app.py
- Open your web browser and go to http://127.0.0.1:5001/ to access the application.
- Navigate to the webcam section to see real-time yoga pose detection.
- You can also sign up and upload an image for pose detection.

## File Structure

- app.py: Main Flask application.
- upload/: Directory to store uploaded images.
- static/: Static files (CSS, JS, images).
- templates/: HTML templates.
- detect_pose.pkl: Machine learning model for pose detection.


## Libraries and Frameworks

- Flask: Web application framework.
- Mediapipe: For pose detection.
- OpenCV: Computer vision library.
- Pandas, NumPy: Data manipulation and numerical computing.
- SQLite: Database for user authentication.

## Contributing
Feel free to contribute to this project by opening issues, submitting pull requests, or providing feedback.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
