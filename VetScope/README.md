# VetScope - AI-Powered Dog Disease Detection

VetScope is a web application that uses artificial intelligence to detect diseases in dogs through image analysis. Built with Flask and TensorFlow, it provides instant disease detection, symptoms information, and care recommendations.

## Features

- AI-powered disease detection
- User authentication system
- Drag-and-drop image upload
- Instant analysis results
- Detailed symptoms and care tips
- Modern, responsive UI
- Secure data handling

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Backend**: Python, Flask
- **Database**: SQLite
- **AI/ML**: TensorFlow, EfficientNetB3
- **Authentication**: Flask-Login
- **ORM**: SQLAlchemy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kaivlyax/VetScope.git
   cd VetScope
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the environment variables:
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development
   ```

5. Initialize the database:
   ```bash
   flask shell
   >>> from app import db
   >>> db.create_all()
   >>> exit()
   ```

6. Run the application:
   ```bash
   flask run
   ```

## Usage

1. Register for an account or login if you already have one
2. Navigate to the dashboard
3. Upload an image of your dog
4. Get instant analysis results with:
   - Disease detection
   - Confidence level
   - Symptoms
   - Care recommendations

## Deployment

The application is configured for deployment on Render. To deploy:

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the amazing deep learning framework
- Flask team for the lightweight WSGI web application framework
- Bootstrap team for the frontend framework 