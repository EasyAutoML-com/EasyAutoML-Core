"""
Standalone Django settings for Backend services (WorkProcessor, WorkMonitor, WorkDispatcher).
This allows Backend to run independently without the WWW folder.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build paths inside the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 

# SECURITY: Secret key from environment variable (required in production)
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "-za&^h2#b9i)&9ev58yc1o**!a2qdeibkr_w!*inr#5*4t#^#z")

# SECURITY WARNING: Using default secret key is only for development!
if SECRET_KEY == "-za&^h2#b9i)&9ev58yc1o**!a2qdeibkr_w!*inr#5*4t#^#z":
    import warnings
    warnings.warn(
        "WARNING: Using default SECRET_KEY. Set DJANGO_SECRET_KEY environment variable in production!",
        UserWarning
    )

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",") if os.getenv("ALLOWED_HOSTS") else ["*"]

# Application definition
# Minimal Django apps + only the models app
# IMPORTANT: models app must come BEFORE django.contrib.auth so that
# the User model is available when django.contrib.auth tries to access it
INSTALLED_APPS = [
    # Core Django apps
    "django.contrib.contenttypes",
    # Models app must be before django.contrib.auth for AUTH_USER_MODEL to work
    "models.apps.ModelsConfig",
    "django.contrib.auth",
    "django.contrib.sites",
    
    # Apps required by models
    "rest_framework",
    "rest_framework.authtoken",
    "eamllogger",
]

# Use custom User model
AUTH_USER_MODEL = 'models.User'
SITE_ID = 1

# Minimal middleware (only what's necessary for models/ORM to work)
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
]

# Database configuration - SQLite only
# Use SQLite database from project root (created by Create Start Set Sqlite Database.py)
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(BASE_DIR, "start_set_databases.sqlite3"))
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": SQLITE_DB_PATH,
    },
}
# SQLite URL needs forward slashes
SQLITE_URL_PATH = SQLITE_DB_PATH.replace("\\", "/")
DATABASE_SQL = f"sqlite:///{SQLITE_URL_PATH}"



    
# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_L10N = True
USE_TZ = False

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# No URLs needed for backend services
ROOT_URLCONF = None

# No static/media files needed for backend services
STATIC_URL = "/static/"
STATIC_ROOT = None

# No templates needed for backend services
TEMPLATES = []

# Mock WSGI application (not used but required by Django)
WSGI_APPLICATION = None








# Model decimal field defaults (used by models/user.py and others)
MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS = 32
MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES = 10

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}



