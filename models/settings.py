"""
Minimal Django settings for shared models package.
This allows the models to be used independently from the web application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

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

# Application definition - Essential apps needed for models
INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.sites',
    
    # Required for User model dependencies
    'rest_framework',
    'rest_framework.authtoken',
    
    # Our shared models package
    'models.apps.ModelsConfig',
]

# Required for django.contrib.sites
SITE_ID = 1

# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Custom user model
AUTH_USER_MODEL = 'models.User'

# Internationalization
# https://docs.djangoproject.com/en/3.2/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Model field defaults
MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS = 20
MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES = 8

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
