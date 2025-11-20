from django.conf import settings
from django.db import models
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.core.mail import send_mail
from django.utils import timezone
# Optional imports for web functionality
try:
    # Only import allauth if it's available and Django is ready
    from django.apps import apps
    if 'allauth.account' in [app.label for app in apps.get_app_configs()]:
        from allauth.account.models import EmailAddress
        ALLAUTH_AVAILABLE = True
    else:
        ALLAUTH_AVAILABLE = False
except (ImportError, RuntimeError, AttributeError):
    # allauth not installed or Django not ready
    ALLAUTH_AVAILABLE = False
    
try:
    from rest_framework.authtoken.models import Token
    REST_FRAMEWORK_AVAILABLE = True
except ImportError:
    REST_FRAMEWORK_AVAILABLE = False

# Import logger - will be updated to use shared logger
from django.utils.module_loading import import_string

# Import constants - will need to be handled
try:
    from SharedConstants import SUPER_ADMIN_EASYAUTOML_EMAIL
except ImportError:
    SUPER_ADMIN_EASYAUTOML_EMAIL = "SuperAdmin@easyautoml.com"

# Lazy import to avoid circular dependency
def get_team_model():
    return import_string('models.team.Team')

def get_logger():
    try:
        from eamllogger.EasyAutoMLLogger import EasyAutoMLLogger
        return EasyAutoMLLogger()
    except ImportError:
        # Fallback to models logger
        try:
            from models.logger import EasyAutoMLLogger
            return EasyAutoMLLogger()
        except ImportError:
            import logging
            return logging.getLogger(__name__)

_logger = get_logger()


class UserManager(BaseUserManager):
    use_in_migrations = True

    def _create_user(self, email, password, first_name=None, last_name=None, **extra_fields):
        """
        Create and save a user with the given username, email, and password.
        """
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.first_name = first_name
        user.last_name = last_name
        user.save(using=self._db)
        return user

    def create_user(self, email=None, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password, first_name=None, last_name=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_super_admin", True)

        if extra_fields.get("is_staff") is not True:
            _logger.error("Superuser must have is_staff=True")
            raise ValueError("Superuser must have is_staff=True")
        if extra_fields.get("is_superuser") is not True:
            _logger.error("Superuser must have is_superuser=True.")
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(email, password, first_name, last_name, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    USERNAME_FIELD = "email"
    EMAIL_FIELD = "email"
    REQUIRED_FIELDS = []

    email = models.EmailField(null=True, unique=True)
    first_name = models.CharField(max_length=140, null=True, blank=True)
    last_name = models.CharField(max_length=140, null=True, blank=True)

    user_profile = models.TextField(null=True, blank=True)
    time_format = models.CharField(max_length=3, default="24H")
    date_format = models.CharField(max_length=3, default="DMY")
    date_separator = models.CharField(max_length=1, default="/")
    datetime_separator = models.CharField(max_length=1, default=" ")
    decimal_separator = models.CharField(max_length=1, default=",")
    coupons_activated_date = models.JSONField(default=dict, blank=True)

    is_super_admin = models.BooleanField(default=False)
    is_staff = models.BooleanField("staff status", default=False)
    is_active = models.BooleanField("active", default=True)
    date_joined = models.DateTimeField("date joined", default=timezone.now)
    coupon_balance = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        default=0,
    )
    user_balance = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        default=0,
    )
    user_ixioo_balance = models.DecimalField(
        null=True,
        blank=True,
        max_digits=settings.MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS,
        decimal_places=settings.MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES,
        default=0,
    )
    last_billing_time = models.DateTimeField(null=True, blank=True)

    objects = UserManager()

    class Meta:
        app_label = 'models'
        swappable = "AUTH_USER_MODEL"
        db_table = "user"
        indexes = [
            models.Index(fields=["email"]),
        ]

    def __str__(self):
        return f"{self.get_full_name()} {self.email}"

    @property
    def APIKey(self):
        if REST_FRAMEWORK_AVAILABLE:
            return Token.objects.get(user=self)
        return None
    full_name = property(lambda self: self.get_full_name())

    @classmethod
    def get_super_admin(cls):
        try:
            admin = cls.objects.get(email=SUPER_ADMIN_EASYAUTOML_EMAIL)
        except Exception:
            # Use warning instead of error to avoid automatic exception raising
            _logger.warning(f"Unable to find super admin '{SUPER_ADMIN_EASYAUTOML_EMAIL}' in the user database")
            raise ValueError(f"Unable to find super admin '{SUPER_ADMIN_EASYAUTOML_EMAIL}' in the user database")
        return admin

    def clean(self):
        super().clean()
        self.email = self.__class__.objects.normalize_email(self.email)

    def get_full_name(self):
        """
        Return the first_name plus the last_name, with a space in between also remove None.
        """
        full_name = "%s %s" % (self.first_name or "", self.last_name or "")
        return full_name.strip()

    def get_user_teams(self):
        Team = get_team_model()
        return Team.objects.filter(users__id=self.id)

    def email_user(self, subject, message, from_email=None, **kwargs):
        """Send an email to this user"""
        send_mail(subject, message, from_email, [self.email], **kwargs)

    def create_team(self, team_name):  # TODO: move to Team model
        Team = get_team_model()
        user_created_groups = [team.name for team in Team.objects.filter(admin_user=self) if team == team_name]

        if user_created_groups:
            _logger.error(f'You have already created the team with name "{team_name}"')
            raise NameError(f'You have already created the team with name "{team_name}"')

        team = Team(name=team_name, admin_user=self)
        team.save()
        team.create_permission()
        team.users.add(self)
        self.user_permissions.add(team.permission)

    def remove_email_verification(self):
        if ALLAUTH_AVAILABLE:
            email = EmailAddress(user=self, email=self.email)
            email.verified = True
            email.save()
