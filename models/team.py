from uuid import uuid4
from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType

# Import constants from apps
from .apps import SUPER_ADMIN_EASYAUTOML_EMAIL, SUPER_ADMIN_EASYAUTOML_TEAM_NAME

# Import logger
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


class Team(models.Model):
    name = models.CharField(_("Team name"), max_length=200)

    admin_user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.CASCADE, blank=True)
    users = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name="Teams", blank=True)

    url = models.URLField(unique=True, default=None, null=True, blank=True)

    permission = models.OneToOneField(Permission, on_delete=models.CASCADE, null=True)

    class Meta:
        app_label = 'models'
        db_table = "Team"
        verbose_name = _("Team")
        verbose_name_plural = _("Teams")
        indexes = [
            models.Index(fields=["name"]),
        ]

    def __str__(self):
        return f"{self.name}"

    def delete(self, using=None, keep_parents=False):
        # Lazy import to avoid circular dependency
        from django.utils.module_loading import import_string
        Machine = import_string('models.machine.Machine')

        for user in self.users.all():
            self.remove_user_from_team(user)

        Machine.objects.filter(machine_owner_team=self).update(machine_owner_team=None)
        super(Team, self).delete(using, True)

    @classmethod
    def get_super_admin_team(cls):
        return cls.objects.get(name=SUPER_ADMIN_EASYAUTOML_TEAM_NAME)

    def add_user_to_team(self, user):
        self.users.add(user)
        user.user_permissions.add(self.permission)

    def remove_user_from_team(self, user):
        if user not in self.users.all():
            _logger.info(f"User <id={user.id}> in not a Team <id={self.id}> member")
            raise AttributeError("This user in not a Team member")

        self.users.remove(user)
        self._remove_user_permission_for_team(user)

    def user_leave_team(self, user):
        if user not in self.users.all():
            _logger.info(f"User <id={user.id}> in not a Team <id={self.id}> member")
            raise ValueError("This user in not a Team member")

        self.users.remove(user)
        self._remove_user_permission_for_team(user)

    def _remove_user_permission_for_team(self, user):
        user.user_permissions.remove(self.permission)

    def generate_link(self):
        self.url = str(uuid4())
        self.save()

    def create_permission(self):
        content_type = ContentType.objects.get_for_model(Team)
        permission = Permission(
            codename=f"can_use_team_{self.id}",
            name=f"Can use team with id={self.id}",
            content_type=content_type,
        )
        permission.save()
        self.permission = permission
        self.save()
        return permission
