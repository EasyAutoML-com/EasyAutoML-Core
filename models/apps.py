from django.apps import AppConfig
from django.db.models.signals import post_migrate

# Import constants from SharedConstants
from SharedConstants import (
    SUPER_ADMIN_EASYAUTOML_EMAIL,
    SUPER_ADMIN_EASYAUTOML_TEAM_NAME,
    SUPER_ADMIN_EASYAUTOML_TEAM_MEMBERS,
)


def create_easyautoml_admin_team(sender, **kwargs):
    """Create the EasyAutoML super admin team after migrations"""
    from django.contrib.auth import get_user_model
    from models import Team
    
    User = get_user_model()

    try:
        Team.objects.get(name=SUPER_ADMIN_EASYAUTOML_TEAM_NAME)
    except:
        # Create admin user if it doesn't exist
        admin_user, created = User.objects.get_or_create(
            email=SUPER_ADMIN_EASYAUTOML_EMAIL,
            defaults={
                'first_name': 'Laurent',
                'last_name': 'Bruere',
                'is_staff': True,
                'is_superuser': True,
                'is_active': True,
            }
        )
        
        if created:
            admin_user.set_password('easyautoml999')
            admin_user.save()
        
        team = Team(
            name=SUPER_ADMIN_EASYAUTOML_TEAM_NAME,
            admin_user=admin_user,
        )
        team.save()
        team.create_permission()
        for email, _, _ in SUPER_ADMIN_EASYAUTOML_TEAM_MEMBERS:
            try:
                user = User.objects.get(email=email)
                team.add_user_to_team(user)
            except User.DoesNotExist:
                # Skip if user doesn't exist
                pass


class ModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'models'
    label = 'models'
    verbose_name = 'Shared Models'
    
    def ready(self):
        """Import models when Django is ready"""
        # Import User model FIRST - it's critical for django.contrib.auth
        # This must happen before django.contrib.auth.ready() is called
        # Since models app comes before django.contrib.auth in INSTALLED_APPS,
        # this ready() method is called first, ensuring User is available
        from . import user
        # Import all other models to register them with Django
        from . import team, machine, nn_model, graph
        from . import logger, data_lines_operation
        from . import machine_table_lock_write, encdec_configuration
        
        # Connect post_migrate signal to create super admin team
        post_migrate.connect(create_easyautoml_admin_team, sender=self)