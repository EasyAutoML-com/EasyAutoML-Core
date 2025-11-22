from django.db import models
from django.conf import settings
from django_mysql.models import SizedBinaryField

# Lazy imports to avoid circular dependencies
def get_team_model():
    from django.utils.module_loading import import_string
    return import_string('models.team.Team')

def get_machine_model():
    from django.utils.module_loading import import_string
    return import_string('models.machine.Machine')

CATEGORIES = (
    ("1", "Data Engineering"),
    ("2", "Machine performance"),
    ("3", "API Integration"),
    ("4", "Other"),
)


class ConsultingRequest(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    team = models.ForeignKey('models.Team', on_delete=models.CASCADE, null=True)
    category = models.CharField(max_length=1, choices=CATEGORIES, null=True, blank=True)
    description = models.TextField(blank=True)
    file = SizedBinaryField(size_class=3, null=True, blank=True)
    uploaded_file = models.FileField(upload_to="consulting", blank=True, null=True)
    machine = models.ForeignKey('models.Machine', null=True, blank=True, on_delete=models.CASCADE)
    is_status_request_open = models.BooleanField(blank=True, null=True)
    is_status_waiting_consultant_approval = models.BooleanField(blank=True, null=True)
    waiting_consultant_approval_date_time = models.DateTimeField(blank=True, null=True)
    is_status_contract_finished = models.BooleanField(blank=True, null=True)
    is_status_contract = models.BooleanField(blank=True, null=True)
    is_status_contract_failed = models.BooleanField(blank=True, null=True)
    is_status_contract_cancel = models.BooleanField(blank=True, null=True)
    contract_start_date_time = models.DateTimeField(blank=True, null=True)
    contract_end_date_time = models.DateTimeField(blank=True, null=True)
    consultant_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="Consultant_User",
        null=True,
    )
    amount_usd = models.IntegerField(null=True, blank=True)
    budget_estimation = models.FloatField(null=True, blank=True)
    delay_days = models.IntegerField(null=True, blank=True)
    date_time_creation = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'models'
        db_table = 'ConsultingRequest'
