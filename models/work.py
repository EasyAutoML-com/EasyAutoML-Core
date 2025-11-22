from django.db import models
from django.conf import settings

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


class Work(models.Model):
    """
    Used inside AI: WorkProcessor, WorkDispatcher
    To create work for existing servers
    """

    creation_date_time = models.DateTimeField(auto_now_add=True)

    machine = models.ForeignKey('models.Machine', related_name="Work", on_delete=models.CASCADE)
    server = models.ForeignKey('models.Server', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        app_label = 'models'
        db_table = "work"
        indexes = [
            models.Index(fields=["machine"]),
            models.Index(fields=["server"]),
        ]

    is_work_training = models.BooleanField(null=True, blank=True)
    is_work_solving = models.BooleanField(null=True, blank=True)

    is_work_status_wait_for_server_start = models.BooleanField(null=True, blank=True)
    is_work_status_server_processing = models.BooleanField(null=True, blank=True)
    is_work_status_server_finished = models.BooleanField(null=True, blank=True)
    is_work_status_server_error = models.BooleanField(null=True, blank=True)

    work_started_at_date_time = models.DateTimeField(auto_now=True)
    work_started_by_server_name = models.CharField(max_length=45, default="")
    work_started_count_of_line_to_process = models.IntegerField(blank=True, null=True)
    work_duration_estimation_seconds = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    work_finished_at_date_time = models.DateTimeField(null=True, blank=True)
    work_finished_by_server_name = models.CharField(max_length=45, default="")
    work_finished_count_of_line_to_process = models.IntegerField(blank=True, null=True)
    work_finished_duration_seconds = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    learning_delay_elapsed_seconds = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )
    solving_delay_elapsed_seconds = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    text_error = models.JSONField(blank=True, default=dict)
    text_warning = models.JSONField(blank=True, default=dict)

    class Meta:
        app_label = 'models'
        db_table = "Work"
        indexes = [
            models.Index(fields=["creation_date_time"]),
            models.Index(fields=["machine"]),
            models.Index(fields=["server"]),
            models.Index(fields=["is_work_status_wait_for_server_start"]),
            models.Index(fields=["is_work_status_server_processing"]),
            models.Index(fields=["is_work_status_server_finished"]),
        ]

    def __str__(self):
        def get_work_status():
            if self.is_work_status_wait_for_server_start:
                return "Waiting for server start"
            elif self.is_work_status_server_processing:
                return "Server processing"
            elif self.is_work_status_server_finished:
                return "Server finished"
            elif self.is_work_status_server_error:
                return "Server error"
            return "Unknown status"

        work_status = get_work_status()
        return f"<Work_{self.id}: Machine={self.machine}, Server={self.server}, Status={work_status}>"

    def save(self, *args, **kwargs):
        possible_work_status = {field: bool(value) for field, value in vars(self).items() if field.startswith("is_work_status")}
        possible_work_mode = {
            field: bool(value)
            for field, value in vars(self).items()
            if field.startswith("is_work") and not field.startswith("is_work_status")
        }

        if sum(possible_work_status.values()) != 1:
            _logger.error(f"Object Work should be only in one status : {possible_work_status.keys()}")
            raise AttributeError(f"Object Work should be only in one status : {possible_work_status.keys()}")
        if sum(possible_work_mode.values()) != 1:
            _logger.error(f"Object Work should be only in one mode : {possible_work_mode.keys()}")
            raise AttributeError(f"Object Work should be only in one mode : {possible_work_mode.keys()}")
        return super(Work, self).save(*args, **kwargs)
