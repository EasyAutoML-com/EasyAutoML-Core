from django.db import models


class Server(models.Model):
    server_name = models.CharField(max_length=50, unique=True)
    server_date_time = models.DateTimeField(auto_now_add=True, help_text="Date of creation")
    server_info_ready = models.BooleanField(null=True, blank=True, help_text="=1 if ready or =0")
    server_gpu_ram = models.JSONField(default=list, blank=True)

    class Meta:
        app_label = 'models'
        db_table = "server"
        indexes = [
            models.Index(fields=["server_info_ready"]),
        ]

    def __str__(self):
        readiness_status = "Ready" if self.server_info_ready else "Busy"
        return ( "<"
            f"Server ID: {self.id}, "
            f"Name: {self.server_name}, "
            f"At: {self.server_date_time}, "
            f"Status: {readiness_status}, "
            ">"
        )
