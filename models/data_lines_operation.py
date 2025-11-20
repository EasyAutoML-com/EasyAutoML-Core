from django.db import models


class DataLinesOperation(models.Model):
    """
    It works only with AI
    """

    machine_id = models.IntegerField()
    date_time = models.DateTimeField(auto_now_add=True)
    is_added_for_learning = models.BooleanField(null=True, blank=True)
    is_added_for_solving = models.BooleanField(null=True, blank=True)

    class Meta:
        app_label = 'models'
        db_table = "DataLinesOperation"
        verbose_name = "Data Lines Operation"
        verbose_name_plural = "Data Lines Operations"
        indexes = [
            models.Index(fields=["machine_id"]),
        ]

    def __str__(self):
        learning_status = "Yes" if self.is_added_for_learning else "No"
        solving_status = "Yes" if self.is_added_for_solving else "No"
        return ( "<"
            f"DataLinesOperation (Machine ID: {self.machine_id}, "
            f"Date: {self.date_time}, "
            f"Added for Learning: {learning_status}, "
            f"Added for Solving: {solving_status})"
            ">"
        )
