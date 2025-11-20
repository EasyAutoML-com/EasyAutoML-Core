from django.db import models
from django.utils import timezone


class MachineTableLockWrite(models.Model):
    """
    Model representing a lock on a table.

    :param table_name: The name of the table to lock.
    :type table_name: str
    :param locked_at: The timestamp when the lock was acquired.
    :type locked_at: datetime
    """
    table_name = models.CharField(max_length=255, unique=True)
    locked_at = models.DateTimeField(default=timezone.now)

    class Meta:
        app_label = 'models'
        db_table = "machine_table_lock_write"
        verbose_name = "Machine Table Lock Write"
        verbose_name_plural = "Machine Table Lock Writes"

    def __str__(self):
        return self.table_name
