from uuid import uuid4
from typing import TYPE_CHECKING
from django.db import models
from django.conf import settings
from .billing_managers import CreditManager, DebitManager, CreditIXIOOManager, DebitIXIOOManager, MachineOperationsManager

if TYPE_CHECKING:
    from django.db.models import QuerySet


class Operation(models.Model):
    id = models.UUIDField(primary_key=True, editable=False, default=uuid4)
    
    # Custom manager for machine operations
    machine_operations = MachineOperationsManager()

    machine = models.ForeignKey(
        "models.Machine",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="machine_operations",
    )

    machine_owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="operations_as_machine_owner",
        db_column="billing__machine_owner",
    )

    operation_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="user_operations",
    )

    credit = models.OneToOneField(
        "models.Credit",
        blank=True,
        null=True,
        on_delete=models.CASCADE,
        related_name="credit_operation",
    )

    debit = models.OneToOneField(
        "models.Debit",
        blank=True,
        null=True,
        on_delete=models.CASCADE,
        related_name="debit_operation",
    )

    credit_ixioo = models.OneToOneField(
        "models.CreditIXIOO",
        blank=True,
        null=True,
        on_delete=models.CASCADE,
        related_name="credit_ixioo_operation",
    )

    debit_ixioo = models.OneToOneField(
        "models.DebitIXIOO",
        blank=True,
        null=True,
        on_delete=models.CASCADE,
        related_name="debit_ixioo_operation",
    )

    date_time = models.DateTimeField(auto_now=True)

    is_training_operation = models.BooleanField(default=False)
    is_solving_operation = models.BooleanField(default=False)
    is_solving_with_api_operation = models.BooleanField(default=False)
    is_solving_with_api_market_operation = models.BooleanField(default=False)
    is_buy_data_source_operation = models.BooleanField(default=False)
    is_machine_copy_operation = models.BooleanField(default=False)
    is_machine_copy_update_operation = models.BooleanField(default=False)

    is_credit_from_user_operation = models.BooleanField(default=False)
    is_pay_a_bill_operation = models.BooleanField(default=False)

    count_of_lines = models.IntegerField(blank=True, null=True)
    count_of_training_epoch = models.IntegerField(blank=True, null=True)

    class Meta:
        app_label = 'models'
        db_table = 'billing_operation'
        verbose_name = "Operation"
        verbose_name_plural = "Operations"
        indexes = [
            models.Index(fields=["date_time"]),
        ]

    @property
    def get_type(self) -> str:
        if self.is_training_operation:
            return "Training"
        elif self.is_solving_operation:
            return "Solving"
        elif self.is_solving_with_api_operation:
            return "Solving API"
        elif self.is_solving_with_api_market_operation:
            return "Solving API Market"
        elif self.is_buy_data_source_operation:
            return "Buy data source"
        elif self.is_machine_copy_operation:
            return "Machine copy"
        elif self.is_machine_copy_update_operation:
            return "Machine copy update"
        elif self.is_pay_a_bill_operation:
            return "Pay bill"
        elif self.is_credit_from_user_operation:
            return "Credit from user"
        else:
            return ""

    def __str__(self):
        return f"Operation {self.get_type} - {self.operation_user} - {self.date_time}"


class Credit(models.Model):
    """Table which contains all increasing of user balance"""

    id = models.UUIDField(primary_key=True, editable=False, default=uuid4)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    date_time = models.DateTimeField(auto_now=True)
    amount = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
    )

    objects = CreditManager()

    class Meta:
        app_label = 'models'
        db_table = 'billing_credit'
        verbose_name = "User balance credit transaction"
        verbose_name_plural = "User balance credit transactions"
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["date_time"]),
        ]


class Debit(models.Model):
    """Table witch contains all decreasing of User balance"""

    id = models.UUIDField(primary_key=True, editable=False, default=uuid4)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    date_time = models.DateTimeField(auto_now=True)
    amount = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
    )

    objects = DebitManager()

    class Meta:
        app_label = 'models'
        db_table = 'billing_debit'
        verbose_name = "User balance debit transaction"
        verbose_name_plural = "User balance debit transactions"
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["date_time"]),
        ]


class CreditIXIOO(models.Model):
    """Table which contains all increasing of IXIOO balance"""

    id = models.UUIDField(primary_key=True, editable=False, default=uuid4)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    date_time = models.DateTimeField(auto_now=True)
    amount = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
    )

    objects = CreditIXIOOManager()

    class Meta:
        app_label = 'models'
        db_table = 'billing_creditixioo'
        verbose_name = "IXIOO balance credit transaction"
        verbose_name_plural = "IXIOO balance credit transactions"
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["date_time"]),
        ]


class DebitIXIOO(models.Model):
    """Table which contains all decreasing of IXIOO balance"""

    id = models.UUIDField(primary_key=True, editable=False, default=uuid4)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    date_time = models.DateTimeField(auto_now=True)
    amount = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
    )

    objects = DebitIXIOOManager()

    class Meta:
        app_label = 'models'
        db_table = 'billing_debitixioo'
        verbose_name = "IXIOO balance debit transaction"
        verbose_name_plural = "IXIOO balance debit transactions"
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["date_time"]),
        ]
