from django.conf import settings
from django.db import models


class Billing(models.Model):
    """
    Used on AI and machine for bill any operation:
    solving, training, solving API, export code, buy dataset, credits, duplicable
    """

    machine = models.ForeignKey("models.Machine", on_delete=models.CASCADE, null=True, blank=True)

    machine_owner_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="machine_owner_user",
        db_column="deprecated_billing_machine_owner_user",
    )

    Operation_User = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="Operation_User",
    )

    Date = models.DateField(auto_now=True)

    OperationIsTraining = models.BooleanField(null=True, blank=True)
    CountOfLinesTraining = models.IntegerField(blank=True, null=True)
    CountOfTrainingEpoch = models.IntegerField(blank=True, null=True)
    TrainingCostMachineUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    OperationIsSolving = models.BooleanField(null=True, blank=True)
    CountOfLinesSolvedOwner = models.IntegerField(blank=True, null=True)
    SolvingCostMachineUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    OperationIsSolvingAPI = models.BooleanField(null=True, blank=True)
    CountOfLinesSolvedAPI = models.IntegerField(null=True, blank=True)
    SolvingAPIIncomeUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )
    SolvingCostMachineAPI_Total_USD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    OperationIsBuyDataSource = models.BooleanField(null=True, blank=True)
    MachineBuyDataSourceIncomeUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    OperationIsMachineCopy = models.BooleanField(null=True, blank=True)
    MachineCopyCostIncomeUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    OperationIsMachineCopyUpdate = models.BooleanField(null=True, blank=True)
    MachineCopyUpdateIncomeUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    OperationIsMachineExport = models.BooleanField(null=True, blank=True)
    MachineExportIncomeUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )
    MachineExportCostUSD = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    CreditFromIxioo = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )
    CreditFromUser = models.DecimalField(
        max_digits=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_MAX_DIGITS', 20),
        decimal_places=getattr(settings, 'MODEL_DECIMAL_FIELD_DEFAULT_DECIMAL_PLACES', 8),
        null=True,
        blank=True,
    )

    class Meta:
        app_label = 'models'
        db_table = "Billing"
        verbose_name = "Billing"
        verbose_name_plural = "Billings"
        indexes = [
            models.Index(fields=["machine"]),
            models.Index(fields=["machine_owner_user"]),
            models.Index(fields=["Operation_User"]),
            models.Index(fields=["Date"]),
            models.Index(fields=["OperationIsTraining"]),
            models.Index(fields=["OperationIsSolving"]),
            models.Index(fields=["OperationIsMachineCopy"]),
            models.Index(fields=["OperationIsMachineCopyUpdate"]),
            models.Index(fields=["OperationIsMachineExport"]),
        ]

    def __str__(self):
        return f"Billing_{self.id}"

    @classmethod
    def billing_operation_types(cls):
        all_fields = vars(cls)
        operations = [field for field, value in all_fields.items() if field.startswith("OperationIs") and value]
        return operations

    @property
    def get_operation_type(self):
        all_fields = vars(self)
        operation = [
            field for field, value in all_fields.items() if (field.startswith("OperationIs") or field.startswith("Credit")) and value
        ][0]
        operation = operation.split("OperationIs", 1)[1] if operation.startswith("OperationIs") else operation
        return operation

    @property
    def count_lines(self):
        all_fields = vars(self)
        all_count_of_lines = [value for field, value in all_fields.items() if field.startswith("CountOfLines") and value]
        if not all_count_of_lines:
            return "---"  # When there is no line in file
        return all_count_of_lines[0]

    @property
    def operation_cost(self):
        all_fields = vars(self)
        all_count_of_lines = [float(value) for field, value in all_fields.items() if field.find("USD") != -1 and value is not None]
        return sum(all_count_of_lines)
