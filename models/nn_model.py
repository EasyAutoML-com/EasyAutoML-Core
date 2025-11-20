from sys import getsizeof
from django.db import models


class NNModel(models.Model):
    nn_model = models.BinaryField(null=True, blank=True, max_length=750000000)

    class Meta:
        app_label = 'models'
        verbose_name = "NN Model"
        verbose_name_plural = "NN Models"
        db_table = "machine_nnmodel"

    def __str__(self):
        return f"NNModel \t size: {getsizeof(self.nn_model) if self.nn_model else 0}"
