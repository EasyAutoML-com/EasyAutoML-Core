from django.db import models
from django.utils.translation import gettext_lazy as _


GRAPH_TYPES = (
    (1, "Scatterplot with multiple semantics"),
    (2, "Timeseries plot with error bands"),
    (3, "Hexbin plot with marginal distributions"),
    (4, "Stacked histogram on a log scale"),
    (5, "Bivariate plot with multiple elements"),
    (6, "Scatterplot with continuous hues and sizes"),
    (7, "Linear regression with marginal distributions"),
    (8, "Grouped boxplots"),
    (9, "Grouped violinplots with split violins"),
    (10, "Plotting a three-way ANOVA"),
    (11, "Plotting large distributions"),
    (12, "Horizontal boxplot with observations"),
    (13, "Horizontal bar plots"),
)
FLOAT_COLUMNS = [3, 5, 7]
DATETIME_COLUMNS = [5]


class Graph(models.Model):
    machine = models.ForeignKey("models.Machine", on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    graph_type = models.IntegerField(default=1, choices=GRAPH_TYPES)
    hue = models.CharField(max_length=255, null=True, default="Light24")
    x = models.CharField(max_length=255, null=True, blank=True)
    y = models.CharField(max_length=255, null=True, blank=True)
    z = models.CharField(max_length=255, null=True, blank=True)
    color = models.CharField(max_length=255, null=True, blank=True)
    animation_frame = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        app_label = 'models'
        db_table = "Graph"
        verbose_name = _("Graph")
        verbose_name_plural = _("Graphs")
        indexes = [
            models.Index(fields=["machine"]),
        ]

    def __str__(self):
        return f"{self.machine} - {self.graph_type}"

    @property
    def graph_title(self):
        return GRAPH_TYPES[self.graph_type - 1][1]

    @property
    def graph_type_str(self):
        return GRAPH_TYPES[self.graph_type - 1][1]
