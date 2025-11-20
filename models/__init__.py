# Shared Django Models Package
# This package contains all Django models that are shared between
# the backend AI/ML components and the web application.

# Default app config for Django
default_app_config = 'models.apps.ModelsConfig'

# NOTE: User model will be imported lazily when accessed via __getattr__
# Since models app comes before django.contrib.auth in INSTALLED_APPS,
# the User model will be imported in apps.py ready() method before
# django.contrib.auth tries to access it
# Lazy imports to avoid circular dependencies during Django setup
def __getattr__(name):
    """Lazy import models to avoid circular dependencies"""
    if name == 'User':
        from .user import User
        return User
    elif name == 'UserManager':
        from .user import UserManager
        return UserManager
    elif name == 'Team':
        from .team import Team
        return Team
    elif name == 'Machine':
        from .machine import Machine
        return Machine
    elif name == 'EasyAutoMLLogger':
        from .logger import EasyAutoMLLogger
        return EasyAutoMLLogger
    elif name == 'Graph':
        from .graph import Graph
        return Graph
    elif name == 'NNModel':
        from .nn_model import NNModel
        return NNModel
    elif name == 'DataLinesOperation':
        from .data_lines_operation import DataLinesOperation
        return DataLinesOperation
    elif name == 'MachineTableLockWrite':
        from .machine_table_lock_write import MachineTableLockWrite
        return MachineTableLockWrite
    elif name == 'EncDecConfiguration':
        from .encdec_configuration import EncDecConfiguration
        return EncDecConfiguration
    # Models that don't exist yet - return None instead of raising error
    elif name in ['Server', 'Operation', 'Credit', 'Debit', 'CreditIXIOO', 'DebitIXIOO', 
                  'ConsultingRequest', 'Work', 'MachineBilling', 'Billing']:
        return None
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # User management
    'User',
    'UserManager',
    'Team',
    
    # Infrastructure
    'EasyAutoMLLogger',
    
    # Machine Learning
    'Machine',
    'Graph',
    'NNModel',
    'DataLinesOperation',
    'MachineTableLockWrite',
    'EncDecConfiguration',
]
