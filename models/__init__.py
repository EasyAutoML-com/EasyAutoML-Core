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
    elif name == 'ConsultingRequest':
        from .consulting import ConsultingRequest
        return ConsultingRequest
    elif name == 'Server':
        from .server import Server
        return Server
    elif name == 'Operation':
        from .billing import Operation
        return Operation
    elif name == 'Credit':
        from .billing import Credit
        return Credit
    elif name == 'Debit':
        from .billing import Debit
        return Debit
    elif name == 'CreditIXIOO':
        from .billing import CreditIXIOO
        return CreditIXIOO
    elif name == 'DebitIXIOO':
        from .billing import DebitIXIOO
        return DebitIXIOO
    elif name == 'Work':
        from .work import Work
        return Work
    elif name == 'Billing':
        from .machine_billing import Billing
        return Billing
    # Models that don't exist yet - return None instead of raising error
    elif name in ['MachineBilling']:
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
    
    # Consulting
    'ConsultingRequest',
    
    # Server
    'Server',
    
    # Billing
    'Operation',
    'Credit',
    'Debit',
    'CreditIXIOO',
    'DebitIXIOO',
    'Billing',
    
    # Work
    'Work',
]
