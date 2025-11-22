from pathlib import Path
import sys
import os


class EasyAutoMLDBModels:
    """
    Function is working with django models.
    Main purpose is to provide access to all shared Django models

    DESCRIPTION:    Provide Django object instance to give Access to all table models in Django EasyAutoML.com database
                    Now uses the shared models package for better separation of concerns

    USAGE:  		To work with Django EasyAutoML.com DBModel and it's tables. Writing or reading database data.
                        More simple way than writing MySQL queries.
    """

    def __init__(self):
        # Use centralized models from /models/ directory
        project_root = Path(__file__).absolute().parent.parent
        models_path = str(project_root / "models")
        
        if models_path not in sys.path:
            sys.path.append(models_path)

        import django
        
        # Set Django settings for standalone backend
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
        
        # Only call django.setup() if Django hasn't been set up yet
        try:
            # Check if Django apps are already configured
            from django.apps import apps
            if not apps.ready:
                django.setup()
        except (ImportError, RuntimeError) as e:
            # If Django is currently being set up (reentrant call), skip it
            if "populate() isn't reentrant" in str(e):
                pass  # Django is already being set up, continue with imports
            else:
                # If Django isn't configured yet, try to set it up
                try:
                    django.setup()
                except RuntimeError as setup_error:
                    if "populate() isn't reentrant" in str(setup_error):
                        pass  # Django is already being set up, continue with imports
                    else:
                        raise

        # Import database connections only after Django setup
        from django.db import connection, connections
        
        # ============================================================================
        # COMPATIBILITY LAYER: Core Version vs Full WWW Version
        # ============================================================================
        # This section handles differences between two deployment versions:
        # - CORE VERSION: Minimal set of models (User, Team, Machine, etc.)
        # - FULL WWW VERSION: Complete set including Server, Operation, Billing, etc.
        #
        # Strategy: Try to import each model. If it doesn't exist in the current
        # version, set it to None. This allows the same codebase to work in both
        # environments without modification.
        # ============================================================================
        
        # Import CORE models (always available in both versions)
        from models import (
            User, Team, Machine,
            EasyAutoMLLogger, Graph, NNModel, DataLinesOperation, 
            MachineTableLockWrite, EncDecConfiguration
        )
        
        # Import WWW-ONLY models (may not exist in core version)
        # These models are only available in the full WWW deployment
        # Try to import each one individually, fall back to None if not available
        Server = None
        Operation = None
        ConsultingRequest = None
        Work = None
        Billing = None
        MachineBilling = None
        
        try:
            from models import Server
        except ImportError:
            pass  # Server model not available in core version
        
        try:
            from models import Operation
        except ImportError:
            pass  # Operation model not available in core version
        
        try:
            from models import ConsultingRequest
        except ImportError:
            pass  # ConsultingRequest model not available in core version
        
        try:
            from models import Work
        except ImportError:
            pass  # Work model not available in core version
        
        try:
            from models import Billing
        except ImportError:
            pass  # Billing model not available in core version
        
        try:
            from models import MachineBilling
        except ImportError:
            pass  # MachineBilling model not available in core version

        # Store database connections - delay cursor creation to avoid hanging
        self.connections = connections
        self.cursor = None  # Lazy initialize on first use
        
        # ============================================================================
        # Register all models as instance attributes for backward compatibility
        # ============================================================================
        # This allows code to access models via: db_models.User, db_models.Machine, etc.
        # Models set to None (from core version) can be checked: if db_models.Server: ...
        # ============================================================================
        
        # CORE models (always available)
        self.User = User
        self.Machine = Machine
        self.MachineTableLockWrite = MachineTableLockWrite
        self.DataLinesOperation = DataLinesOperation
        self.EasyAutoMLLogger = EasyAutoMLLogger
        self.Team = Team
        self.Graph = Graph
        self.NNModel = NNModel
        self.EncDecConfiguration = EncDecConfiguration
        
        # WWW-ONLY models (None in core version, actual model class in WWW version)
        self.Server = Server
        self.Operation = Operation
        self.Work = Work
        self.ConsultingRequest = ConsultingRequest
        self.Billing = Billing
        self.MachineBilling = MachineBilling
        
        # Aliases for backward compatibility with legacy code
        self.Consulting = ConsultingRequest  # Alias: ConsultingRequest
        self.Logger = EasyAutoMLLogger  # Alias: EasyAutoMLLogger
    
    @property
    def logger(self):
        """Return a logger instance"""
        return self.EasyAutoMLLogger()
    
    def get_cursor(self):
        """Lazily get database cursor on first use"""
        if self.cursor is None:
            from django.db import connection
            try:
                self.cursor = connection.cursor()
            except Exception as e:
                # If connection fails, log and return None
                print(f"Warning: Could not get database cursor: {e}")
                return None
        return self.cursor


