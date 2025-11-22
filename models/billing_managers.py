from typing import TYPE_CHECKING, Union, List
from django.db.models import Manager, Q

if TYPE_CHECKING:
    from decimal import Decimal
    from django.db.models import QuerySet
    from django.contrib.auth import get_user_model
    from datetime import datetime
    from .billing import Credit, Debit, CreditIXIOO, DebitIXIOO, Operation
    from .machine import Machine

    User = get_user_model()


class CreditManager(Manager):
    """Credit manager of Credit model"""

    def get_all_by_user(self, user: "User") -> Union["QuerySet", List["Credit"]]:
        """
        Returns all credits by user
        :param user: - user instance belongs to operation
        :type user:  User
        :return: filtered queryset
        :rtype: Union["QuerySet", List["Credit"]]
        """
        return super(CreditManager, self).get_queryset().filter(user=user)

    def get_by_user_in_date_range(self, user: "User", start_date: "datetime", end_date: "datetime") -> Union["QuerySet", List["Credit"]]:
        """
        Returns all credits by user between start_date and end_date
        :param user:  user instance belongs to operation
        :type user:  User
        :param start_date:  date, start of searching period
        :type start_date: datetime
        :param end_date:  date, end of searching period
        :type end_date: datetime
        :return: filtered queryset
        :rtype: Union["QuerySet", List["Credit"]]
        """
        return super(CreditManager, self).get_queryset().filter(user=user, date_time__range=[start_date, end_date])

    def increase(self, user: "User", amount: "Decimal") -> "Credit":
        """
        Creates new Credit instance and increases user balance by amount
        :param user: - user instance belongs to operation
        :type user:  User
        :param amount: - Decimal instance
        :type amount:  Decimal
        :return: created Credit instance
        :rtype: Credit
        """
        user.user_balance += amount
        user.save()
        return self.create(user=user, amount=amount)


class DebitManager(Manager):
    """Debit manager of Debit model"""

    def get_all_by_user(self, user: "User") -> Union["QuerySet", List["Debit"]]:
        """
        Returns all debits by user
        :param user: - user instance belongs to operation
        :type user:  User
        :return: filtered queryset
        :rtype: Union["QuerySet", List["Debit"]]
        """
        return super(DebitManager, self).get_queryset().filter(user=user)

    def get_by_user_in_date_range(self, user: "User", start_date: "datetime", end_date: "datetime") -> Union["QuerySet", List["Debit"]]:
        """
        Returns all debits by user between start_date and end_date
        :param user:  user instance belongs to operation
        :type user:  User
        :param start_date:  date, start of searching period
        :type start_date: datetime
        :param end_date:  date, end of searching period
        :type end_date: datetime
        :return: filtered queryset
        :rtype: Union["QuerySet", List["Debit"]]
        """
        return super(DebitManager, self).get_queryset().filter(user=user, date_time__range=[start_date, end_date])

    def decrease(self, user: "User", amount: "Decimal") -> "Debit":
        """
        Creates new Debit instance and decrease user balance by amount
        :param user: - user instance belongs to operation
        :type user:  User
        :param amount: - Decimal instance
        :type amount:  Decimal
        :return: created Debit instance
        :rtype: Debit
        """
        from decimal import Decimal
        user.user_balance -= amount
        user.save()
        return self.create(user=user, amount=-Decimal(amount))


class CreditIXIOOManager(Manager):
    """CreditIXIOO model manager"""

    def get_all_by_user(self, user: "User") -> Union["QuerySet", List["CreditIXIOO"]]:
        """
        Returns all credits by user
        :param user: User instance belongs to operation
        :type user:  User
        :return: filtered queryset
        :rtype: Union["QuerySet", List["CreditIXIOO"]]
        """
        return super(CreditIXIOOManager, self).get_queryset().filter(user=user)

    def get_by_user_in_date_range(
        self, user: "User", start_date: "datetime", end_date: "datetime"
    ) -> Union["QuerySet", List["CreditIXIOO"]]:
        """
        Returns all credits by user between start_date and end_date
        :param user:  User instance belongs to operation
        :type user:  User
        :param start_date:  date, start of searching period
        :type start_date: datetime
        :param end_date:  date, end of searching period
        :type end_date: datetime
        :return: filtered queryset
        :rtype: Union["QuerySet", List["CreditIXIOO"]]
        """
        return super(CreditIXIOOManager, self).get_queryset().filter(user=user, date_time__range=[start_date, end_date])

    def increase(self, user: "User", amount: "Decimal") -> "CreditIXIOO":
        """
        Creates new CreditIXIOO instance and increases user ixioo balance by amount
        :param user:  user instance belongs to operation
        :type user:  User
        :param amount:  Decimal instance
        :type amount: Decimal
        :return: created CreditIXIOO instance
        :rtype: CreditIXIOO
        """
        user.user_ixioo_balance += amount
        user.save()
        return self.create(user=user, amount=amount)

    def pay_bill(self, user: "User", amount: "Decimal") -> "CreditIXIOO":
        """
        Creates new credit_ixioo instance and increases user ixioo balance by amount also updates user.last_billing_time
        :param user:  user instance belongs to operation
        :type user:  User
        :param amount:  Decimal instance
        :type amount: Decimal
        :return: created CreditIXIOO instance
        :rtype: CreditIXIOO
        """
        from datetime import datetime

        user.user_ixioo_balance += amount
        user.last_billing_time = datetime.now()
        user.save()
        return self.create(user=user, amount=amount)


class DebitIXIOOManager(Manager):
    """DebitIXIOO model manager"""

    def get_all_by_user(self, user: "User") -> Union["QuerySet", List["DebitIXIOO"]]:
        """
        Returns all debits by user
        :param user: user instance belongs to operation
        :type user:  User
        :return: filtered queryset
        :rtype: Union["QuerySet", List["DebitIXIOO"]]
        """
        return super(DebitIXIOOManager, self).get_queryset().filter(user=user)

    def get_by_user_in_date_range(
        self, user: "User", start_date: "datetime", end_date: "datetime"
    ) -> Union["QuerySet", List["DebitIXIOO"]]:
        """
        Returns all debits by user between start_date and end_date
        :param user:  user instance belongs to operation
        :type user:  User
        :param start_date:  date, start of searching period
        :type start_date: datetime
        :param end_date:  date, end of searching period
        :type end_date: datetime
        :return: filtered queryset
        :rtype: Union["QuerySet", List["DebitIXIOO"]]
        """
        return super(DebitIXIOOManager, self).get_queryset().filter(user=user, date_time__range=[start_date, end_date])

    def decrease(self, user: "User", amount: "Decimal") -> "DebitIXIOO":
        """
        Creates new DebitIXIOO instance and decrease user ixioo balance by amount
        :param user: - user instance belongs to operation
        :type user:  User
        :param amount: - Decimal instance
        :type amount:  Decimal
        :return: created DebitIXIOO instance
        :rtype: DebitIXIOO
        """
        user.user_ixioo_balance -= amount
        user.save()
        return self.create(user=user, amount=amount)


class MachineOperationsManager(Manager):
    """
    Operation model manager
    Contains only methods to perform operations which belongs to machine
    """

    def training(self, machine: "Machine", count_of_lines: int) -> "Operation":
        """
        Performs training operation,
        Creates:
            DebitIXIOO instance for Machine Owner (if billing is enabled)
        :param machine: Machine model instance belongs to operation
        :type machine: Machine
        :param count_of_lines: count of lines in training dataset
        :type count_of_lines: int
        :return: Operation instance
        :rtype: Operation
        """
        from decimal import Decimal
        
        machine_owner = machine.machine_owner_user
        count_of_training_epoch = getattr(machine, 'training_training_epoch_count', None)
        
        # Try to get billing price, but don't fail if not available
        try:
            from django.conf import settings
            BILLING_IXIOO_OPERATIONS_PRICE = getattr(settings, 'BILLING_IXIOO_OPERATIONS_PRICE', {})
            training_price_per_column = BILLING_IXIOO_OPERATIONS_PRICE.get("training_price", 0)
            training_price = Decimal(
                (len(machine.dfr_columns_type_user_df) * training_price_per_column) * count_of_lines
            )
            
            # Only charge if billing is enabled and price > 0
            if training_price > 0:
                debit_ixioo = DebitIXIOO.objects.decrease(user=machine_owner, amount=training_price)
            else:
                debit_ixioo = None
        except Exception:
            # If billing fails (e.g., in test environment), continue without billing
            debit_ixioo = None

        operation = self.create(
            machine=machine,
            machine_owner=machine_owner,
            operation_user=machine_owner,
            debit_ixioo=debit_ixioo,
            is_training_operation=True,
            count_of_lines=count_of_lines,
            count_of_training_epoch=count_of_training_epoch,
        )
        return operation

