"""Test for Kirsten."""
from typing import Any, Optional


class A(object):
    """Parent test class."""

    def call(self, time, seconds, hours, *args, **kwargs) -> Any:
        """Test call method.

        Args:
            args: Args.
            kwargs: Kwargs.
        """
        pass


class B(A):
    """Child test class."""

    def call(self, number: float = 0, *args, **kwargs) -> None:
        """Test call method extended.

        Args:
            number: Float to print.
            args: Args.
            kwargs: Kwargs.
        """
        print("The number is: ", number)


class C(A):
    """Child test class."""

    def call(self, time, hours, *args, **kwargs) -> None:
        """Test call method extended.

        Args:
            number: Float to print.
            args: Args.
            kwargs: Kwargs.
        """
        print("The number is: ", number)


class D(A):
    """Child test class."""

    def call(self, number: Optional[float] = None, *args, **kwargs) -> None:
        """Test call method extended.

        Args:
            number: Float to print.
            args: Args.
            kwargs: Kwargs.

        Raises:
            ValueError: Cannot print number if no number is given.
        """
        if number is None:
            raise ValueError("Cannot print number if no number is given.")
        print("The number is: ", number)
