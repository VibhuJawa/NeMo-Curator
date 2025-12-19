# Stubs for argparse (Python 3.4)

import sys
from collections.abc import Callable, Iterable, Sequence
from typing import IO, Any, TypeAlias, TypeVar

_T = TypeVar("_T")

if sys.version_info >= (3,):
    _Text = str
else:
    _Text: TypeAlias = str | unicode

ONE_OR_MORE = ...  # type: str
OPTIONAL = ...  # type: str
PARSER = ...  # type: str
REMAINDER = ...  # type: str
SUPPRESS = ...  # type: str
ZERO_OR_MORE = ...  # type: str

class ArgumentError(Exception): ...

class ArgumentParser:
    if sys.version_info >= (3, 5):
        def __init__(
            self,
            prog: str | None = ...,
            usage: str | None = ...,
            description: str | None = ...,
            epilog: str | None = ...,
            parents: Sequence[ArgumentParser] = ...,
            formatter_class: type[HelpFormatter] = ...,
            prefix_chars: _Text = ...,
            fromfile_prefix_chars: str | None = ...,
            argument_default: str | None = ...,
            conflict_handler: _Text = ...,
            add_help: bool = ...,
            allow_abbrev: bool = ...,
        ) -> None: ...
    else:
        def __init__(
            self,
            prog: _Text | None = ...,
            usage: _Text | None = ...,
            description: _Text | None = ...,
            epilog: _Text | None = ...,
            parents: Sequence[ArgumentParser] = ...,
            formatter_class: type[HelpFormatter] = ...,
            prefix_chars: _Text = ...,
            fromfile_prefix_chars: _Text | None = ...,
            argument_default: _Text | None = ...,
            conflict_handler: _Text = ...,
            add_help: bool = ...,
        ) -> None: ...

    def add_argument(
        self,
        *name_or_flags: _Text | Sequence[_Text],
        action: _Text | type[Action] = ...,
        nargs: int | _Text = ...,
        const: Any = ...,
        default: Any = ...,
        type: Callable[[str], _T] | FileType = ...,
        choices: Iterable[_T] = ...,
        required: bool = ...,
        help: _Text = ...,
        metavar: _Text | tuple[_Text, ...] = ...,
        dest: _Text = ...,
        version: _Text = ...,
    ) -> None: ...  # weirdly documented
    def parse_args(self, args: Sequence[_Text] | None = ..., namespace: Namespace | None = ...) -> Namespace: ...
    def add_subparsers(
        self,
        title: _Text = ...,
        description: _Text | None = ...,
        prog: _Text = ...,
        parser_class: type[ArgumentParser] = ...,
        action: type[Action] = ...,
        option_string: _Text = ...,
        dest: _Text | None = ...,
        help: _Text | None = ...,
        metavar: _Text | None = ...,
    ) -> _SubParsersAction: ...
    def add_argument_group(self, title: _Text | None = ..., description: _Text | None = ...) -> _ArgumentGroup: ...
    def add_mutually_exclusive_group(self, required: bool = ...) -> _MutuallyExclusiveGroup: ...
    def set_defaults(self, **kwargs: Any) -> None: ...
    def get_default(self, dest: _Text) -> Any: ...
    def print_usage(self, file: IO[str] | None = ...) -> None: ...
    def print_help(self, file: IO[str] | None = ...) -> None: ...
    def format_usage(self) -> str: ...
    def format_help(self) -> str: ...
    def parse_known_args(
        self, args: Sequence[_Text] | None = ..., namespace: Namespace | None = ...
    ) -> tuple[Namespace, list[str]]: ...
    def convert_arg_line_to_args(self, arg_line: _Text) -> list[str]: ...
    def exit(self, status: int = ..., message: _Text | None = ...) -> None: ...
    def error(self, message: _Text) -> None: ...

class HelpFormatter:
    # not documented
    def __init__(
        self, prog: _Text, indent_increment: int = ..., max_help_position: int = ..., width: int | None = ...
    ) -> None: ...

class RawDescriptionHelpFormatter(HelpFormatter): ...
class RawTextHelpFormatter(HelpFormatter): ...
class ArgumentDefaultsHelpFormatter(HelpFormatter): ...

if sys.version_info >= (3,):
    class MetavarTypeHelpFormatter(HelpFormatter): ...

class Action:
    def __init__(
        self,
        option_strings: Sequence[_Text],
        dest: _Text = ...,
        nargs: int | _Text | None = ...,
        const: Any = ...,
        default: Any = ...,
        type: Callable[[str], _T] | FileType | None = ...,
        choices: Iterable[_T] | None = ...,
        required: bool = ...,
        help: _Text | None = ...,
        metavar: _Text | tuple[_Text, ...] = ...,
    ) -> None: ...
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: _Text | Sequence[Any] | None,
        option_string: _Text = ...,
    ) -> None: ...

class Namespace:
    def __getattr__(self, name: _Text) -> Any: ...
    def __setattr__(self, name: _Text, value: Any) -> None: ...

class FileType:
    if sys.version_info >= (3, 4):
        def __init__(
            self, mode: _Text = ..., bufsize: int = ..., encoding: _Text | None = ..., errors: _Text | None = ...
        ) -> None: ...
    elif sys.version_info >= (3,):
        def __init__(self, mode: _Text = ..., bufsize: int = ...) -> None: ...
    else:
        def __init__(self, mode: _Text = ..., bufsize: int | None = ...) -> None: ...

    def __call__(self, string: _Text) -> IO[Any]: ...

class _ArgumentGroup:
    def add_argument(
        self,
        *name_or_flags: _Text | Sequence[_Text],
        action: _Text | type[Action] = ...,
        nargs: int | _Text = ...,
        const: Any = ...,
        default: Any = ...,
        type: Callable[[str], _T] | FileType = ...,
        choices: Iterable[_T] = ...,
        required: bool = ...,
        help: _Text = ...,
        metavar: _Text | tuple[_Text, ...] = ...,
        dest: _Text = ...,
        version: _Text = ...,
    ) -> None: ...
    def add_mutually_exclusive_group(self, required: bool = ...) -> _MutuallyExclusiveGroup: ...

class _MutuallyExclusiveGroup(_ArgumentGroup): ...

class _SubParsersAction:
    # TODO: Type keyword args properly.
    def add_parser(self, name: _Text, **kwargs: Any) -> ArgumentParser: ...

# not documented
class ArgumentTypeError(Exception): ...
