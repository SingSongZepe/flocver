import typing

from value.strings import *
from value.value import *

def ln(mes: str) -> None:
    print(sslog + LOG_NORMAL + mes)

def lw(mes: str) -> None:
    print(sslog + LOG_WARNING + mes)

def le(mes: str) -> None:
    print(sslog + LOG_ERROR + mes)



