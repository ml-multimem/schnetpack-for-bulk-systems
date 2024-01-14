import sys
from importlib import import_module
from typing import Union, List, OrderedDict, IO, Any
from pathlib import Path, PurePath

from schnetpack.data.particles import Particles

from ase.io.formats import IOFormat, UnknownFileTypeError, parse_filename, string2index, _iread

NameOrFile = Union[str, PurePath, IO]

class IOFormatExt(IOFormat):
    """Extend IOFormat to accept modules outside of ase.db"""
    def __init__(self, name: str, desc: str, code: str, module_name: str,
                encoding: str = None) -> None:
    
        super().__init__(name, desc, code, module_name, encoding)

    @property
    def module(self):
        try:
            return import_module(self.module_name)
        except ImportError as err:
            raise UnknownFileTypeError(
                f'File format not recognized: {self.name}.  Error: {err}')


def read_particles_db(
    filename: NameOrFile,
    index: Any = None,
    format: str = None,
    parallel: bool = True,
    do_not_split_by_at_sign: bool = False,
    **kwargs
) -> Union[Particles, List[Particles]]:
    """Read Particles object(s) from file.

    filename: str or file
        Name of the file to read from or a file descriptor.
    index: int, slice or str
        The last configuration will be returned by default.  Examples:

            * ``index=0``: first configuration
            * ``index=-2``: second to last
            * ``index=':'`` or ``index=slice(None)``: all
            * ``index='-3:'`` or ``index=slice(-3, None)``: three last
            * ``index='::2'`` or ``index=slice(0, None, 2)``: even
            * ``index='1::2'`` or ``index=slice(1, None, 2)``: odd
    format: str
        Used to specify the file-format.  
        Only SQLite3DatabaseExt currently supported.
    parallel: bool
        Default is to read on master and broadcast to slaves.  Use
        parallel=False to read on all slaves.
    do_not_split_by_at_sign: bool
        If False (default) ``filename`` is splited by at sign ``@``

    Many formats allow on open file-like object to be passed instead
    of ``filename``. In this case the format cannot be auto-decected,
    so the ``format`` argument should be explicitly given."""

    if isinstance(filename, PurePath):
        filename = str(filename)
    if filename == '-':
        filename = sys.stdin
    if isinstance(index, str):
        try:
            index = string2index(index)
        except ValueError:
            pass

    filename, index = parse_filename(filename, index, do_not_split_by_at_sign)
    if index is None:
        index = -1
    format = format # or filetype(filename, read=isinstance(filename, str)) 
                    # Only SQLite3DatabaseExt currently supported

    fmt = IOFormatExt(name = 'db', 
                      desc = 'ASE SQLite Ext database file', 
                      code = '+S', 
                      module_name='schnetpack.md.io.' + 'sqlextdb',
                      encoding=None)

    # Make sure module is importable, since this could also raise an error.
    fmt.module 

    io = fmt

    if isinstance(index, (slice, str)):
        return list(_iread(filename, index, format, io, parallel=parallel,
                        **kwargs))
    else:
        return next(_iread(filename, slice(index, None), format, io,
                        parallel=parallel, **kwargs))
        
