"""SQLite3 backend.

Versions:

1) Added 3 more columns.
2) Changed "user" to "username".
3) Now adding keys to keyword table and added an "information" table containing
   a version number.
4) Got rid of keywords.
5) Add fmax, smax, mass, volume, charge
6) Use REAL for magmom and drop possibility for non-collinear spin
7) Volume can be None
8) Added name='metadata' row to "information" table
9) Row data is now stored in binary format.
16/02/23 ER
-) Added 2 more columns for particle types and molecule membership
23/02/23
-) Added 4 more columns for bonds and angles lists and types
"""

from ase.db.sqlite import *

from schnetpack.data.particlesrow import ParticlesRow

import json
import numbers
import os
import sqlite3

import numpy as np

from ase.data import atomic_numbers
from ase.db.core import (Database, ops, now, invop)

init_statements = [
    """CREATE TABLE systems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- ID's, timestamps and user name
    unique_id TEXT UNIQUE,
    ctime REAL,
    mtime REAL,
    username TEXT,
    numbers BLOB,  -- stuff that defines an Atoms object
    positions BLOB,
    cell BLOB,
    pbc INTEGER,
    initial_magmoms BLOB,
    initial_charges BLOB,
    masses BLOB,
    tags BLOB,
    momenta BLOB,
    particle_types BLOB,
    molecule_ids BLOB,
    bonds_list BLOB,
    angles_list BLOB,
    bond_types BLOB,
    angle_types BLOB,
    constraints TEXT,  -- constraints and calculator
    calculator TEXT,
    calculator_parameters TEXT,
    energy REAL,  -- calculated properties
    free_energy REAL,
    forces BLOB,
    stress BLOB,
    dipole BLOB,
    magmoms BLOB,
    magmom REAL,
    charges BLOB,
    key_value_pairs TEXT,  -- key-value pairs and data as json
    data BLOB,
    natoms INTEGER,  -- stuff for making queries faster
    fmax REAL,
    smax REAL,
    volume REAL,
    mass REAL,
    charge REAL)""",

    """CREATE TABLE species (
    Z INTEGER,
    n INTEGER,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE keys (
    key TEXT,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE text_key_values (
    key TEXT,
    value TEXT,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE number_key_values (
    key TEXT,
    value REAL,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE information (
    name TEXT,
    value TEXT)""",

    "INSERT INTO information VALUES ('version', '{}')".format(VERSION)]


class SQLite3DatabaseExt(SQLite3Database):
    type = 'db'
    initialized = False
    _allow_reading_old_format = False
    default = 'NULL'  # used for autoincrement id
    connection = None
    version = None
    columnnames = [line.split()[0].lstrip()
                   for line in init_statements[0].splitlines()[1:]]

    def _initialize(self, con):
        if self.initialized:
            return

        self._metadata = {}

        cur = con.execute(
            'SELECT COUNT(*) FROM sqlite_master WHERE name="systems"')

        if cur.fetchone()[0] == 0:
            for statement in init_statements:
                con.execute(statement)
            if self.create_indices:
                for statement in index_statements:
                    con.execute(statement)
            con.commit()
            self.version = VERSION
        else:
            cur = con.execute(
                'SELECT COUNT(*) FROM sqlite_master WHERE name="user_index"')
            if cur.fetchone()[0] == 1:
                # Old version with "user" instead of "username" column
                self.version = 1
            else:
                try:
                    cur = con.execute(
                        'SELECT value FROM information WHERE name="version"')
                except sqlite3.OperationalError:
                    self.version = 2
                else:
                    self.version = int(cur.fetchone()[0])

                cur = con.execute(
                    'SELECT value FROM information WHERE name="metadata"')
                results = cur.fetchall()
                if results:
                    self._metadata = json.loads(results[0][0])

        if self.version > VERSION:
            raise IOError('Can not read new ase.db format '
                          '(version {}).  Please update to latest ASE.'
                          .format(self.version))
        if self.version < 5 and not self._allow_reading_old_format:
            raise IOError('Please convert to new format. ' +
                          'Use: python -m ase.db.convert ' + self.filename)

        self.initialized = True

    def _write(self, atoms, key_value_pairs, data, id):
        ext_tables = key_value_pairs.pop("external_tables", {})
        Database._write(self, atoms, key_value_pairs, data)

        mtime = now()

        encode = self.encode
        blob = self.blob

        if not isinstance(atoms, ParticlesRow):
            row = ParticlesRow(atoms)
            row.ctime = mtime
            row.user = os.getenv('USER')
        else:
            row = atoms
            # Extract the external tables from AtomsRow
            names = self._get_external_table_names()
            for name in names:
                new_table = row.get(name, {})
                if new_table:
                    ext_tables[name] = new_table

        if not id and not key_value_pairs and not ext_tables:
            key_value_pairs = row.key_value_pairs

        for k, v in ext_tables.items():
            dtype = self._guess_type(v)
            self._create_table_if_not_exists(k, dtype)

        constraints = row._constraints
        if constraints:
            if isinstance(constraints, list):
                constraints = encode(constraints)
        else:
            constraints = None

        values = (row.unique_id,                                # 0
                  row.ctime,                                    # 1
                  mtime,                                        # 2
                  row.user,                                     # 3   
                  blob(row.numbers),                            # 4
                  blob(row.positions),                          # 5
                  blob(row.cell),                               # 6
                  int(np.dot(row.pbc, [1, 2, 4])),              # 7
                  blob(row.get('initial_magmoms')),             # 8
                  blob(row.get('initial_charges')),             # 9
                  blob(row.get('masses')),                      # 10
                  blob(row.get('tags')),                        # 11   
                  blob(row.get('momenta')),                     # 12
                  blob(row.particle_types),                     # 13
                  blob(row.molecule_ids),                       # 14
                  blob(row.bonds_list),                         # 15
                  blob(row.angles_list),                        # 16
                  blob(row.bond_types),                         # 17
                  blob(row.angle_types),                        # 18
                  constraints)                                  # 19

        if 'calculator' in row:
            values += (row.calculator, encode(row.calculator_parameters))
        else:
            values += (None, None)  # 20, 21 

        if not data:
            data = row._data

        with self.managed_connection() as con:
            if not isinstance(data, (str, bytes)):
                data = encode(data, binary=self.version >= 9)

            values += (row.get('energy'),                           # 22
                       row.get('free_energy'),                      # 23
                       blob(row.get('forces')),                     # 24
                       blob(row.get('stress')),                     # 25
                       blob(row.get('dipole')),                     # 26
                       blob(row.get('magmoms')),                    # 27
                       row.get('magmom'),                           # 28
                       blob(row.get('charges')),                    # 29
                       encode(key_value_pairs),                     # 30
                       data,                                        # 31
                       len(row.numbers),                            # 32
                       float_if_not_none(row.get('fmax')),          # 33
                       float_if_not_none(row.get('smax')),          # 34
                       float_if_not_none(row.get('volume')),        # 35
                       float(row.mass),                             # 36
                       float(row.charge),)                          # 37
                                            

            cur = con.cursor()
            if id is None:
                q = self.default + ', ' + ', '.join('?' * len(values))
                cur.execute('INSERT INTO systems VALUES ({})'.format(q),
                            values)
                id = self.get_last_id(cur)
            else:
                self._delete(cur, [id], ['keys', 'text_key_values',
                                         'number_key_values', 'species'])
                q = ', '.join(name + '=?' for name in self.columnnames[1:])
                cur.execute('UPDATE systems SET {} WHERE id=?'.format(q),
                            values + (id,))

            count = row.count_atoms()
            if count:
                species = [(atomic_numbers[symbol], n, id)
                           for symbol, n in count.items()]
                cur.executemany('INSERT INTO species VALUES (?, ?, ?)',
                                species)

            text_key_values = []
            number_key_values = []
            for key, value in key_value_pairs.items():
                if isinstance(value, (numbers.Real, np.bool_)):
                    number_key_values.append([key, float(value), id])
                else:
                    assert isinstance(value, str)
                    text_key_values.append([key, value, id])

            cur.executemany('INSERT INTO text_key_values VALUES (?, ?, ?)',
                            text_key_values)
            cur.executemany('INSERT INTO number_key_values VALUES (?, ?, ?)',
                            number_key_values)
            cur.executemany('INSERT INTO keys VALUES (?, ?)',
                            [(key, id) for key in key_value_pairs])

            # Insert entries in the valid tables
            for tabname in ext_tables.keys():
                entries = ext_tables[tabname]
                entries['id'] = id
                self._insert_in_external_table(
                    cur, name=tabname, entries=ext_tables[tabname])

        return id

    def _convert_tuple_to_row(self, values):
        deblob = self.deblob
        decode = self.decode

        #values = self._old2new(values) #Ignored because I rechecked all the numbering
        dct = {'id': values[0],
               'unique_id': values[1],
               'ctime': values[2],
               'mtime': values[3],
               'user': values[4],
               'numbers': deblob(values[5], np.int32),
               'positions': deblob(values[6], shape=(-1, 3)),
               'cell': deblob(values[7], shape=(3, 3))}


        if values[8] is not None:
            dct['pbc'] = (values[8] & np.array([1, 2, 4])).astype(bool)
        if values[9] is not None:
            dct['initial_magmoms'] = deblob(values[9])
        if values[10] is not None:
            dct['initial_charges'] = deblob(values[10])
        if values[11] is not None:
            dct['masses'] = deblob(values[11])
        if values[12] is not None:
            dct['tags'] = deblob(values[12], np.int32)
        if values[13] is not None:
            dct['momenta'] = deblob(values[13], shape=(-1, 3))

        if values[14] is not None:
            dct['particle_types'] = deblob(values[14], np.int32)
        if values[15] is not None:
            dct['molecule_ids'] = deblob(values[15], np.int32)
        if values[16] is not None:
            dct['bonds_list'] = deblob(values[16], np.int32)
        if values[17] is not None:
            dct['angles_list'] = deblob(values[17], np.int32)
        if values[18] is not None:
            dct['bond_types'] = deblob(values[18], np.int32)
        if values[19] is not None:
            dct['angle_types'] = deblob(values[19], np.int32)

        if values[20] is not None:
            dct['constraints'] = values[20]
        if values[21] is not None:
            dct['calculator'] = values[21]
        if values[22] is not None:
            dct['calculator_parameters'] = decode(values[22])
        if values[23] is not None:
            dct['energy'] = values[23]
        if values[24] is not None:
            dct['free_energy'] = values[24]
        if values[25] is not None:
            dct['forces'] = deblob(values[25], shape=(-1, 3))
        if values[26] is not None:
            dct['stress'] = deblob(values[26])
        if values[27] is not None:
            dct['dipole'] = deblob(values[27])
        if values[28] is not None:
            dct['magmoms'] = deblob(values[28])
        if values[29] is not None:
            dct['magmom'] = values[29]
        if values[30] is not None:
            dct['charges'] = deblob(values[30])
        if values[31] != '{}':
            dct['key_value_pairs'] = decode(values[31])

        if len(values) >= 33 and values[32] != 'null':
            dct['data'] = decode(values[32], lazy=True)

        # Now we need to update with info from the external tables
        external_tab = self._get_external_table_names()
        tables = {}
        for tab in external_tab:
            row = self._read_external_table(tab, dct["id"])
            tables[tab] = row

        dct.update(tables)
        return ParticlesRow(dct)

    def create_select_statement(self, keys, cmps,
                                sort=None, order=None, sort_table=None,
                                what='systems.*'):
        tables = ['systems']
        where = []
        args = []
        for key in keys:
            if key == 'forces':
                where.append('systems.fmax IS NOT NULL')
            elif key == 'strain':
                where.append('systems.smax IS NOT NULL')
            elif key in ['energy', 'fmax', 'smax',
                         'constraints', 'calculator']:
                where.append('systems.{} IS NOT NULL'.format(key))
            else:
                if '-' not in key:
                    q = 'systems.id in (select id from keys where key=?)'
                else:
                    key = key.replace('-', '')
                    q = 'systems.id not in (select id from keys where key=?)'
                where.append(q)
                args.append(key)

        # Special handling of "H=0" and "H<2" type of selections:
        bad = {}
        for key, op, value in cmps:
            if isinstance(key, int):
                bad[key] = bad.get(key, True) and ops[op](0, value)

        for key, op, value in cmps:
            if key in ['id', 'energy', 'magmom', 'ctime', 'user',
                       'calculator', 'natoms', 'pbc', 'unique_id',
                       'fmax', 'smax', 'volume', 'mass', 'charge',
                       'particle_type', 'molecule_id' ]:  # <<< Added singular properties here, which are found in Particle
                       # Do we need also the properties of Particles? 
                if key == 'user':
                    key = 'username'
                elif key == 'pbc':
                    assert op in ['=', '!=']
                    value = int(np.dot([x == 'T' for x in value], [1, 2, 4]))
                elif key == 'magmom':
                    assert self.version >= 6, 'Update your db-file'
                where.append('systems.{}{}?'.format(key, op))
                args.append(value)
            elif isinstance(key, int):
                if self.type == 'postgresql':
                    where.append(
                        'cardinality(array_positions(' +
                        'numbers::int[], ?)){}?'.format(op))
                    args += [key, value]
                else:
                    if bad[key]:
                        where.append(
                            'systems.id not in (select id from species ' +
                            'where Z=? and n{}?)'.format(invop[op]))
                        args += [key, value]
                    else:
                        where.append('systems.id in (select id from species ' +
                                     'where Z=? and n{}?)'.format(op))
                        args += [key, value]

            elif self.type == 'postgresql':
                jsonop = '->'
                if isinstance(value, str):
                    jsonop = '->>'
                elif isinstance(value, bool):
                    jsonop = '->>'
                    value = str(value).lower()
                where.append("systems.key_value_pairs {} '{}'{}?"
                             .format(jsonop, key, op))
                args.append(str(value))

            elif isinstance(value, str):
                where.append('systems.id in (select id from text_key_values ' +
                             'where key=? and value{}?)'.format(op))
                args += [key, value]
            else:
                where.append(
                    'systems.id in (select id from number_key_values ' +
                    'where key=? and value{}?)'.format(op))
                args += [key, float(value)]

        if sort:
            if sort_table != 'systems':
                tables.append('{} AS sort_table'.format(sort_table))
                where.append('systems.id=sort_table.id AND '
                             'sort_table.key=?')
                args.append(sort)
                sort_table = 'sort_table'
                sort = 'value'

        sql = 'SELECT {} FROM\n  '.format(what) + ', '.join(tables)
        if where:
            sql += '\n  WHERE\n  ' + ' AND\n  '.join(where)
        if sort:
            # XXX use "?" instead of "{}"
            sql += '\nORDER BY {0}.{1} IS NULL, {0}.{1} {2}'.format(
                sort_table, sort, order)

        return sql, args

    def _select(self, keys, cmps, explain=False, verbosity=0,
                limit=None, offset=0, sort=None, include_data=True,
                columns='all'):

        values = np.array([None for i in range(33)])
        values[31] = '{}'
        values[32] = 'null'

        if columns == 'all':
            columnindex = list(range(32))
        else:
            columnindex = [c for c in range(0, 32)
                           if self.columnnames[c] in columns]
        if include_data:
            columnindex.append(32)

        if sort:
            if sort[0] == '-':
                order = 'DESC'
                sort = sort[1:]
            else:
                order = 'ASC'
            if sort in ['id', 'energy', 'username', 'calculator',
                        'ctime', 'mtime', 'magmom', 'pbc',
                        'fmax', 'smax', 'volume', 'mass', 'charge', 'natoms',
                        'particle_type', 'molecule_id' ]:  # <<< Added singular properties here, which are found in Particle
                       # Do we need also the properties of Particles? 

                sort_table = 'systems'
            else:
                for dct in self._select(keys + [sort], cmps=[], limit=1,
                                        include_data=False,
                                        columns=['key_value_pairs']):
                    if isinstance(dct['key_value_pairs'][sort], str):
                        sort_table = 'text_key_values'
                    else:
                        sort_table = 'number_key_values'
                    break
                else:
                    # No rows.  Just pick a table:
                    sort_table = 'number_key_values'

        else:
            order = None
            sort_table = None

        what = ', '.join('systems.' + name
                         for name in
                         np.array(self.columnnames)[np.array(columnindex)])

        sql, args = self.create_select_statement(keys, cmps, sort, order,
                                                 sort_table, what)

        if explain:
            sql = 'EXPLAIN QUERY PLAN ' + sql

        if limit:
            sql += '\nLIMIT {0}'.format(limit)

        if offset:
            sql += self.get_offset_string(offset, limit=limit)

        if verbosity == 2:
            print(sql, args)

        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(sql, args)
            if explain:
                for row in cur.fetchall():
                    yield {'explain': row}
            else:
                n = 0
                for shortvalues in cur.fetchall():
                    values[columnindex] = shortvalues
                    yield self._convert_tuple_to_row(tuple(values))
                    n += 1

                if sort and sort_table != 'systems':
                    # Yield rows without sort key last:
                    if limit is not None:
                        if n == limit:
                            return
                        limit -= n
                    for row in self._select(keys + ['-' + sort], cmps,
                                            limit=limit, offset=offset,
                                            include_data=include_data,
                                            columns=columns):
                        yield row
