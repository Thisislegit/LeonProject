import collections
import contextlib
import socket

import psycopg2
import psycopg2.extensions
import ray
from psycopg2.extensions import POLL_OK, POLL_READ, POLL_WRITE
from select import select
from . import plans_lib
from . import postgres
from leon_experience import TIME_OUT

# Change these strings so that psycopg2.connect(dsn=dsn_val) works correctly
# for local & remote Postgres.

# JOB/IMDB.
# LOCAL_DSN = "postgres://psycopg:psycopg@localhost/imdb"

from config import read_config
conf = read_config()


database = conf['PostgreSQL']['database']
user = conf['PostgreSQL']['user']
password = conf['PostgreSQL']['password']
host = conf['PostgreSQL']['host']
port = conf['PostgreSQL']['port']
LOCAL_DSN = ""
REMOTE_DSN = ""
leon_port = conf['leon']['Port']
free_size = conf['leon']['free_size']
# TPC-H.
# LOCAL_DSN = "postgres://psycopg:psycopg@localhost/tpch-sf10"
# REMOTE_DSN = "postgres://psycopg:psycopg@localhost/tpch-sf10"

# A simple class holding an execution result.
#   result: a list, outputs from cursor.fetchall().  E.g., the textual outputs
#     from EXPLAIN ANALYZE.
#   has_timeout: bool, set to True iff an execution has run outside of its
#     allocated timeouts; False otherwise (e.g., for non-execution statements
#     such as EXPLAIN).
#   server_ip: str, the private IP address of the Postgres server that
#     generated this Result.
Result = collections.namedtuple(
    'Result',
    ['result', 'has_timeout', 'server_ip'],
)
ExecPlan = collections.namedtuple(
    'ExecPlan', ['plan', 'timeout', 'eq_set', 'cost'])

# ----------------------------------------
#     Psycopg setup
# ----------------------------------------


def wait_select_inter(conn):
    while 1:
        try:
            state = conn.poll()
            if state == POLL_OK:
                break
            elif state == POLL_READ:
                select([conn.fileno()], [], [])
            elif state == POLL_WRITE:
                select([], [conn.fileno()], [])
            else:
                raise conn.OperationalError("bad state from poll: %s" % state)
        except KeyboardInterrupt:
            conn.cancel()
            # the loop will be broken by a server error
            continue


psycopg2.extensions.set_wait_callback(wait_select_inter)


@contextlib.contextmanager
def Cursor():
    """Get a cursor to local Postgres database."""
    # TODO: create the cursor once per worker node.
    conn = psycopg2.connect(database=database, user=user,
                            password=password, host=host, port=port)
    conn.set_client_encoding('UTF8')

    conn.set_session(autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET client_encoding TO 'UTF8';")
            cursor.execute(f"set leon_port={leon_port};")
            cursor.execute(f"set free_size={free_size};")
            # cursor.execute("load 'pg_hint_plan';")

            yield cursor
    finally:
        conn.close()


# ----------------------------------------
#     Postgres execution
# ----------------------------------------


def _SetGeneticOptimizer(flag, cursor):
    # NOTE: DISCARD would erase settings specified via SET commands.  Make sure
    # no DISCARD ALL is called unexpectedly.
    assert cursor is not None
    assert flag in ['on', 'off', 'default'], flag
    cursor.execute('set geqo = {};'.format(flag))
    assert cursor.statusmessage == 'SET'

def ExecuteRemote(sql, verbose=False, geqo_off=False, timeout_ms=None):
    return _ExecuteRemoteImpl.remote(sql, verbose, geqo_off, timeout_ms)


@ray.remote(resources={'pg': 1})
def _ExecuteRemoteImpl(sql, verbose, geqo_off, timeout_ms):
    with Cursor(dsn=REMOTE_DSN) as cursor:
        return Execute(sql, verbose, geqo_off, timeout_ms, cursor)


def Execute(sql, verbose=False, geqo_off=False, timeout_ms=None, cursor=None):
    """Executes a sql statement.

    Returns:
      A pg_executor.Result.
    """
    # if verbose:
    #  print(sql)

    _SetGeneticOptimizer('off' if geqo_off else 'on', cursor)
    if timeout_ms is not None:
        cursor.execute('SET statement_timeout to {}'.format(int(timeout_ms)))
    else:
        # Passing None / setting to 0 means disabling timeout.
        cursor.execute('SET statement_timeout to 0')
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        has_timeout = False
    except Exception as e:
        if isinstance(e, psycopg2.errors.QueryCanceled):
            assert 'canceling statement due to statement timeout' \
                   in str(e).strip(), e
            result = []
            has_timeout = True
        elif isinstance(e, psycopg2.errors.InternalError_):
            print(
                'psycopg2.errors.InternalError_, treating as a' \
                ' timeout'
            )
            print(e)
            result = []
            has_timeout = True
        elif isinstance(e, psycopg2.OperationalError):
            if 'SSL SYSCALL error: EOF detected' in str(e).strip():
                # This usually indicates an expensive query, putting the server
                # into recovery mode.  'cursor' may get closed too.
                print('Treating as a timeout:', e)
                result = []
                has_timeout = True
            else:
                # E.g., psycopg2.OperationalError: FATAL: the database system
                # is in recovery mode
                raise e
        else:
            raise e
    try:
        pass
        # _SetGeneticOptimizer('default', cursor)
    except psycopg2.InterfaceError as e:
        # This could happen if the server is in recovery, due to some expensive
        # queries just crashing the server (see the above exceptions).
        assert 'cursor already closed' in str(e), e
        pass
    ip = socket.gethostbyname(socket.gethostname())
    return Result(result, has_timeout, ip)



@contextlib.contextmanager
def MyCursor(database_port):
    """Get a cursor to local Postgres database."""
    # TODO: create the cursor once per worker node.
    conn = psycopg2.connect(database=database, user=user,
                            password=password, host=host, port=database_port)
    conn.set_client_encoding('UTF8')
    conn.set_session(autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET client_encoding TO 'UTF8';")
            # cursor.execute("load 'pg_hint_plan';")
            cursor.execute(f"set free_size={free_size};")

            yield cursor
    finally:
        conn.close()


def actor_call_leon(actor, item):
    return actor.ActorExecute_leon.remote(item)

@ray.remote
class ActorThatQueries:
    def __init__(self, actor_port, our_port):
        # Initialize and configure your database connection here
        self.port = actor_port
        self.our_port = our_port
        self.TIME_OUT = TIME_OUT
    

    def ActorExecute_leon(self, plan: ExecPlan):
        # Implement the logic to query the database
        node = plan.plan
        timeout = plan.timeout
        explain_str1 = 'explain(verbose, format json, analyze)'
        explain_str2 = 'explain(verbose, format json)'
        sql = node.info['sql_str']

        s1 = str(explain_str1).rstrip() + '\n' + sql
        s2 = str(explain_str2).rstrip() + '\n' + sql

        with MyCursor(self.port) as cursor:
            cursor.execute('SET enable_leon=on;')
            cursor.execute(f"set leon_port={self.our_port};")
            # picknode:{Relid};{Cost};
            cursor.execute(f"SET leon_query_name='picknode:{plan.eq_set};{plan.cost}';") # 第0个plan 0 
            result = Execute(s2, True, True, timeout, cursor).result

        with MyCursor(self.port) as cursor:
            cursor.execute('SET enable_leon=on;')
            cursor.execute(f"set leon_port={self.our_port};")
            cursor.execute(f"SET leon_query_name='picknode:{plan.eq_set};{plan.cost}';") # 第0个plan 0 
            result = Execute(s1, True, True, timeout, cursor).result
        if not result:
            node.info['latency'] = self.TIME_OUT
        else:
            json_dict = result[0][0][0]['Plan']

            def find_actual_total_time(n):
                if 'Total Cost' in n and 'Actual Total Time' in n:
                    if abs(n['Total Cost'] - round(plan.cost * 100) / 100.0) < 0.02:
                        return n['Actual Total Time']
                
                if 'Plans' in n:
                    for sub_plan in n['Plans']:
                        result = find_actual_total_time(sub_plan)
                        if result is not None:
                            return result
                return None

            latency = find_actual_total_time(json_dict)
            if latency is None:
                # Lose node: 
                # Cannot find the node that we picked
                # This is a bug
                # We will count the number of such cases
                return None
            else:
                node.info['latency'] = latency
        return (node, plan.eq_set)