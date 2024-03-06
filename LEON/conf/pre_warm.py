import psycopg2
import configparser
import time

def read_config(section):    
    config = configparser.ConfigParser()
    config.read('leon.cfg')
    return config[section]

def prewarm_pg(port):
    conf = read_config('PostgreSQL')
    database = conf['database']
    user = conf['user']
    password = conf['password']
    host = conf['host']
    # port = int(conf['port'])
    print("PostgreSQL Start PreWarming:")
    start = time.time()
    with psycopg2.connect(database=database, user=user, password=password, host=host, port=port) as conn:
        with conn.cursor() as cur:
            cur.execute("load 'pg_prewarm'")
    
            sql = \
            "select pg_prewarm('aka_name', 'buffer', 'main'); \
                    select pg_prewarm('aka_title', 'buffer', 'main'); \
                    select pg_prewarm('cast_info', 'buffer', 'main'); \
                    select pg_prewarm('char_name', 'buffer', 'main'); \
                    select pg_prewarm('comp_cast_type', 'buffer', 'main'); \
                    select pg_prewarm('company_name', 'buffer', 'main'); \
                    select pg_prewarm('company_type', 'buffer', 'main'); \
                    select pg_prewarm('complete_cast', 'buffer', 'main'); \
                    select pg_prewarm('info_type', 'buffer', 'main'); \
                    select pg_prewarm('keyword', 'buffer', 'main'); \
                    select pg_prewarm('kind_type', 'buffer', 'main'); \
                    select pg_prewarm('link_type', 'buffer', 'main'); \
                    select pg_prewarm('movie_companies', 'buffer', 'main'); \
                    select pg_prewarm('movie_info', 'buffer', 'main'); \
                    select pg_prewarm('movie_info_idx', 'buffer', 'main'); \
                    select pg_prewarm('movie_keyword', 'buffer', 'main'); \
                    select pg_prewarm('movie_link', 'buffer', 'main'); \
                    select pg_prewarm('name', 'buffer', 'main'); \
                    select pg_prewarm('person_info', 'buffer', 'main'); \
                    select pg_prewarm('role_type', 'buffer', 'main'); \
                    select pg_prewarm('title', 'buffer', 'main'); \
                    select pg_prewarm('company_id_movie_companies', 'buffer', 'main'); \
                    select pg_prewarm('company_type_id_movie_companies', 'buffer', 'main'); \
                    select pg_prewarm('info_type_id_movie_info_idx', 'buffer', 'main'); \
                    select pg_prewarm('info_type_id_movie_info', 'buffer', 'main'); \
                    select pg_prewarm('info_type_id_person_info', 'buffer', 'main'); \
                    select pg_prewarm('keyword_id_movie_keyword', 'buffer', 'main'); \
                    select pg_prewarm('kind_id_aka_title', 'buffer', 'main'); \
                    select pg_prewarm('kind_id_title', 'buffer', 'main'); \
                    select pg_prewarm('linked_movie_id_movie_link', 'buffer', 'main'); \
                    select pg_prewarm('link_type_id_movie_link', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_aka_title', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_cast_info', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_complete_cast', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_movie_companies', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_movie_info_idx', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_movie_keyword', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_movie_link', 'buffer', 'main'); \
                    select pg_prewarm('movie_id_movie_info', 'buffer', 'main'); \
                    select pg_prewarm('person_id_aka_name', 'buffer', 'main'); \
                    select pg_prewarm('person_id_cast_info', 'buffer', 'main'); \
                    select pg_prewarm('person_id_person_info', 'buffer', 'main'); \
                    select pg_prewarm('person_role_id_cast_info', 'buffer', 'main'); \
                    select pg_prewarm('role_id_cast_info', 'buffer', 'main'); \
                    select pg_prewarm('aka_name_pkey', 'buffer', 'main'); \
                    select pg_prewarm('aka_title_pkey', 'buffer', 'main'); \
                    select pg_prewarm('cast_info_pkey', 'buffer', 'main');  \
                    select pg_prewarm('char_name_pkey', 'buffer', 'main'); \
                    select pg_prewarm('comp_cast_type_pkey', 'buffer', 'main'); \
                    select pg_prewarm('company_name_pkey', 'buffer', 'main'); \
                    select pg_prewarm('company_type_pkey', 'buffer', 'main'); \
                    select pg_prewarm('complete_cast_pkey', 'buffer', 'main'); \
                    select pg_prewarm('info_type_pkey', 'buffer', 'main'); \
                    select pg_prewarm('keyword_pkey', 'buffer', 'main'); \
                    select pg_prewarm('kind_type_pkey', 'buffer', 'main'); \
                    select pg_prewarm('link_type_pkey', 'buffer', 'main'); \
                    select pg_prewarm('movie_companies_pkey', 'buffer', 'main'); \
                    select pg_prewarm('movie_info_idx_pkey', 'buffer', 'main'); \
                    select pg_prewarm('movie_info_pkey', 'buffer', 'main'); \
                    select pg_prewarm('movie_keyword_pkey', 'buffer', 'main'); \
                    select pg_prewarm('movie_link_pkey', 'buffer', 'main'); \
                    select pg_prewarm('name_pkey', 'buffer', 'main'); \
                    select pg_prewarm('role_type_pkey', 'buffer', 'main'); \
                    select pg_prewarm('title_pkey', 'buffer', 'main'); \
            " 
            cur.execute(sql)
            print(f"PostgreSQL Finish PreWarming, total time: {time.time() - start} s")

if __name__ == "__main__":
    for port in eval(read_config('leon')['other_db_port']) + [int(read_config('PostgreSQL')['Port'])]:
        print(port)
        prewarm_pg(port)
    