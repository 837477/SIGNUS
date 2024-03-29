'''
MongoDB Management Modules and Models
'''
from datetime import datetime
from pymongo import MongoClient
from flask import g, current_app
from werkzeug.security import generate_password_hash
from app.controllers.auth import signup


def get_mongo_cur():
    ''' return mongodb cursor'''
    return MongoClient(current_app.config['MONGODB_URI'])


def open_mongo_cur():
    ''' store mongodb cursor in the g'''
    g.mongo_cur = MongoClient(current_app.config['MONGODB_URI'])


def close_mongo_cur():
    ''' pop & close mongodb cursor in the g'''
    mongo_cur = g.pop('mongo_cur', None)
    if mongo_cur:
        mongo_cur.close()


def init_models(config):
    '''mongodb-init function'''
    cur = MongoClient(config.MONGODB_URI)
    db_name = config.MONGODB_DB_NAME
    
    # Create master_config collection
    col = cur[db_name]['master_config']
    col.insert_many(config.DEFAULT_MASTER_CONFIG)

    # Create realtime collection
    col = cur[db_name]['realtime']
    col.insert_one(config.DEFAULT_REALTIME)

    # Create admin account
    signup(cur,
           config.SEJONG_ID,
           config.SEJONG_PW,
           config.ADMIN_ID,
           config.ADMIN_PW,
           "SIGNUS")
    
    cur.close()
