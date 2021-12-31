from db import mongobase
from constants import dbconstants


def get_mongo_connector(host_name=dbconstants.LOCAL_MONGO_HOSTNAME,port_no=dbconstants.LOCAL_MONGO_PORT,
                        db_name=dbconstants.DB_NAME, collection_name=dbconstants.COLLECTION_NAME):
    mongo = mongobase.MongoConnector(uri=host_name, port_no=port_no)
    mongo.set_db(db_name)
    mongo.set_collection(collection_name)
    return mongo


def check_duplicate_document(document, mongo_connector):
    query_dict = dict({'model_type': document['model_type'],
                       'data_filename': document['data_filename'],
                       'model_name': document['model_name'],
                       'correction_model': document['correction_model'],
                       'noise_type': document['noise_type'],
                       'noise_level': document['noise_level']})
    if mongo_connector.check_document(query_dict) is False:
        return False
    return True


def insert_document(document, mongo_connector):
    if check_duplicate_document(document,mongo_connector) is False:
        mongo_connector.insert_document(document)
    mongo_connector.close_connection()


def check_duplicate_training_model(document, mongo_connector):
    query_dict = dict({
        'model_operation_type': document['model_operation_type'],
        'nn_model_type': document['nn_model_type'],
        'train_batch_size': document['train_batch_size'],
        'test_batch_size': document['test_batch_size'],
        'data_filename': document['data_filename']
        })
    if mongo_connector.check_document(query_dict) is False:
        return False
    return True


def insert_document_training(document, mongo_connector):
    if check_duplicate_training_model(document, mongo_connector) is False:
        mongo_connector.insert_document(document)
    mongo_connector.close_connection()
