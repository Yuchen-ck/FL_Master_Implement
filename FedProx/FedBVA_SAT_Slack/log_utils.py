
def save_result(start_time,Aggregate_Algorithm,train_times,test_r2_score ,client_number ,client_rate ,client_epoch , cr):
    import logging
    logging.basicConfig(level=logging.DEBUG, filename='output.log',format='%(asctime)s - %(message)s')
    logger = logging.getLogger(name=__name__)

    log_start_time =" - Start Time: %4s " % start_time
    log_aggregate_algo = " - Aggregate Algorithm : %4s " % Aggregate_Algorithm
    log_train_times = "- Running time: %d:%02d:%02d" % train_times
    log_context = " - Test: Accuracy : %4f " % test_r2_score
    log_parameters = " -Client_number : %2d" % client_number + " -Client_rate : %2f" % client_rate  + " -Communicate Round : %2d" % cr  + " -Client_epoch : %2d" % client_epoch

    log_context =  log_start_time + log_aggregate_algo + log_train_times + log_context + log_parameters 
    logger.info(log_context)