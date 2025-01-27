import os
import sys
import logging
import configparser
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from supabase import create_client, Client


class BufferingHandler(logging.Handler):
    """
    Custom logging handler that buffers log records in memory. This handler is useful for situations where
    logs need to be accumulated and processed in bulk rather than being written out individually.
    Attributes:
        buffer (list): A list to hold formatted log records.
        filename (str): The name of the log file for which this handler is created.
    """
    def __init__(self, filename: str) -> None:
        """
        Initializes the BufferingHandler with a specified filename.
        Parameters:
            filename (str): The name of the log file associated with this handler.
        """
        super().__init__()
        self.buffer = []
        self.filename = filename

    def emit(self, record: logging.LogRecord) -> None:
        """
        Formats and appends a log record to the buffer.
        Parameters:
            record (logging.LogRecord): The log record to be processed and added to the buffer.
        """
        # Append the log record to the buffer
        self.buffer.append(self.format(record))

    def flush(self) -> str:
        """
        Flushes the buffer by joining all buffered log records into a single string. Clears the buffer afterward.
        Returns:
            str: A single string containing all buffered log records separated by newlines.
                  Returns an empty string if the buffer is empty.
        """
        if len(self.buffer) > 0:
            return '\n'.join(self.buffer)
        else:
            return ''


class Utils:
    """
    Provides utility functions and classes for logging, configuration management, and interaction with cloud storage.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, session_id: int, interview_id: int) -> None:
        """
        Parameters:
            session_id (int)
            interview_id (int)
        Functionality:
            Initializes logging, configuration, and database client (supabase_client).
        """
        if not self.__initialized:
            self.config = self.__get_config()

            self.session_id = session_id
            self.interview_id = interview_id

            # S3 Folders
            self.output_s3_folder = '{}/{}/output'.format(self.session_id, self.interview_id)

            # Create loggers
            self.log = self.__init_logs()
            self.log.propagate = False

            self.supabase_client = self.__check_supabase_connection()
            self.supabase_connection = self.__connect_to_bucket()
            self.supabase: Client = create_client(self.config['SUPABASE']['Url'], os.environ.get('SUPABASE_KEY'))

            self.__initialized = True

    def __del__(self):
        self.__initialized = False

    def __init_logs(self) -> logging.Logger:
        """
        Initializes and configures logging for the application. This method sets up separate log handlers
        for INFO and ERROR level messages to ensure logs are captured appropriately.
        Returns:
            logging.Logger: The configured root logger with handlers for INFO and ERROR logs.
        Functionality:
            - Sets logging level to INFO for general logs.
            - Configures formatters to include timestamp, log level, and message details.
            - Creates separate file handlers for INFO and ERROR logs with buffering capabilities.
        """
        logger = logging.getLogger('audioLog')
        logger.setLevel(logging.INFO)

        # Create a file handler for INFO messages
        handler = BufferingHandler('audioLog_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S')))
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'))

        # Add the handlers to the root logger
        logger.addHandler(handler)
        logger.datefmt = '%d/%b/%Y %H:%M:%S'
        logger.encoding = 'utf-8'
        return logger

    def __get_config(self) -> configparser.ConfigParser:
        """
        Loads and returns the configuration settings from a 'config.ini' file. This method ensures that the application
        configuration is centrally managed and easily accessible.
        Returns:
            configparser.ConfigParser: A configuration parser object loaded with settings from 'config.ini'.
        Raises:
            IOError: If 'config.ini' is not found, raises an IOError and halts the program, indicating the dependency
                     on this configuration file for the application's operation.
        """
        config = configparser.ConfigParser()
        if len(config.sections()) == 0:
            try:
                base_path = os.path.dirname(os.path.dirname(__file__))
                path = os.path.join(base_path, 'config', 'audioConfig.ini')
                with open(path) as f:
                    config.read_file(f)
            except IOError as e:
                print("No file 'audioConfig.ini' is present, the program can not continue")
                raise e
        return config

    def __check_supabase_connection(self) -> Client:
        """
        Attempts to establish a connection with the Supabase client using the application's configuration settings.
        Returns:
            Client: The connected Supabase client if the connection is successful.
        Raises:
            Exception: Logs and raises an exception if the connection to Supabase fails, including error details.
                       This method will also terminate the application (sys.exit(1)) if a connection cannot be
                       established, as the connection is critical for the application's functionality.
        """
        try:
            client = create_client(self.config['SUPABASE']['Url'], os.environ.get('SUPABASE_KEY'))
        except Exception as e:
            message = ('Error connecting to Supabase, the program can not continue.', str(e))
            self.log.error(message)
            print(message)
            sys.exit(1)
        return client

    def __connect_to_bucket(self) -> Any:
        """
        Establishes and returns a connection to a designated S3 bucket using the Supabase client.
        This method is essential for managing file storage operations within the application.
        Returns:
            Any: The connection object to the designated S3 bucket if the connection is successful.
        Raises:
            Exception: Logs an error and terminates the application if the connection to the S3 bucket fails.
                       This ensures that the application does not continue without necessary storage capabilities.
        """
        bucket_name = self.config['SUPABASE']['InputBucket']
        connection = self.supabase_client.storage.from_(bucket_name)
        try:
            connection.list()
            self.log.info('Connection to S3 bucket {} successful'.format(bucket_name))
        except Exception as e:
            message = ('Error connecting to S3 bucket {}, the program can not continue.'.
                       format(bucket_name), str(e))
            self.log.error(message)
            print(message)
            sys.exit(1)
        return connection

    def end_log(self) -> None:
        """
        Functionality:
            Flushes buffered logs to a file and uploads it to S3.
            Ends logging for the session.
        """
        log_handlers = logging.getLogger('audioLog').handlers[:]
        print('Audio analysis finished. Saving {} log'.format(len(log_handlers)))
        for handler in log_handlers:
            if isinstance(handler, BufferingHandler):
                log = handler.flush()
                if log:
                    s3_path = '{}/{}/logs/{}'.format(self.session_id, self.interview_id, handler.filename)
                    try:
                        self.supabase_connection.upload(file=log.encode(),
                                                        path=s3_path,
                                                        file_options={'content-type': 'text/plain'}
                                                        )
                        self.log.info('Log file {} uploaded to S3 bucket'.format(handler.filename))
                    except Exception as e:
                        self.log.error('Error uploading the file {} to the S3 bucket : {}.'.
                                       format(handler.filename, str(e)))
            logging.getLogger('audioLog').removeHandler(handler)

    def get_segments_from_db(self) -> pd.DataFrame | None:
        """
        Fetches audio segment data from the database for a given interview.
        Returns:
            pd.DataFrame | None: A DataFrame containing audio segment data or None if an error occurs.
        Raises:
            Exception: An exception is raised if there is an issue fetching data from the database.
        """
        try:
            res = (self.supabase.table('results').select('id', 'start', 'end')
                   .eq('interview_id', self.interview_id)
                   .eq('speaker', 0)
                   .execute())
            results = pd.DataFrame(res.data)
            results.set_index('id', inplace=True)
            return results
        except Exception as e:
            message = ('Error getting the segments from the database.', str(e))
            self.log.error(message)
            raise e

    def open_input_file(self, s3_path: str, file_name: str) -> bytes | None:
        """
        Retrieves an audio file from the S3 bucket based on the provided path and file name.
        Parameters:
            s3_path (str): The path within the S3 bucket where the file is stored.
            file_name (str): The name of the file to retrieve.
        Returns:
            bytes | None: The content of the file as bytes, or None if an error occurs.
        Raises:
            Exception: An exception is raised if there is an issue downloading the file.
        """
        try:
            self.log.info('Getting file {} from the S3 bucket'.format(file_name))
            file_bytes = self.supabase_connection.download(s3_path)
            return file_bytes
        except Exception as e:
            message = ('Error downloading the file {} from the S3 bucket. '.
                       format(file_name), str(e))
            self.log.error(message)
            raise e

    def update_results(self, results: pd.DataFrame) -> None:
        """
        Updates the database with the results of the audio emotion analysis.
        Parameters:
            results (pd.DataFrame): The DataFrame containing updated results for each audio segment.
        Raises:
            Exception: An exception is raised if there is an issue updating the database.
        """
        try:
            for row in results.itertuples():
                (self.supabase.table('results')
                 .update({'audio_emotions': row.audio_emotions})
                 .eq('id', row.Index)
                 .execute()
                 )
            self.log.info('Results from audio updated in the database')
        except Exception as e:
            self.log.error('Error updating the results from audio in the database')
            raise e

    def adjust_values(self, input_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Adjusts the values in the input dictionary to ensure their total sum does not exceed 100.
        This method checks if the total sum of the values in the dictionary is greater than 100.
        If so, it scales the values down proportionally to make the total sum exactly 100.
        Parameters:
            input_dict (Dict[str, float]): A dictionary with string keys and float values to be adjusted.
        Returns:
            Dict[str, float]: The adjusted dictionary with values scaled to ensure their total sum is 100
                              if the initial sum exceeded 100; otherwise, returns the original dictionary.
        """
        total_sum = sum(input_dict.values())

        if total_sum <= 100:
            return input_dict

        # Scale values to make the total sum 100
        scale_factor = 100 / total_sum
        adjusted_dict = {k: v * scale_factor for k, v in input_dict.items()}

        return adjusted_dict
