import os
import sys
import json
import torch
import logging
import warnings
import tempfile
import configparser
import pandas as pd
from datetime import datetime
from typing import Tuple, Any, Dict, List
from supabase import create_client, Client
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import (AutoImageProcessor,
                          AutoModelForImageClassification)


class BufferingHandler(logging.Handler):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.buffer = []
        self.filename = filename

    def emit(self, record: logging.LogRecord) -> None:
        # Append the log record to the buffer
        self.buffer.append(self.format(record))

    def flush(self) -> str:
        if len(self.buffer) > 0:
            return '\n'.join(self.buffer)
        else:
            return ''


class Utils:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, session_id: int, interview_id: int) -> None:
        if not self.__initialized:
            self.config = self.__get_config()

            self.session_id = session_id
            self.interview_id = interview_id
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # S3 Folders
            self.output_s3_folder = '{}/{}/output'.format(self.session_id, self.interview_id)

            # Create loggers
            self.log = self.__init_logs()
            self.log.propagate = False

            self.supabase_client = self.__check_supabase_connection()
            self.supabase_connection = self.__connect_to_bucket()

            (self.vte_model,
             self.vte_processor) = self.__init_models()

            self.__initialized = True

    def __init_logs(self) -> logging.Logger:
        logger = logging.getLogger('videoLog')
        logger.setLevel(logging.INFO)

        # Create a file handler for INFO messages
        handler = BufferingHandler('videoLog_{}'.format(datetime.now().strftime('%Y_%m_%d_%H.%M.%S')))
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'))

        # Add the handlers to the root logger
        logger.addHandler(handler)
        logger.datefmt = '%d/%b/%Y %H:%M:%S'
        logger.encoding = 'utf-8'
        return logger

    def __get_config(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        if len(config.sections()) == 0:
            try:
                base_path = os.path.dirname(os.path.dirname(__file__))
                path = os.path.join(base_path, 'config', 'videoConfig.ini')
                with open(path) as f:
                    config.read_file(f)
            except IOError:
                print("No file 'videoConfig.ini' is present, the program can not continue")
                sys.exit()
        return config

    def __init_models(self) -> Tuple:
        # Video to emotions
        vte_model_id = self.config['VIDEOEMOTION']['ModelId']
        vte_model = AutoModelForImageClassification.from_pretrained(vte_model_id, output_attentions=True)
        vte_model.to(self.device)
        vte_processor = AutoImageProcessor.from_pretrained(vte_model_id, output_attentions=True)
        self.log.info('Video-to-emotions model {} loaded in {}'.format(vte_model_id, self.device))

        return (vte_model,
                vte_processor)

    def __check_supabase_connection(self) -> Client:
        try:
            client = create_client(self.config['SUPABASE']['Url'], os.environ.get('SUPABASE_KEY'))
        except Exception as e:
            message = ('Error connecting to Supabase, the program can not continue.', str(e))
            self.log.error(message)
            print(message)
            sys.exit(1)
        return client

    def __connect_to_bucket(self) -> Any:
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

    def save_to_s3(self, filename: str, content: bytes, file_format: str, s3_subfolder: str = None) -> bool:
        match file_format:
            case 'audio': content_type = 'audio/mpeg'
            case 'video': content_type = 'video/mp4'
            case 'text': content_type = 'text/plain'
            case _: content_type = 'text/plain'

        try:
            s3_path = '{}/{}/{}'.format(self.output_s3_folder,
                                        s3_subfolder,
                                        filename) if s3_subfolder else '{}/{}'.format(self.output_s3_folder, filename)
            self.supabase_connection.upload(file=content, path=s3_path, file_options={"content-type": content_type})
            self.log.info('File {} uploaded to S3 bucket at {}'.format(filename, s3_path))
            return True
        except Exception as e:
            message = ('Error uploading the file {} to the S3 bucket.'.
                       format(filename), str(e))
            self.log.error(message)
            return False

    def end_log(self) -> None:
        log_handlers = logging.getLogger('videoLog').handlers[:]
        for handler in log_handlers:
            if isinstance(handler, BufferingHandler):
                log = handler.flush()
                if log:
                    self.save_to_s3('{}.log'.format(handler.filename), log.encode(), 'text', 'logs')
            logging.getLogger('videoLog').removeHandler(handler)

    def get_speakers_from_s3(self) -> Dict[str, List[Tuple[float, float]]]:
        path = '{}/temp/speakers.json'.format(self.output_s3_folder)
        res = self.supabase_connection.download(path)
        speakers_str = res.decode()
        speakers = json.loads(speakers_str)
        return speakers

    def df_to_temp_s3(self, df: pd.DataFrame, filename: str) -> None:
        s3_path = '{}/temp/{}.tmp'.format(self.output_s3_folder, filename)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file_path = temp_file.name
            try:
                df.to_hdf(temp_file_path, key='data', mode='w', complevel=9, complib='blosc')

                with open(temp_file_path, 'rb') as f:
                    try:
                        self.supabase_connection.upload(file=f, path=s3_path,
                                                        file_options={'content-type': 'application/octet-stream'})
                        self.log.info('File {} uploaded to S3 bucket'.format(s3_path))
                    except Exception as e:
                        message = (
                            'Error uploading the file to the S3 bucket. ', str(e))
                        self.log.info(message)
            finally:
                temp_file.close()
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)