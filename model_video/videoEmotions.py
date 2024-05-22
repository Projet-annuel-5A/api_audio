import os
import cv2
import torch
import tempfile
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict
from utils.utils import Utils
from dotenv import load_dotenv


class VideoEmotions:
    def __init__(self, session_id: int, interview_id: int) -> None:
        # Load environment variables from .env file
        load_dotenv()
        self.utils = Utils(session_id, interview_id)

    def __predict(self, image: np.ndarray) -> Dict[str, float]:
        inputs = self.utils.vte_processor(image, return_tensors="pt")
        vte_model = self.utils.vte_model(**inputs)
        logits = vte_model.logits
        attention_weights = vte_model.attentions

        m = nn.Softmax(dim=0)
        values = m(logits[0])
        print('Values:', values)

        w_values = np.zeros(len(values))
        for i in range(len(w_values)):
            result = values[i].item() * attention_weights[i]
            w_values[i] = torch.sum(result)

        # Convert to percentage
        total = np.sum(w_values)
        w_values = w_values * 100 / total
        print('W_values:', w_values)

        output = dict(zip(self.utils.vte_model.config.id2label.values(), w_values))
        print('Output:', output)

        sorted_values = {k: v for k, v in sorted(output.items(), key=lambda x: x[1], reverse=True)}
        print('Sorted values:', sorted_values)

        return sorted_values

    def process(self, _speakers: Dict) -> pd.DataFrame:
        res = pd.DataFrame(columns=['speaker', 'part', 'video_emotions'])
        print('Processing video emotions')

        s3_path = '{}/{}/raw/{}'.format(self.utils.session_id,
                                        self.utils.interview_id,
                                        self.utils.config['GENERAL']['Filename'])
        video_bytes = self.utils.supabase_connection.download(s3_path)
        print('Video bytes downloaded')

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name
            try:
                temp_file.write(video_bytes)

                clip = cv2.VideoCapture(temp_file_path)
                # Get video fps
                fps = clip.get(cv2.CAP_PROP_FPS)

                # Set the interval for extracting frames
                timing = self.utils.config.getfloat('VIDEOEMOTION', 'Interval')
                interval = int(fps) * timing

                for current_speaker in _speakers.keys():
                    print('Processing speaker:', current_speaker)
                    parts = _speakers[current_speaker]

                    for i in range(len(parts)):
                        print('Processing part ', i)
                        video_emotions = {}
                        start_time = parts[i][0]
                        end_time = parts[i][1]
                        print('Start time: ', start_time, 'End time: ', end_time)

                        # Calculate frame indices for starting and ending times
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)
                        print('Start frame: ', start_frame, 'End frame: ', end_frame)

                        # Set starting frame
                        clip.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        print('Starting frame set to frame: ', start_frame)

                        # Read the video frame by frame and send respective frames to prediction
                        frame_count = 0
                        image_count = 0
                        while clip.isOpened() and frame_count <= (end_frame - start_frame):
                            image_name = 'image_{:05d}'.format(image_count)
                            ret, frame = clip.read()

                            # If there are no more frames, break the loop
                            if not ret:
                                break

                            # Detect emotions from the frame if it's a multiple of the interval
                            if frame_count % interval == 0:
                                print('Processing frame: ', frame_count)
                                video_emotions[image_name] = self.__predict(frame)
                                image_count += 1
                                last_frame = None
                            else:
                                last_frame = frame
                            frame_count += 1

                            # Save the last frame
                            if last_frame is not None:
                                video_emotions[image_name] = self.__predict(last_frame)
                                print('Processing frame: ', frame_count)

                        res.loc[len(res)] = [int(current_speaker.split('_')[1]),
                                             i,
                                             video_emotions]

                # Release the video capture object
                clip.release()
                print('Video capture object released')
            finally:
                temp_file.close()
                print('Temporary file closed')
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        self.utils.log.info('Emotions extraction from video have finished')

        return res
