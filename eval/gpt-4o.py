# pip install openai
import numpy as np
import json
from tqdm import tqdm
import argparse
import os


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data_folder', type=str, default="./TemporalBench_local_data", help='Path to dataset (from Huggingface)')
parser.add_argument('--data_json', type=str, default="temporalbench_short_qa.json", help='which type ')
parser.add_argument('--ckpt_folder', type=str, default="lmms-lab", help='Folder to model checkpoints')
parser.add_argument('--model_name', type=str, default="gpt-4o", help='Path to model checkpoints')
parser.add_argument("--output_folder", type=str, default="./outputs", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=8, help="Number of frames to sample.")
parser.add_argument("--detail_level", type=str, default='low', help="Number of frames to sample.")


# Parse arguments
args = parser.parse_args()

output_dir = os.path.join(args.output_folder, args.data_json.split('.')[0])
nframes = args.nframes
os.makedirs(output_dir, exist_ok=True)




##################### Initilaize the model #####################


from openai import OpenAI
from openai import AzureOpenAI


MODEL = "gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
detail_level = args.detail_level




# Function to extract frames from video
import cv2, base64


def process_video(video_path, fixed_num_frames=-1, fps_sample=None):
    base64Frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps_sample is not None:
        frames_to_skip = int(fps / fps_sample)
        print(frames_to_skip, total_frames, fps)
        frame_indices = range(0, total_frames, frames_to_skip)
        if fixed_num_frames!= None and fixed_num_frames> 0:
            frame_indices = frame_indices[:fixed_num_frames]
    else:
        sample_num_frames = min(fixed_num_frames, total_frames) if fixed_num_frames > 0 else total_frames
        frame_indices = np.linspace(0, total_frames - 1, sample_num_frames, dtype=int)

    for frame_idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    return base64Frames



##################### Get response #####################





def get_response(video_path, fixed_num_frames, question):
    if fixed_num_frames ==0:
        content =    f"These are the frames from the video. {question}"
    else:
        base64Frames = process_video(video_path, fixed_num_frames, fps_sample = None)    
        content = [
            "These are the frames from the video.",
            *map(lambda x: {"type": "image_url", 
                            "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": detail_level}}, base64Frames),
            f"{question}",
        ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in video understanding."},
            {"role": "user", "content": content}
        ],
        temperature=0
    )
    return response.choices[0].message.content
    


with open(os.path.join(args.data_folder, args.data_json), 'r') as f:
    questions = json.load(f)
    



text_ans_file = open(os.path.join(output_dir, f"{args.model_name}-frame{nframes}.jsonl"), 'w')



for question in tqdm(questions):
    try:
          # Load and process video
          video_path = os.path.join(args.data_folder, question["video_name"])
          response = get_response(video_path, nframes, question)
          text_ans_file.write(json.dumps(dict(idx=question["idx"], response = response)) + '\n')
          text_ans_file.flush()
        
    except Exception as e:
        print(f"Error running video: {e}")
        continue

text_ans_file.close()
