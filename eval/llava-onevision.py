import numpy as np
import copy
import warnings
from decord import VideoReader, cpu
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
parser.add_argument('--model_name', type=str, default="llava-onevision-qwen2-7b-ov", help='Path to model checkpoints')
parser.add_argument("--output_folder", type=str, default="./outputs", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=1, help="Number of frames to sample.")

# Parse arguments
args = parser.parse_args()

output_dir = os.path.join(args.output_folder, args.data_json.split('.')[0])
nframes = args.nframes
os.makedirs(output_dir, exist_ok=True)


##################### Initilaize the model #####################


from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained =  os.path.join(args.ckpt_folder, args.model_name)
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

model.eval()




# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


##################### Get response #####################


def get_response(video_path, nframes, question):
        video_frames = load_video(video_path, nframes)
        # print(video_frames.shape) # (16, 1024, 576, 3)
        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        # Prepare conversation input
        conv_template = "qwen_2"
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n" + question["question"] 
      #   + "\nPlease only output one English character."

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [frame.size for frame in video_frames]

        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        
        reponse = text_outputs[0]
        return reponse


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
