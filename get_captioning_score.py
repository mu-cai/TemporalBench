import json
from tqdm import tqdm
import argparse
import os


from sentence_transformers import SentenceTransformer, util




# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data_folder', type=str, default="./TemporalBench_local_data", help='Path to dataset (from Huggingface)')
parser.add_argument("--output_folder", type=str, default="./outputs", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=1, help="Number of frames to sample.")

# Parse arguments
args = parser.parse_args()

output_dir = os.path.join(args.output_folder, 'temporalbench_short_caption')


with open(os.path.join(args.data_folder, 'temporalbench_short_caption.json'), 'r') as f:
    questions = json.load(f)
    
id2question = {q['idx']: q for q in questions}



result_predictions = os.listdir(output_dir)




def calculate_average_similarity(ref_list, gt_list, model_name='all-MiniLM-L6-v2'):
    # Initialize the model and move to CUDA
    model = SentenceTransformer(model_name)
    model = model.to('cuda')
    
    # Combine ref and gt lists into a big batch for encoding
    combined_sentences = ref_list + gt_list
    
    # Encode the batch with CUDA
    embeddings = model.encode(combined_sentences, convert_to_tensor=True, device='cuda')
    
    # Split embeddings into ref and gt parts
    ref_embeddings = embeddings[:len(ref_list)]
    gt_embeddings = embeddings[len(ref_list):]
    
    # Calculate cosine similarities between each ref and gt pair
    cosine_scores = util.cos_sim(ref_embeddings, gt_embeddings).diagonal()
    
    # Calculate the average similarity
    avg_similarity = cosine_scores.mean().item()
    
    return avg_similarity
  
  
for pred_name in result_predictions:
    with open(os.path.join(output_dir, pred_name), 'r') as f:
        preds = [json.loads(line) for line in f]
    
    gt_captions = [ id2question[pred['idx']]['GT'] for pred in preds]
    pred_captions = [ pred['response'] for pred in preds]
    similarity = calculate_average_similarity(pred_captions, gt_captions)*100
    print('*' * 20, pred_name, '*' * 20)
    print('Average similarity:', f'{similarity:.4f}')
    
    
        