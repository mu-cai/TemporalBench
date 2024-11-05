import json
from tqdm import tqdm
import argparse
import os



# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data_folder', type=str, default="./TemporalBench_local_data", help='Path to dataset (from Huggingface)')
parser.add_argument('--data_json', type=str, default="temporalbench_short_qa.json", help='which type ')
parser.add_argument("--output_folder", type=str, default="./outputs", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=1, help="Number of frames to sample.")

# Parse arguments
args = parser.parse_args()

output_dir = os.path.join(args.output_folder, args.data_json.split('.')[0])



with open(os.path.join(args.data_folder, args.data_json), 'r') as f:
    questions = json.load(f)
    
id2question = {q['idx']: q for q in questions}


result_predictions = os.listdir(output_dir)

for pred_name in result_predictions:
    with open(os.path.join(output_dir, pred_name), 'r') as f:
        preds = [json.loads(line) for line in f]
    correct_count = 0
    multiple_binary_qa_correct = {}
    binary_qa_per_dataset = {}
    multiple_binary_qa_per_dataset = {}
    for pred in preds:
      idx = pred['idx']
      gt = id2question[idx]['GT']
      predict_correct = gt.lower() == pred['response'][0].lower()
      if predict_correct:
        correct_count += 1
      
      video_name = id2question[idx]['video_name']
      if video_name not in multiple_binary_qa_correct:
        multiple_binary_qa_correct[video_name] = True
      if not predict_correct:
            multiple_binary_qa_correct[video_name]= False
      
      dataset = id2question[idx]['dataset']
      if dataset not in binary_qa_per_dataset:
        binary_qa_per_dataset[dataset] = []
        multiple_binary_qa_per_dataset[dataset] = {}
      binary_qa_per_dataset[dataset].append(predict_correct)
      if video_name not in multiple_binary_qa_per_dataset[dataset]:
        multiple_binary_qa_per_dataset[dataset][video_name] = True
      if not predict_correct:
        multiple_binary_qa_per_dataset[dataset][video_name] = False
      
      
      
    try:
      width_dataset = 40   # for dataset names
      width_counts = 15    # for correct/total counts
      width_percentage = 1 # for percentages

      print('*' * 20, pred_name, '*' * 20)
      print(f"{'Binary Accuracy:':<{width_dataset}} {correct_count}/{len(preds):<{width_counts}} {correct_count/len(preds) * 100:>{width_percentage}.2f}%")
      mba_correct = sum([1 for v in multiple_binary_qa_correct.values() if v])
      print(f"{'Multiple Binary Accuracy:':<{width_dataset}} {mba_correct}/{len(multiple_binary_qa_correct):<{width_counts}} {mba_correct/len(multiple_binary_qa_correct) * 100:>{width_percentage}.2f}%")
# Print header
      print(f"|--- {'Dataset':<{width_dataset}}Binary Accuracy  {'':<{7}}  {'':>{width_percentage}} "
      f"||| Multiple Binary Accuracy {'':<{width_counts}}  {'':>{width_percentage}}")
      for dataset, binary_qa in binary_qa_per_dataset.items():
            mba_correct = sum([1 for v in multiple_binary_qa_per_dataset[dataset].values() if v])
            print(f"|--- {dataset + ' ':<{width_dataset}} {sum(binary_qa)}/{len(binary_qa):<{width_counts}} {sum(binary_qa)/len(binary_qa) * 100:>{width_percentage}.2f}% "
                  f"||| {mba_correct}/{len(multiple_binary_qa_per_dataset[dataset]):<{width_counts}} {mba_correct/len(multiple_binary_qa_per_dataset[dataset]) * 100:>{width_percentage}.2f}%")


    except Exception as e:
      print(f"Error running video: {e}")
      continue

