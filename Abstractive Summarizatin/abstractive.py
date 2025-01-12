from transformers import T5Tokenizer, T5ForConditionalGeneration, PegasusTokenizer, PegasusForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
import torch
import evaluate
import bert_score

# Load evaluation metrics from the evaluate library
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# Load BERTScore for evaluation
def calculate_bertscore(predictions, references):
    P, R, F1 = bert_score.score(predictions, references, lang="en", verbose=True)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

# Helper function to run a summarization model
def summarize_with_model(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load models and tokenizers once (at startup)
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Store models in a dictionary
models = {
    "T5": (t5_model, t5_tokenizer),
    "PEGASUS": (pegasus_model, pegasus_tokenizer),
    "BART": (bart_model, bart_tokenizer)
}

# Function to summarize and evaluate
def evaluate_summarization(models, input_text):
    summaries = {}
    metrics = {}

    for model_name, (model, tokenizer) in models.items():
        summary = summarize_with_model(model, tokenizer, input_text)
        summaries[model_name] = summary

        # ROUGE
        rouge_result = rouge.compute(predictions=[summary], references=[input_text])
        
        # METEOR
        meteor_result = meteor.compute(predictions=[summary], references=[input_text])
        
        # BERTScore
        bertscore_result = calculate_bertscore([summary], [input_text])
        
        # Store metrics
        metrics[model_name] = {
            "rouge": rouge_result,
            "meteor": meteor_result,
            "bertscore": bertscore_result
        }

    return summaries, metrics

# Example usage inside an API route
# @app.route('/summarize', methods=['POST'])
# def summarize_route():
#     input_text = request.json['text']
#     summaries, evaluation_metrics = evaluate_summarization(models, input_text)
#     return jsonify({'summaries': summaries, 'metrics': evaluation_metrics})
