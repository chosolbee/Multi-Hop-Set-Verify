# Multi-Hop-Set-Verify

```shell
# Extract corpus
python3 Full-Pipeline/contriever/multihop_data_extractor.py \
    --input "path_to_raw_dataset" \
    --output "path_to_save_corpus"

# Create document embeddings
python3 Full-Pipeline/contriever/passage_embedder.py \
    --passages "path_to_corpus" \
    --output_dir "directory_to_save_embeddings"
```

```shell
# Test each module separately
python3 Full-Pipeline/contriever/passage_retriever.py \
    --passages "path_to_corpus" \
    --embeddings "directory_with_embeddings" \
    --save_or_load_index  # will create a faiss index for the corpus

python3 Full-Pipeline/query_generator/query_generator.py \
    --cache-dir "llama_cache_dir"

python3 Full-Pipeline/verifier/verifier.py \
    --checkpoint-path "verifier_checkpoint_path" \
```
```shell
# Run the full pipeline
python3 -m Full-Pipeline.pipeline \
    --passages "path_to_corpus" \
    --embeddings "directory_to_save_embeddings" \
    --questions Full-Pipeline/musique_questions.jsonl \
    --verifier-checkpoint-path "verifier_checkpoint_path"
```
