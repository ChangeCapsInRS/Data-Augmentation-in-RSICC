ui: clean_output
	streamlit run src/ui.py

paraphrase:
	python3 src/paraphrase.py

print_clips:
	python3 src/clip_scores.py \
        --fine_tuned \
        --images_path data/SampleImagesWithoutAugmentation \
        --captions_path data/merged_no_aug_without_unicode_sampled.json

clean_output:
	rm -rf output output_random