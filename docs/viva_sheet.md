# Viva sheet (answers use only our own measured numbers)

The examiner will change one condition on the spot. Answer by saying which table row you would look at.

## Basics
**Why MobileNetV2 at width 0.35?** Small and pretrained. E6 (width 0.5, 719k params, 2.88 MB fp32 est.) gave val macro F1 0.886 vs 0.890 for width 0.35, so the extra size bought nothing in our runs.
**Why validation and not test for choices?** Choosing by test scores would make test optimistic. Test was used once, for E7.
**How did you find the rare classes?** Computed from class counts: panicle blight 337, leaf streak 380 (next: 479).
**How do you know there is no leakage?** Stratified split, seed 42, MD5 check across all splits with asserts, removed 12 + 22 identical files, augmentation on train only. Limit: near-duplicates not detected; 4 exact duplicates inside test were kept.

## Results
**Which single choice mattered most?** Fine-tuning the backbone: val macro F1 0.509 (E2 frozen) to 0.826 (E3), and 0.681 to 0.890 in the no-augmentation recipe. Class weights mattered most for rare recall: panicle blight 0.88 (E4) vs 0.76 (E5).
**Why does E7 beat E4?** E7 trained on blur/noise/low-res; shift B macro F1 0.625 vs 0.429 (val). But E7 also differs in epochs (30 vs 15) and patience, so we cannot attribute it to augmentation alone.
**Why did E3 (augmented) lose to E4?** Probably heavy zoom-in (25%) cropping small lesions and only 15 epochs. Not proven.
**Why four E1 rows?** Two early unseeded runs (0.366 and 0.668, provenance not recorded) plus seeds 1 and 2 (0.570 and 0.735); a third seed was not finished in time. Range 0.366 to 0.735 shows large initialisation variance; seeds do not change the data shuffle order.
**Is it overfitting or shift (Q2)?** Both, shift dominates. Un-augmented train/val/shift B accuracy: E4 0.994/0.895/0.441; E7 0.950/0.915/0.641.
**What is the model's weakest point?** Test: downy_mildew precision 0.731; `normal` predicted as `hispa` 17 times. Under shift B: leaf streak precision falls to 0.412 and panicle blight recall to 0.569.

## Honest limits
**Are the field results real?** No. Simulated proxy; real photos were not collectable in the time and resources available.
**Is 2.6 ms a phone number?** No. Colab CPU, one thread, model only. Phones are slower; the 100x margin under 300 ms is the argument.
**Is shift B independent of training?** Only partly: motion blur ~ Gaussian blur, salt-and-pepper ~ noise, harsh JPEG ~ low-res. Low light is the only unrelated type (E7 0.754 vs E4 0.702).
**Single seed?** Yes for E2-E7; differences of 1-2 points are noise. E1 shows large seed variance.

## "What if the examiner changes a condition?"
| Change | Answer |
|---|---|
| Size limit 1 MB | Use dynamic int8: 0.58 MB, test macro F1 0.8832, panicle blight recall 0.902, latency 4.7 ms |
| Size limit 0.3 MB | Not met by our files; would need width 0.25 or pruning, and retraining; accuracy unknown until tested |
| Latency 10 ms | Float32 2.6 ms and int8 4.7 ms on Colab; a phone is slower, so measure on device |
| No class weights | E5: panicle blight recall 0.76 (barely above 0.75) |
| Rare-class recall target 0.90 | Test panicle blight 0.882 misses it; options: more rare-class data, higher class weight, threshold tuning on validation |
| New disease class | Replace the Dense head, add images, retrain/fine-tune; rare-class handling and class weights recomputed |
| Different false-alarm cost | Use predicted probabilities and a per-class threshold chosen on validation |
| Field photos blurry | Raise blur/motion-blur share in training augmentation; E7 motion-blur acc 0.583 (E4 0.245) on val shift B |
