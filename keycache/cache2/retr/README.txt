10_final/cache2/retr -- attributes and embeddings of the retrieval reference (retr) and of the lookup rule (PROTOCOL.md section 6,
"Review round 1", item R4).  Written by 10_final/feat2/retr (build_attributes.py, embed.py; this file: make_readme.py).  Reader: feat2/retr/retr.py
(retr.store(<backbone>)); usage and tests: feat2/retr/USAGE.txt.  Label-free: no gate decision, no outcome of any log is stored here.

FILES
  <backbone>/attributes.jsonl       qwen25vl, llava: the two attributes of every sequence of the backbone's key cache and of every edit request
  <backbone>/attributes_meta.json   md5 of the table, counts, the checks made while building it, one rendered prompt per family / row type
  emb/image_keys.jsonl              one line per distinct image content (md5), sorted by md5: md5, image (representative file), n_paths, backbones
  emb/question_keys.jsonl           one line per distinct question key (question.strip().lower()), sorted: question_key, backbones
  emb/keys_meta.json                counts and md5 of the two key lists, the attribute tables they were built from
  emb/image_emb.npy                 float32 (n_images, 1024): row i = embedding of image content i of image_keys.jsonl
  emb/question_emb.npy              float32 (n_questions, 384): row i = embedding of question key i of question_keys.jsonl
  emb/meta.json                     encoders, exact preprocessing, library versions, md5 of the matrices, norm statistics, GPU memory measured
  emb/parts/                        the resumable parts the matrices were assembled from (256 keys each) and the record of every run
The embeddings are shared by the backbones: an image content or a question string that occurs in both caches has ONE row.

ATTRIBUTE TABLE (one JSON object per line, keys in this order)
  table, row, sid, kind, family, ptype, image, image_md5, question, question_key, edit, edit_uid, edit_b, edit_b_uid
  lines 0 .. n - 1: table = 'sequence', line j = the sequence at position j of 10_final/cache/<backbone>/index.jsonl (row = j; kinds rel, tgen, igen,
  probe, hn_i, hn_q, hn_x, benign).  The following lines: table = 'edit', kind = 'request': one line per edit of every edit stream of the cache, in
  stream order (sid '<family>:request:<position>'; row = null).
  image       path as in the benchmark (VLKEB: relative to $RESCOPE_EXT_VIEW/data/datasets/VLKEB/mmkb_images; MMKE-Bench: absolute); null for text-only rows
  image_md5   md5 of the image FILE (the image content identifier); null where there is no image
  question    the question string as the user asks it: no chat template, no answer
  question_key   question.strip().lower(): the identity under which questions are embedded and compared
  edit, edit_uid   edit sequences and requests: the edit (stream position, item uid); hard negatives: the edit whose image (HN-I, HN-X) or question (HN-Q)
              the row uses; edit_b, edit_b_uid (HN-X): the edit whose question the row uses
  How derived and checked (attributes_meta.json, field checks): from the data loaders the caches were built with (edit stream items, probe items,
  hard-negative and benign row files), compared with the cache index for EVERY sequence; md5 recomputed from every image file; EVERY prompt rendered
  with the backbone's tokenizer and prompt code: the rendered text is the template around exactly the question string and its token count (with the
  image slots) equals the captured prompt length of the cache.

backbone  sequences  edit requests  text rows  distinct image md5  distinct question keys           md5 of attributes.jsonl                                                                            edit streams
--------  ---------  -------------  ---------  ------------------  ----------------------  --------------------------------  --------------------------------------------------------------------------------------
qwen25vl      14300           2500       2500                7720                    5974  5f1c1c9e39c5a1468136b5d918567a28  Q: TB-Q2000-ae-l2_3 (2000 edits, VLKEB); QK: TB-Q500-mmke-l2_3 (500 edits, MMKE-Bench)
llava          8800           1000       2000                4556                    3958  39c9c1b08ca145760a8f07ca98812972                                                 L: TB-L1000-ae-l2_3 (1000 edits, VLKEB)

EMBEDDINGS
file                  shape                               md5             key list               md5 of the key list
----------------  ---------  --------------------------------  -------------------  --------------------------------
image_emb.npy     9165x1024  5ac1a8993557894c9cc02142fc4233b7     image_keys.jsonl  311c4596e0f014326c1b7615ecb80230
question_emb.npy   6424x384  ce5cc4cfb55b787bf1e017449baa90ea  question_keys.jsonl  673912223862f204fd1d1d8aa6423632
  3111 image contents and 3508 question keys occur in both caches; 302 image contents exist under several file paths (one row each).
  Unit norm: max | |e| - 1 | = 6.4e-08 (images), 9.0e-08 (questions).
  Storage form: x / |x| in float64, every component rounded to the nearest multiple of 2^-24, float32.  For two such rows the float64 dot product is
  EXACT (all partial sums are multiples of 2^-48 below 2), so a cosine score is a pure function of the two rows: the same bits in every process.

  IMAGES    CLIP vision tower of the local LLaVA-1.5 checkpoint, loaded alone: $RESCOPE_EXT_VIEW/data/models/llava-1.5-7b
            391 tensors with the prefix 'vision_tower.vision_model.' from ['model-00001-of-00003.safetensors'] (safetensors lazy access; nothing else is read)
            303507456 parameters, sha1 of the loaded weights 096330852f9ac2dcb254fa180b4cd04d7188d099; CLIPVisionModel built from the checkpoint's vision_config
            {'hidden_size': 1024, 'image_size': 336, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_hidden_layers': 24, 'patch_size': 14, 'projection_dim': 768} + defaults {'hidden_act': 'quick_gelu', 'layer_norm_eps': 1e-05, 'num_channels': 3}
            weights: float32 (the checkpoint's float16 values, cast exactly); eval mode; one image per forward; device of the build: GPU (cuda:0), TF32 off, cudnn deterministic
            decode: PIL.Image.open(path).convert('RGB'), no EXIF transpose, Image.MAX_IMAGE_PIXELS = None (as vendor_v7.vlm.load_image)
            image processor: transformers.models.clip.image_processing_clip.CLIPImageProcessor of the checkpoint's preprocessor_config.json
            convert RGB -> resize shortest edge 336 (bicubic, resample 3) -> centre crop 336 x 336 -> rescale 1/255 -> normalise (CLIP mean / std) -> float32 (1, 3, 336, 336)
            {'crop_size': {'height': 336, 'width': 336}, 'do_center_crop': True, 'do_convert_rgb': True, 'do_normalize': True, 'do_rescale': True, 'do_resize': True, 'image_mean': [0.48145466, 0.4578275, 0.40821073], 'image_processor_type': 'CLIPImageProcessor', 'image_std': [0.26862954, 0.26130258, 0.27577711], 'resample': 3, 'rescale_factor': 0.00392156862745098, 'size': {'shortest_edge': 336}}
            embedding: pooler_output = post_layernorm(CLS token of the last layer), 1024-d; no visual projection (not in the checkpoint)
  QUESTIONS all-MiniLM-L6-v2 (local): $RESCOPE_EXT_VIEW/data/models/all-MiniLM-L6-v2 through sentence-transformers (Transformer -> Pooling -> Normalize)
            model card: tokenizer (BERT, uncased) -> token embeddings -> mean pooling over the attention mask -> L2 normalisation
            max_seq_length 256 (longest question of the build: 29 tokens: nothing is truncated); one string per forward (no padding); float32, cpu
            input: the question key = question.strip().lower()
  versions  torch 2.8.0+cu128, transformers 5.17.0, PIL 12.3.0, numpy 2.4.6, torchvision 0.23.0+cu128, safetensors 0.8.0, sentence_transformers 6.1.0
  GPU memory of the image job (one process, B = 1): 1814 MiB (nvidia-smi), 1205 MiB allocated; gq run --mem 2000

REPRODUCIBILITY (feat2/retr/tests.py)
  attributes.jsonl, image_keys.jsonl, question_keys.jsonl: a re-run reproduces the files byte for byte (test repro).
  question_emb.npy: a complete re-run reproduces the matrix bit for bit (test reembed_questions_all); the model card's recipe through transformers
  agrees within one grid step.  image_emb.npy: 50 images re-embedded on the GPU are bit for bit the stored rows (test reembed_images_gpu); on the CPU
  (another device, float32) they differ by up to about 2e-6 per component (test reembed_images_cpu): rebuild on a GPU, the stored file is the reference.

REBUILD (python = $RESCOPE_PYTHON; cd 10_final/feat2/retr)
  python build_attributes.py qwen25vl; python build_attributes.py llava           CPU, seconds
  python embed.py keys                                                              CPU
  python embed.py questions                                                         CPU, 40 s, resumable parts
  $GQ run --mem 2000 --prio 4 --name retr_img -- python embed.py images             GPU, 4.5 min, resumable parts
  python embed.py assemble; python make_readme.py; python tests.py
NOT COVERED: the caches of the scout's logs (10_final/cache/QO: VLKEB one-hop family on Qwen; 10_final/cache/blip2) have no attribute table here.
