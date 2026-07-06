#!/usr/bin/env bash
# Phase-2 violence queue: sph_ot (novel) + FCF-P/E (authors' upstream, non-fatal) -> violence Q16 eval
# -> gallery. Core-6 (LSSE x2 / ODACE x3 / ESD) already trained. Each part non-fatal (§5).
# tmux: tmux new-session -d -s violfull 'bash eval/run_violence_full.sh'
set -u
cd /mnt/d/unlearning/SD_unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lsse
mkdir -p logs
ST=models/fcf/VIOLFULL_STATUS
: > "$ST"
echo "VIOLENCE FULL START $(date)" | tee -a "$ST"

run() {  # $1=label  $2=shell-string (non-fatal, tee, rc logged)
  local label=$1; shift
  echo "[$label] START $(date +%F_%T)" | tee -a "$ST"
  bash -c "$1" 2>&1 | tee "logs/violfull_${label}.log"
  local rc=${PIPESTATUS[0]}
  echo "[$label] rc=$rc END $(date +%F_%T)" | tee -a "$ST"
  return 0
}

# ---------------- sph_ot (SLERP-OT): N6 OT noise -> N5 spherical FCF-P ----------------
run sphot_otnoise "cd models/novel && python scripts/learn_ot_noise.py --explicit_file data/prompts/violence_explicit.txt --output outputs/ot_noise/violence_learned.json"
run sphot_train   "cd models/novel && python train.py --config configs/violence_v2.yaml --manifold spherical --ot_noise_file outputs/ot_noise/violence_learned.json"

# ---------------- FCF-P/E via AUTHORS' upstream (no trainer edits; eta 0.25 default = nudity) -------
FOUT=/mnt/d/unlearning/SD_unlearning/models/fcf/official_fcf_violence
mkdir -p "$FOUT"
run fcf_stage1 "cd /mnt/d/unlearning/FCF_upstream && python concept_forgetting_train.py --input_prompts data/train/violence.csv --save_path ${FOUT}/stage1.pt"
run fcf_p      "cd /mnt/d/unlearning/FCF_upstream && cp ${FOUT}/stage1.pt ${FOUT}/fcf_p.pt && python features_forgetting_P.py --model_path ${FOUT}/fcf_p.pt"
run fcf_e      "cd /mnt/d/unlearning/FCF_upstream && cp ${FOUT}/stage1.pt ${FOUT}/fcf_e.pt && python features_forgetting_E.py --model_path ${FOUT}/fcf_e.pt --experienxe_path experience.pth"
# convert .pt state_dict -> HF CLIPTextModel 'final' dir (best-effort, non-fatal)
CONV='import sys,torch; from transformers import CLIPTextModel;
pt,out=sys.argv[1],sys.argv[2];
m=CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4",subfolder="text_encoder");
sd=torch.load(pt,map_location="cpu"); sd=sd.get("state_dict",sd) if isinstance(sd,dict) else sd;
print("missing/unexpected:",m.load_state_dict(sd,strict=False)); m.save_pretrained(out); print("saved",out)'
run fcf_p_conv "[ -f ${FOUT}/fcf_p.pt ] && python -c '$CONV' ${FOUT}/fcf_p.pt models/fcf/official_fcf_p_violence/final || echo 'no fcf_p.pt'"
run fcf_e_conv "[ -f ${FOUT}/fcf_e.pt ] && python -c '$CONV' ${FOUT}/fcf_e.pt models/fcf/official_fcf_e_violence/final || echo 'no fcf_e.pt'"

# ---------------- violence Q16 eval (only keys whose output dir exists) ----------------
MODELS="lsse_r2q_a_violence,lsse_geo_e2_violence,odace_violence,odace_benign_violence,odace_benign_n1_violence,esd_u_violence"
[ -d models/novel/outputs/fcf_p_v2_violence/final ]   && MODELS="$MODELS,sph_ot_violence"
[ -d models/fcf/official_fcf_p_violence/final ]        && MODELS="$MODELS,fcf_p_violence"
[ -d models/fcf/official_fcf_e_violence/final ]        && MODELS="$MODELS,fcf_e_violence"
echo "EVAL MODELS = $MODELS" | tee -a "$ST"
run violence_eval "python models/fcf/eval_violence_q16.py --models '$MODELS'"

# ---------------- gallery rebuild ----------------
run gallery "python compare/build_live_gallery.py"

echo "VIOLENCE FULL ALL DONE $(date)" | tee -a "$ST"
touch models/fcf/VIOLFULL_DONE
