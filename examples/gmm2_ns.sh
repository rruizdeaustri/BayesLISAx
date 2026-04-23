# From your project root (editable install already done)
# Fixed Dim
python -m sbi_samplers.cli \
  --algo ns \
  --problem gmm2 \
  --n 400 \
  --n-live 500 \
  --num-delete-ratio 0.3 \
  --num-inner-steps 6 (3*D) \
  --tol 3 \
  --seed 0


# TransD
python -m sbi_samplers.cli \
  --algo ns \
  --problem transdim \
  --family sbi_samplers.problems.gmm_family:GaussianMixtureFamily \
  --family-kwargs '{"max_K": 5, "n": 400, "sigma": 0.5, "lower": -10.0, "upper": 10.0}' \
  --n-live 1000 \
  --num-delete-ratio 0.3 \
  --tol 3 \
  --seed 0

python -m sbi_samplers.cli \
  --algo ns \
  --problem transdim \
  --family sbi_samplers.families.gmm_family:GaussianMixtureFamily \
  --family-kwargs '{"max_K":5, "n":400}' \
  --n-live 1000 \
  --num-delete-ratio 0.2 \
  --num-inner-steps 30 \
  --tol 0.5 \
  --n 1000 \
  --seed 123


