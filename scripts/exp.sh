SOLVER=ffo
SEED=0
YDIM=20
NCON=5

for SEED in {1..5}
do
	for YDIM in 5 10 20 50 100 200 300 400 500 600 700 800 900 1000
	do
		NCON=$((YDIM/5))
		for SOLVER in cvxpylayer, ffo
		do
			sbatch --export=SOLVER=$SOLVER,SEED=$SEED,YDIM=$YDIM,NCON=$NCON scripts/bilevel.sbatch
		done
	done
done
