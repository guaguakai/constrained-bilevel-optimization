# Experiment on the impact of ydim to the computation cost

ITER=10
FOLDER=exp2
EPS=0.01

for SEED in {1..5}
do
	for YDIM in {1..30}00
	do
		NCON=$((YDIM/5))
		for SOLVER in cvxpylayer ffo
		do
			sbatch --export=FOLDER=$FOLDER,SOLVER=$SOLVER,SEED=$SEED,YDIM=$YDIM,NCON=$NCON,ITER=$ITER,EPS=$EPS scripts/bilevel.sbatch
		done
	done
done
