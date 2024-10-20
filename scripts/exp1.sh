FOLDER=exp1
EPS=0.01
ITER=1000

for SEED in {1..10}
do
	for YDIM in  5 10 20 50 100 200
	do
		NCON=$((YDIM/5))
		for SOLVER in cvxpylayer ffo # ffo_complex
		do
			sbatch --export=FOLDER=$FOLDER,SOLVER=$SOLVER,SEED=$SEED,YDIM=$YDIM,NCON=$NCON,ITER=$ITER,EPS=$EPS scripts/bilevel.sbatch
		done
	done
done
