# Convergence analysis of different eps

ITER=1000
FOLDER=exp3

for SEED in {1..10}
do
	for YDIM in 100 200 500
	do
		NCON=$((YDIM/5))
		for SOLVER in ffo_complex # cvxpylayer ffo
		do
            		for EPS in 0.0001 0.001 0.01 0.1 1
			do
    				sbatch --export=FOLDER=$FOLDER,SOLVER=$SOLVER,SEED=$SEED,YDIM=$YDIM,NCON=$NCON,ITER=$ITER,EPS=$EPS scripts/bilevel.sbatch
			done
		done
	done
done
