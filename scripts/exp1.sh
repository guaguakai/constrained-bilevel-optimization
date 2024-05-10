FOLDER=exp1

for SEED in 1  # {1..5}
do
	for YDIM in 500 600 700 800 900 1000 # 5 10 20 50 100 200 300 400 500 600 700 800 900 1000
	do
		NCON=$((YDIM/5))
		for SOLVER in ffo # cvxpylayer ffo
		do
			sbatch --export=FOLDER=$FOLDER,SOLVER=$SOLVER,SEED=$SEED,YDIM=$YDIM,NCON=$NCON scripts/bilevel.sbatch
		done
	done
done
