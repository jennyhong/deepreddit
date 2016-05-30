for hsz in 20 30 50
# for hsz in 20 30
do
  for lr in 0.0001 0.0005 0.001 0.005 0.01
  # for lr in 0.0001
  do
    for anneal in 1.0 1.2 1.5
    # for anneal in 1.0
    do
      for reg in 0.001 0.01 0.1 1.0
      # for reg in 0.01 0.1
      do
        python baseline_cmdline.py --hiddensize=$hsz --lr=$lr --annealby=$anneal --l2reg=$reg &
      done
    done
  done
done