source $HOME/rc/tensorflow/bin/activate
alias tfpython="LD_LIBRARY_PATH=\"$HOME/rc/my_libc_env/lib/x86_64-linux-gnu/:$HOME/rc/my_libc_env/usr/lib64/\" $HOME/rc/my_libc_env/lib/x86_64-linux-gnu/ld-2.17.so $HOME/rc/tensorflow/bin/python"

for hsz in 20 30 50
# for hsz in 20 30
do
  for lr in 0.0001 0.0005 0.001 0.005 0.01
  # for lr in 0.0001
  do
    for anneal in 1.0 1.2 1.5
    # for anneal in 1.0
    do
      # for reg in 0.001 0.01 0.1 1.0
      # for reg in 0.01 0.1
      # for reg in 0.001 # madmax3
      for reg in 0.01 # madmax
      do
        tfpython baseline_cmdline.py --hiddensize=$hsz --lr=$lr --annealby=$anneal --l2reg=$reg &
      done
    done
  done
done
