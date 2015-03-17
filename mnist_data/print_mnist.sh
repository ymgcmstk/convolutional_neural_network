cut -c $1 train-labels.txt
head -n $1 train-images.txt | tail -n 1 | awk '{ for (i = 1; i <= NF; i++) printf("%s%s", $i > 0 ? "◼︎" : "◻︎", i % 28 ? "" : "\n"); }'
